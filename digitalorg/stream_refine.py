"""Opt-in box-only SAM3 feature reuse and bounded inference data pipeline.

This module never reuses image features across distinct frames. The only shared
attention projections are in the first SAM two-way block, before image tokens
become prompt-dependent. Refine self-attention is left unchanged.
"""
from __future__ import annotations

import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .fast_refine import FastSam3Refiner, boxes_on_cpu, crop_batch, paste_binary


def _attend(module, q, k, v):
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    return module.out_proj(module._recombine_heads(out))


def _image_kv(self, q, k, v):
    cache = self._active_image_cache
    if "kv" not in cache:
        cache["kv"] = (self._separate_heads(self.k_proj(k[:1]), self.num_heads),
                       self._separate_heads(self.v_proj(v[:1]), self.num_heads))
    kh, vh = cache["kv"]
    qh = self._separate_heads(self.q_proj(q), self.num_heads)
    return _attend(self, qh, kh.expand(len(q), -1, -1, -1), vh.expand(len(q), -1, -1, -1))


def _image_q(self, q, k, v):
    cache = self._active_image_cache
    if "q" not in cache:
        cache["q"] = self._separate_heads(self.q_proj(q[:1]), self.num_heads)
    kh = self._separate_heads(self.k_proj(k), self.num_heads)
    vh = self._separate_heads(self.v_proj(v), self.num_heads)
    return _attend(self, cache["q"].expand(len(k), -1, -1, -1), kh, vh)


def _decode_cpu(image, boxes, roi_pad):
    big, small = boxes_on_cpu(boxes, roi_pad)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if isinstance(image, (str, Path)):
        with Image.open(image) as src:
            image = src.convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("image must be a path or PIL image")
    if image.size != (1024, 1024):
        raise ValueError("Streaming backend requires 1024 x 1024 images and matching boxes")
    cpu = torch.empty((3, 1024, 1024), dtype=torch.uint8, pin_memory=True)
    cpu.copy_(torch.from_numpy(np.array(image)).permute(2, 0, 1))
    return cpu, boxes, big, small


@dataclass
class EncodedFrame:
    """Explicit immutable-image handle. Discard it when the image/model changes."""
    owner: object
    rgb: torch.Tensor
    maps: list[torch.Tensor]
    ready: torch.cuda.Event
    projections: dict = field(default_factory=dict)


@dataclass
class _PendingMasks:
    cpu: torch.Tensor
    ready: torch.cuda.Event

    def result(self):
        self.ready.synchronize()
        # Returned arrays own references to this unique pinned allocation.
        return list(self.cpu.numpy())


class StreamingSam3Refiner(FastSam3Refiner):
    """1024-square box inference with explicit cache and ordered streaming APIs.

    SAM and Refine weights must be immutable, resident and in eval mode.
    CUDA work is serialized on one owned compute stream; upload and download
    streams overlap transfers. CPU image decoding is bounded by prefetch_depth.
    """
    def __init__(self, *args, shared_image_kv=True, **kwargs):
        if kwargs.get("exact", False) and shared_image_kv:
            raise ValueError("Exact mode forbids shared projection re-batching")
        super().__init__(*args, **kwargs)
        if self.device.type != "cuda":
            raise ValueError("Streaming backend requires CUDA")
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.upload_stream = torch.cuda.Stream(device=self.device)
        self.download_stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        self.compute_stream.wait_stream(current)
        self.upload_stream.wait_stream(current)
        self._owner = object()
        self._dense_pe = None
        self.shared_image_kv = shared_image_kv
        # A private decoder avoids altering the caller's SAM model.
        decoder = self.model.inst_interactive_predictor.model.sam_mask_decoder
        self.decoder = copy.deepcopy(decoder).eval() if shared_image_kv else decoder
        self.compute_stream.wait_stream(current)
        if shared_image_kv:
            l0 = self.decoder.transformer.layers[0]
            for attn in (l0.cross_attn_token_to_image, l0.cross_attn_image_to_token):
                if attn.use_fa3:
                    raise ValueError("Shared projection backend has only been validated with PyTorch SDPA")
            l0.cross_attn_token_to_image.forward = MethodType(_image_kv, l0.cross_attn_token_to_image)
            l0.cross_attn_image_to_token.forward = MethodType(_image_q, l0.cross_attn_image_to_token)

    def _encode_cpu(self, cpu):
        # Called with the lock held and the compute stream selected.
        with torch.cuda.stream(self.upload_stream):
            raw = cpu.to(self.device, non_blocking=True)
            uploaded = torch.cuda.Event()
            uploaded.record()
        self.compute_stream.wait_event(uploaded)
        raw.record_stream(self.compute_stream)
        state = self.processor.set_image(raw)
        predictor = self.model.inst_interactive_predictor
        tracker = predictor.model
        _, feats, _, _ = tracker._prepare_backbone_features(state["backbone_out"]["sam2_backbone_out"])
        feats[-1] = feats[-1] + tracker.no_mem_embed
        maps = [f.permute(1, 2, 0).view(1, -1, *s)
                for f, s in zip(feats[::-1], predictor._bb_feat_sizes[::-1])][::-1]
        rgb = raw[None].float()
        ready = torch.cuda.Event()
        ready.record()
        return EncodedFrame(self._owner, rgb, maps, ready)

    @torch.no_grad()
    def encode_image(self, image):
        """Encode once for repeated box queries on exactly the same image."""
        cpu, _, _, _ = _decode_cpu(image, [], self.roi_pad)
        with self._lock, torch.cuda.stream(self.compute_stream), torch.autocast("cuda", enabled=False):
            return self._encode_cpu(cpu)

    def _decode_boxes(self, frame, boxes):
        tracker = self.model.inst_interactive_predictor.model
        predictor = self.model.inst_interactive_predictor
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
        coords = predictor._transforms.transform_boxes(boxes_t, normalize=True, orig_hw=(1024, 1024))
        labels = torch.tensor([[2, 3]], dtype=torch.int, device=self.device).repeat(len(boxes), 1)
        sparse, dense = tracker.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        if self._dense_pe is None:
            self._dense_pe = tracker.sam_prompt_encoder.get_dense_pe()
        if self.shared_image_kv:
            l0 = self.decoder.transformer.layers[0]
            l0.cross_attn_token_to_image._active_image_cache = frame.projections
            l0.cross_attn_image_to_token._active_image_cache = frame.projections
        try:
            low, _, _, _ = self.decoder(
                image_embeddings=frame.maps[-1], image_pe=self._dense_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False, repeat_image=len(boxes) > 1, high_res_features=frame.maps[:-1])
        finally:
            if self.shared_image_kv:
                l0.cross_attn_token_to_image._active_image_cache = None
                l0.cross_attn_image_to_token._active_image_cache = None
        return low

    def _masks_gpu(self, frame, boxes, big, small, sam_only=False, return_coarse=False):
        if frame.owner is not self._owner:
            raise ValueError("Encoded frame belongs to a different model instance")
        if self.exact and len(boxes) > self.max_instances:
            raise ValueError("Exact mode refuses implicit re-batching; use the reference batch size")
        self.compute_stream.wait_event(frame.ready)
        outputs, coarse = [], []
        for start in range(0, len(big), self.max_instances):
            stop = start + self.max_instances
            logits = self._decode_boxes(frame, boxes[start:stop])
            if sam_only or return_coarse:
                full = F.interpolate(logits, (1024, 1024), mode="bilinear", align_corners=False)
                masks = ((full[:, 0] > self.model.inst_interactive_predictor.mask_threshold).byte() * 255)
                if sam_only:
                    outputs.append(masks)
                    continue
                coarse.append(masks)
            probs = F.interpolate(logits.sigmoid(), (256, 256), mode="bilinear", align_corners=False)
            if self.fused_roi:
                bt = torch.as_tensor(big[start:stop], device=self.device)
                st = torch.as_tensor(small[start:stop], device=self.device)
                rgb, prompt = self._crop_cuda(frame.rgb, probs, bt, st)
            else:
                rgb, prompt = crop_batch(frame.rgb, probs, big[start:stop], small[start:stop])
            refined = self.runner(rgb, prompt)
            outputs.append(self._paste_cuda(refined, bt, self.threshold) if self.fused_roi
                           else paste_binary(refined, big[start:stop], self.threshold))
        def combine(tensors):
            if not tensors:
                return torch.empty((0, 1024, 1024), dtype=torch.uint8, device=self.device)
            return tensors[0] if len(tensors) == 1 else torch.cat(tensors)
        return combine(outputs), combine(coarse)

    def _download(self, masks):
        computed = torch.cuda.Event()
        computed.record(self.compute_stream)
        cpu = torch.empty(masks.shape, dtype=masks.dtype, pin_memory=True)
        with torch.cuda.stream(self.download_stream):
            self.download_stream.wait_event(computed)
            cpu.copy_(masks, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
        masks.record_stream(self.download_stream)
        return _PendingMasks(cpu, ready)

    @torch.no_grad()
    def segment_encoded(self, frame, boxes, *, sam_only=False, return_coarse=False):
        """Return full CPU masks, reusing only this handle's immutable image."""
        big, small = boxes_on_cpu(boxes, self.roi_pad)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        with self._lock, torch.cuda.stream(self.compute_stream), torch.autocast("cuda", enabled=False):
            out, coarse = self._masks_gpu(frame, boxes, big, small, sam_only, return_coarse)
            pending = self._download(out)
            prior = self._download(coarse) if return_coarse else None
        return (prior.result(), pending.result()) if prior else pending.result()

    @torch.no_grad()
    def segment(self, image, boxes, return_coarse=False, *, sam_only=False):
        cpu, boxes, big, small = _decode_cpu(image, boxes, self.roi_pad)
        if not len(boxes):
            return ([], []) if return_coarse else []
        with self._lock, torch.cuda.stream(self.compute_stream), torch.autocast("cuda", enabled=False):
            frame = self._encode_cpu(cpu)
            out, coarse = self._masks_gpu(frame, boxes, big, small, sam_only, return_coarse)
            pending = self._download(out)
            prior = self._download(coarse) if return_coarse else None
        return (prior.result(), pending.result()) if prior else pending.result()

    def segment_many(self, requests, *, prefetch_depth=2, workers=2, sam_only=False):
        """Yield ordered mask lists for an iterable of (image, boxes) pairs.

        Each image is decoded and encoded afresh. The bounded lookahead allows
        CPU decoding and output DMA to overlap GPU inference for other images.
        Close the generator if stopping early. Detector and file export are not
        included; this API accepts boxes already expressed in 1024 coordinates.
        """
        if prefetch_depth < 1 or workers < 1:
            raise ValueError("prefetch_depth and workers must be positive")
        requests = iter(requests)
        inputs, outputs = deque(), deque()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="organoid-decode") as pool:
            def submit_one():
                try:
                    image, boxes = next(requests)
                except StopIteration:
                    return False
                # Snapshot mutable box lists before handing them to a worker.
                boxes = np.asarray(boxes).copy()
                if isinstance(image, Image.Image):
                    image = image.copy()
                inputs.append(pool.submit(_decode_cpu, image, boxes, self.roi_pad))
                return True
            try:
                for _ in range(prefetch_depth):
                    if not submit_one():
                        break
                while inputs:
                    cpu, boxes, big, small = inputs.popleft().result()
                    submit_one()
                    with torch.no_grad(), self._lock, torch.cuda.stream(self.compute_stream), torch.autocast("cuda", enabled=False):
                        if len(boxes):
                            frame = self._encode_cpu(cpu)
                            out, _ = self._masks_gpu(frame, boxes, big, small, sam_only=sam_only)
                        else:
                            out = torch.empty((0, 1024, 1024), dtype=torch.uint8, device=self.device)
                        outputs.append(self._download(out))
                    if len(outputs) >= prefetch_depth:
                        yield outputs.popleft().result()
                while outputs:
                    yield outputs.popleft().result()
            finally:
                for future in inputs:
                    future.cancel()
                # Pending host copies must finish even on early close/error.
                self.upload_stream.synchronize()
                self.compute_stream.synchronize()
                self.download_stream.synchronize()
