"""Opt-in low-resolution SAM3 refinement for 1024-pixel square inputs.

Weights, ROI size and interpolation geometry match the FP32 low-resolution
reference. This is not equivalent to the released full-mask SAM3 wrapper,
which uses different prompts and extra connected-component postprocessing.
"""
from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def boxes_on_cpu(boxes, roi_pad=0.1):
    b = np.asarray(boxes, dtype=np.float64)
    if b.size == 0:
        return np.empty((0, 4), np.int64), np.empty((0, 4), np.int64)
    if b.ndim != 2 or b.shape[1] != 4 or not np.isfinite(b).all():
        raise ValueError("Expected finite Nx4 XYXY boxes")
    if not np.isfinite(roi_pad) or roi_pad < 0:
        raise ValueError("roi_pad must be finite and nonnegative")
    if np.any(b < 0) or np.any(b > 1024) or np.any(b[:, 2:] <= b[:, :2]):
        raise ValueError("Boxes must have positive area and lie in the 1024 image")
    b = b.astype(np.int64)
    pad = np.rint(np.maximum(b[:, 2:] - b[:, :2] + 1, 1) * roi_pad).astype(np.int64)
    b[:, :2] = np.clip(b[:, :2] - pad, 0, 1023)
    b[:, 2:] = np.maximum(np.clip(b[:, 2:] + pad + 1, 1, 1024), b[:, :2] + 1)
    small = np.concatenate([b[:, :2] // 4, np.minimum(256, (b[:, 2:] + 3) // 4)], axis=1)
    return b, small


def crop_batch(rgb, probabilities, boxes, small_boxes):
    images, prompts = [], []
    for i, (b, s) in enumerate(zip(boxes, small_boxes)):
        x1, y1, x2, y2 = b
        images.append(F.interpolate(rgb[..., y1:y2, x1:x2], size=(256, 256), mode="bilinear", align_corners=False))
        x1, y1, x2, y2 = s
        prompts.append(F.interpolate(probabilities[i:i+1, :, y1:y2, x1:x2], size=(256, 256), mode="bilinear", align_corners=False))
    return torch.cat(images), torch.cat(prompts)


def paste_binary(probabilities, boxes, threshold=.5):
    # Threshold after each ROI interpolation. Outside each ROI is exactly zero,
    # so no float32 full-image canvas or subsequent no-op resize is necessary.
    canvas = torch.zeros((len(boxes), 1024, 1024), dtype=torch.uint8, device=probabilities.device)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        roi = F.interpolate(probabilities[i:i+1], size=(y2-y1, x2-x1), mode="bilinear", align_corners=False)
        canvas[i, y1:y2, x1:x2] = (roi[0, 0] > threshold).to(torch.uint8) * 255
    return canvas


def _prompt_probability_forward(self, mask_bchw_256, roi_rgb_bchw):
    """Specialization for validated soft probabilities: no GPU scalar branches."""
    m64 = self.mask_proj_spatial(F.interpolate(mask_bchw_256, size=(64, 64), mode="bilinear", align_corners=False))
    r64 = F.interpolate(roi_rgb_bchw.float() / 255.0, size=(64, 64), mode="bilinear", align_corners=False)
    i64 = self.img_proj_spatial(r64)
    mtok = self.mask_to_tok(m64).flatten(2).transpose(1, 2)
    itok = self.img_to_tok(i64).flatten(2).transpose(1, 2)
    if self.pos_tok.shape[1] != mtok.shape[1]:
        raise ValueError("Fast path requires the pretrained 64-pixel prompt layout")
    tok = mtok + itok + self.pos_tok
    mem = self.mem_tokens.expand(mtok.shape[0], -1, -1)
    tok_out = self.encoder(torch.cat([mem, tok], dim=1))[:, mem.size(1):]
    tok_map = tok_out.transpose(1, 2).reshape(mtok.shape[0], mtok.shape[2], 64 // self.patch, 64 // self.patch)
    return self.to_out(self.refine_out(self.tok_to_map(tok_map))) * self.gate


class RefineRunner:
    """Bounded CUDA graph cache; every replay receives fresh ROI tensors."""
    def __init__(self, refine, use_graphs=True, max_graphs=4, channels_last=False, precision="fp32"):
        if refine.training:
            raise ValueError("RefineNet must be in eval mode")
        self.model = copy.deepcopy(refine).eval()
        self.model.prompt_encoder.forward = MethodType(_prompt_probability_forward, self.model.prompt_encoder)
        self.channels_last = channels_last
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16 or bf16")
        self.precision = precision
        if channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.use_graphs = use_graphs
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        self.max_graphs = max_graphs
        self.graphs = OrderedDict()

    def compute(self, rgb, prompt):
        with torch.autocast(device_type=rgb.device.type, enabled=self.precision != "fp32",
                            dtype=torch.float16 if self.precision == "fp16" else torch.bfloat16,
                            cache_enabled=False):
            return self.model(rgb, prompt_mask_bchw_256=prompt).float().sigmoid()

    @torch.no_grad()
    def __call__(self, rgb, prompt):
        if rgb.shape[1:] != (3, 256, 256) or prompt.shape != (len(rgb), 1, 256, 256):
            raise ValueError("Expected paired Nx3x256x256 RGB and Nx1x256x256 probability tensors")
        if not self.use_graphs or not rgb.is_cuda:
            return self.compute(rgb.contiguous(memory_format=torch.channels_last) if self.channels_last else rgb, prompt)
        key = (rgb.shape[0], rgb.device, rgb.dtype)
        if key not in self.graphs:
            x = torch.empty_like(rgb, memory_format=torch.channels_last if self.channels_last else torch.contiguous_format)
            p = torch.empty_like(prompt)
            x.copy_(rgb)
            p.copy_(prompt)
            stream = torch.cuda.Stream(device=rgb.device)
            stream.wait_stream(torch.cuda.current_stream(rgb.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.compute(x, p)
            torch.cuda.current_stream(rgb.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self.compute(x, p)
            self.graphs[key] = (graph, x, p, out)
            while len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        self.graphs.move_to_end(key)
        graph, x, p, out = self.graphs[key]
        x.copy_(rgb)
        p.copy_(prompt)
        graph.replay()
        return out


@torch.no_grad()
def raw_sam_logits(model, state, boxes):
    """Use SAM3's prompt encoder and decoder without mask postprocessing."""
    predictor = model.inst_interactive_predictor
    tracker = predictor.model
    _, feats, _, _ = tracker._prepare_backbone_features(state["backbone_out"]["sam2_backbone_out"])
    feats[-1] = feats[-1] + tracker.no_mem_embed
    maps = [f.permute(1, 2, 0).view(1, -1, *size)
            for f, size in zip(feats[::-1], predictor._bb_feat_sizes[::-1])][::-1]
    boxes_t = torch.as_tensor(np.asarray(boxes), dtype=torch.float32, device=maps[-1].device)
    coords = predictor._transforms.transform_boxes(boxes_t, normalize=True,
        orig_hw=(state["original_height"], state["original_width"]))
    labels = torch.tensor([[2, 3]], dtype=torch.int, device=coords.device).repeat(len(boxes), 1)
    sparse, dense = tracker.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    low, _, _, _ = tracker.sam_mask_decoder(
        image_embeddings=maps[-1], image_pe=tracker.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False, repeat_image=len(boxes) > 1, high_res_features=maps[:-1])
    return low


class FastSam3Refiner:
    """Resident, opt-in 1024-square inference. FP32 is the default.

    The passed SAM model and processor must already be loaded on one device.
    A first call with a new ROI count captures a CUDA graph and is slower.
    Calls on this instance are serialized because graph buffers are reused.
    """
    def __init__(self, model, processor, refine, use_graphs=True, roi_pad=.1, fused_roi=False, channels_last=False, precision="fp32", max_instances=100, threshold=.5, exact=False):
        if exact and (precision != "fp32" or fused_roi or channels_last):
            raise ValueError("Exact mode forbids mixed precision, fused interpolation and channels-last")
        if max_instances < 1 or not 0 < threshold < 1:
            raise ValueError("max_instances must be positive and threshold must lie between zero and one")
        self.model, self.processor = model, processor
        self.device = next(refine.parameters()).device
        self.runner = RefineRunner(refine, use_graphs=use_graphs, channels_last=channels_last, precision=precision)
        self.roi_pad = roi_pad
        self.max_instances, self.threshold = max_instances, threshold
        self.fused_roi = fused_roi
        self.exact = exact
        if fused_roi:
            if self.device.type != "cuda":
                raise ValueError("Fused ROI kernels require CUDA and Triton")
            from ._roi_kernels import crop_cuda, paste_cuda
            self._crop_cuda, self._paste_cuda = crop_cuda, paste_cuda
        self._lock = threading.Lock()

    @torch.no_grad()
    def segment(self, image, boxes, return_coarse=False):
        with self._lock, torch.autocast(device_type=self.device.type, enabled=False):
            if isinstance(image, (str, Path)):
                with Image.open(image) as source:
                    image = source.convert("RGB")
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                raise TypeError("image must be a path or PIL image")
            if image.size != (1024, 1024):
                raise ValueError("Fast path requires a 1024 x 1024 image and matching boxes")
            big, small = boxes_on_cpu(boxes, self.roi_pad)
            if self.exact and len(big) > self.max_instances:
                raise ValueError("Exact mode refuses implicit re-batching; use the reference batch size")
            if len(big) == 0:
                return ([], []) if return_coarse else []
            state = self.processor.set_image(image)
            rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].to(device=self.device, dtype=torch.float32)
            results, coarse = [], []
            for start in range(0, len(big), self.max_instances):
                stop = start + self.max_instances
                logits = raw_sam_logits(self.model, state, boxes[start:stop])
                probs = logits.sigmoid()
                if probs.shape[-2:] != (256, 256):
                    probs = F.interpolate(probs, size=(256, 256), mode="bilinear", align_corners=False)
                if self.fused_roi:
                    big_t = torch.as_tensor(big[start:stop], device=self.device)
                    small_t = torch.as_tensor(small[start:stop], device=self.device)
                    rois, prompts = self._crop_cuda(rgb, probs, big_t, small_t)
                else:
                    rois, prompts = crop_batch(rgb, probs, big[start:stop], small[start:stop])
                refined = self.runner(rois, prompts)
                masks = (self._paste_cuda(refined, big_t, self.threshold) if self.fused_roi
                         else paste_binary(refined, big[start:stop], self.threshold)).cpu().numpy()
                results.extend(masks)
                if return_coarse:
                    m = F.interpolate(logits, size=(1024, 1024), mode="bilinear", align_corners=False)
                    coarse.extend(((m[:, 0] > self.model.inst_interactive_predictor.mask_threshold).byte() * 255).cpu().numpy())
            return (coarse, results) if return_coarse else results
