from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .config import DigitalOrgConfig
from .schemas import Detection


class Sam3BoxRefiner:
    """SAM3 geometric prompt segmentation + ROIRefineNet post-refine."""

    def __init__(self, cfg: DigitalOrgConfig):
        self.cfg = cfg
        self.model = None
        self.processor = None
        self._infer_fn = None
        self._fast = None

    def load(self) -> None:
        if self.model is not None:
            return
        repo = str(Path(self.cfg.sam3_repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from sam3.refine.sam3_refine_infer import build_sam3_inst_model, infer_one_sam3_with_refine

        import torch
        previous_autocast = torch.is_autocast_enabled("cuda")
        try:
            self.model, self.processor = build_sam3_inst_model(
                bpe_path=self.cfg.sam3_bpe_path,
                checkpoint_path=self.cfg.sam3_checkpoint,
                device=self.cfg.device,
            )
        finally:
            if self.cfg.refine_backend != "legacy":
                torch.set_autocast_enabled("cuda", previous_autocast)
        self._infer_fn = infer_one_sam3_with_refine
        if self.cfg.refine_backend != "legacy":
            try:
                from sam3.refine.util import ROIRefineNet
                from .fast_refine import FastSam3Refiner
                if not self.cfg.refine_checkpoint:
                    raise ValueError("The fast backend requires a trained RefineNet checkpoint")
                checkpoint = torch.load(self.cfg.refine_checkpoint, map_location="cpu", weights_only=True)
                weights = checkpoint.get("refine", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
                refine = ROIRefineNet(variant="atto", convnext_ckpt=None, use_prompt=True, build_prompt=False)
                refine.load_state_dict(weights, strict=True)
                refine.to(self.cfg.device).eval()
                self.model.to(self.cfg.device).eval()
                self.processor.device = torch.device(self.cfg.device)
                backend = FastSam3Refiner
                extra = {}
                if self.cfg.refine_backend == "lowres_stream":
                    from .stream_refine import StreamingSam3Refiner
                    backend = StreamingSam3Refiner
                    extra["shared_image_kv"] = self.cfg.fast_refine_shared_image_kv
                    if self.cfg.fast_refine_box_only:
                        from .structural_refine import StructuralSam3Refiner
                        backend = StructuralSam3Refiner
                        extra["overlap"] = self.cfg.fast_refine_overlap
                self._fast = backend(
                    self.model, self.processor, refine, use_graphs=self.cfg.fast_refine_graphs,
                    exact=self.cfg.fast_refine_exact,
                    precision=self.cfg.fast_refine_precision, channels_last=self.cfg.fast_refine_channels_last,
                    fused_roi=self.cfg.fast_refine_fused_roi, roi_pad=self.cfg.roi_pad,
                    max_instances=self.cfg.fast_refine_max_instances, threshold=self.cfg.mask_threshold, **extra)
            except Exception:
                self.model = self.processor = None
                raise

    def segment(
        self,
        image_path: str | Path,
        detections: list[Detection],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not detections:
            return [], []
        self.load()
        boxes = [d.bbox_xyxy for d in detections]
        if self._fast is not None:
            result = self._fast.segment(image_path, boxes, return_coarse=self.cfg.save_sam_masks)
            return result if self.cfg.save_sam_masks else ([], result)
        return self._infer_fn(
            self.model,
            self.processor,
            str(image_path),
            boxes,
            self.cfg.refine_checkpoint,
            device=self.cfg.device,
            roi_pad=self.cfg.roi_pad,
        )

    def segment_many(self, requests):
        """Yield (SAM masks, refined masks) for pre-detected image requests.

        The streaming backend overlaps decode, GPU inference and host transfer.
        Coarse-mask export and legacy backends use the sequential API instead.
        """
        if self.cfg.refine_backend != "lowres_stream" or self.cfg.save_sam_masks:
            for image_path, detections in requests:
                yield self.segment(image_path, detections)
            return
        self.load()
        pairs = ((path, [d.bbox_xyxy for d in detections]) for path, detections in requests)
        results = self._fast.segment_many(
            pairs, prefetch_depth=self.cfg.fast_refine_prefetch_depth,
            workers=self.cfg.fast_refine_decode_workers)
        try:
            for masks in results:
                yield [], masks
        finally:
            results.close()
