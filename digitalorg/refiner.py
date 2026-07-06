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

    def load(self) -> None:
        if self.model is not None:
            return
        repo = str(Path(self.cfg.sam3_repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from sam3.refine.sam3_refine_infer import build_sam3_inst_model, infer_one_sam3_with_refine

        self.model, self.processor = build_sam3_inst_model(
            bpe_path=self.cfg.sam3_bpe_path,
            checkpoint_path=self.cfg.sam3_checkpoint,
            device=self.cfg.device,
        )
        self._infer_fn = infer_one_sam3_with_refine

    def segment(
        self,
        image_path: str | Path,
        detections: list[Detection],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not detections:
            return [], []
        self.load()
        boxes = [d.bbox_xyxy for d in detections]
        return self._infer_fn(
            self.model,
            self.processor,
            str(image_path),
            boxes,
            self.cfg.refine_checkpoint,
            device=self.cfg.device,
            roi_pad=self.cfg.roi_pad,
        )
