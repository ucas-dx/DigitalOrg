from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DigitalOrgConfig:
    digitalorgdet_repo: str = "${DIGITALORGDET_REPO}"
    sam3_repo: str = "${SAM3_REPO}"

    digitalorgdet_model: str = (
        "${DIGITALORGDET_MODEL}"
    )
    digitalorgdet_weight: str | None = None

    sam3_bpe_path: str = "${SAM3_BPE_PATH}"
    sam3_checkpoint: str = "${SAM3_CHECKPOINT}"
    refine_checkpoint: str | None = "${DIGITALORG_REFINE_CHECKPOINT}"

    device: str = "cuda:0"
    detect_conf: float = 0.001
    detect_iou: float = 0.7
    max_det: int = 3000
    min_box_area: float = 1.0
    roi_pad: float = 0.1
    mask_threshold: float = 0.5

    save_sam_masks: bool = False
    refine_backend: str = "legacy"
    fast_refine_precision: str = "fp32"
    fast_refine_exact: bool = True
    fast_refine_graphs: bool = True
    fast_refine_fused_roi: bool = False
    fast_refine_channels_last: bool = False
    fast_refine_max_instances: int = 100
    fast_refine_shared_image_kv: bool = False
    fast_refine_prefetch_depth: int = 2
    fast_refine_decode_workers: int = 2
    fast_refine_box_only: bool = False
    fast_refine_overlap: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.refine_backend not in ("legacy", "lowres_fast", "lowres_stream"):
            raise ValueError("refine_backend must be legacy, lowres_fast or lowres_stream")
        if self.fast_refine_precision not in ("fp32", "fp16"):
            raise ValueError("fast_refine_precision must be fp32 or fp16")
        if self.fast_refine_max_instances < 1:
            raise ValueError("fast_refine_max_instances must be positive")
        if self.fast_refine_prefetch_depth < 1 or self.fast_refine_decode_workers < 1:
            raise ValueError("prefetch depth and decode workers must be positive")
        if self.fast_refine_box_only or self.fast_refine_overlap:
            if self.refine_backend != "lowres_stream" or not self.fast_refine_exact:
                raise ValueError("Structural optimization requires lowres_stream with exact mode")
        if self.fast_refine_overlap and not self.fast_refine_box_only:
            raise ValueError("Overlap requires the box-only structural adapter")
        if self.refine_backend != "legacy" and self.fast_refine_exact:
            if self.fast_refine_precision != "fp32" or self.fast_refine_fused_roi or self.fast_refine_channels_last or self.fast_refine_shared_image_kv:
                raise ValueError("Exact mode requires FP32, original interpolation/layout and no shared KV projections")
        self._expand_paths()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DigitalOrgConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Backward compatibility for early DigitalOrg configs.
        legacy_model_key = "yo" + "lo_world_model"
        legacy_weight_key = "yo" + "lo_world_weight"
        if legacy_model_key in data and "digitalorgdet_model" not in data:
            data["digitalorgdet_model"] = data.pop(legacy_model_key)
        if legacy_weight_key in data and "digitalorgdet_weight" not in data:
            data["digitalorgdet_weight"] = data.pop(legacy_weight_key)
        legacy_repo_key = "org" + "line_repo"
        if legacy_repo_key in data and "digitalorgdet_repo" not in data:
            data["digitalorgdet_repo"] = data.pop(legacy_repo_key)
        known = {k: data.pop(k) for k in list(data.keys()) if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg._expand_paths()
        cfg.extra.update(data)
        return cfg

    def _expand_paths(self) -> None:
        for key in (
            "digitalorgdet_repo",
            "sam3_repo",
            "digitalorgdet_model",
            "digitalorgdet_weight",
            "sam3_bpe_path",
            "sam3_checkpoint",
            "refine_checkpoint",
        ):
            value = getattr(self, key)
            if isinstance(value, str):
                setattr(self, key, os.path.expandvars(os.path.expanduser(value)))

    def to_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extra"}
        out.update(self.extra)
        return out

    @property
    def detector_device(self) -> str:
        if self.device.startswith("cuda:"):
            return self.device.split(":", 1)[1]
        if self.device == "cuda":
            return "0"
        return self.device
