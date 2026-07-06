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
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
