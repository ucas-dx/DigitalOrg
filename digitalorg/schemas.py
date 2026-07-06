from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Detection:
    bbox_xyxy: list[float]
    score: float
    prompt: str
    class_id: int = 0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["area"] = self.area
        return d


@dataclass
class InstanceResult:
    instance_id: int
    bbox_xyxy: list[float]
    score: float
    prompt: str
    mask_path: str | None = None
    sam_mask_path: str | None = None
    area: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
