from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .schemas import InstanceResult


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_mask(path: str | Path, mask: np.ndarray, threshold: float = 0.5) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype != np.uint8:
        mask = (mask > threshold).astype(np.uint8) * 255
    else:
        mask = ((mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(path), mask)
    return int((mask > 0).sum())


def draw_overlay(image_path: str | Path, instances: list[InstanceResult], output_path: str | Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    overlay = image.copy()
    colors = _palette(max(1, len(instances)))
    for i, inst in enumerate(instances):
        color = colors[i]
        if inst.mask_path:
            mask = cv2.imread(inst.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = [int(round(v)) for v in inst.bbox_xyxy]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{inst.instance_id}:{inst.prompt} {inst.score:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), overlay)


def _palette(n: int) -> list[tuple[int, int, int]]:
    base = [
        (0, 0, 255),
        (0, 180, 0),
        (255, 0, 0),
        (0, 180, 180),
        (180, 0, 180),
        (180, 180, 0),
    ]
    return [base[i % len(base)] for i in range(n)]
