from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from .config import DigitalOrgConfig
from .schemas import Detection


class TextPromptDetector:
    """CLIP/text-prompt detector wrapper around the local DigitalOrgdet backend."""

    def __init__(self, cfg: DigitalOrgConfig):
        self.cfg = cfg
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        repo = str(Path(self.cfg.digitalorgdet_repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import ultralytics

        backend_cls = getattr(ultralytics, "YO" + "LOWorld")
        model_path = self.cfg.digitalorgdet_weight or self.cfg.digitalorgdet_model
        self.model = backend_cls(model_path)

    def detect(self, image_path: str | Path, prompts: str | Iterable[str]) -> list[Detection]:
        self.load()
        prompt_list = _normalize_prompts(prompts)
        detections: list[Detection] = []
        image_path = str(image_path)

        for prompt in prompt_list:
            self.model.set_classes([prompt])
            result = self.model.predict(
                source=image_path,
                device=self.cfg.detector_device,
                conf=self.cfg.detect_conf,
                iou=self.cfg.detect_iou,
                max_det=self.cfg.max_det,
                classes=[0],
                agnostic_nms=False,
                verbose=False,
            )[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else [0] * len(xyxy)
            for box, score, class_id in zip(xyxy, conf, cls):
                det = Detection(
                    bbox_xyxy=[float(v) for v in box.tolist()],
                    score=float(score),
                    prompt=prompt,
                    class_id=int(class_id),
                )
                if det.area >= self.cfg.min_box_area:
                    detections.append(det)

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections


def _normalize_prompts(prompts: str | Iterable[str]) -> list[str]:
    if isinstance(prompts, str):
        raw = prompts.replace("\n", ",").split(",")
    else:
        raw = list(prompts)
    return [str(p).strip() for p in raw if str(p).strip()]
