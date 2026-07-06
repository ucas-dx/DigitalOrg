from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import DigitalOrgConfig
from .detector import TextPromptDetector
from .io import draw_overlay, ensure_dir, save_json, save_mask
from .refiner import Sam3BoxRefiner
from .schemas import InstanceResult


class DigitalOrgPipeline:
    """Complete DigitalOrg image inference pipeline."""

    def __init__(self, cfg: DigitalOrgConfig):
        self.cfg = cfg
        self.detector = TextPromptDetector(cfg)
        self.refiner = Sam3BoxRefiner(cfg)

    @classmethod
    def from_config_file(cls, path: str | Path) -> "DigitalOrgPipeline":
        return cls(DigitalOrgConfig.from_yaml(path))

    def predict_image(
        self,
        image_path: str | Path,
        prompts: str | Iterable[str],
        output_dir: str | Path | None = None,
        save_outputs: bool = True,
    ) -> dict:
        image_path = Path(image_path)
        detections = self.detector.detect(image_path, prompts)
        sam_masks, refine_masks = self.refiner.segment(image_path, detections)

        out_dir = ensure_dir(output_dir) if output_dir else None
        instances: list[InstanceResult] = []
        for idx, det in enumerate(detections):
            mask_path = None
            sam_mask_path = None
            area = None
            if save_outputs and out_dir is not None:
                mask_path = str(out_dir / "masks" / f"instance_{idx:04d}.png")
                area = save_mask(mask_path, refine_masks[idx], self.cfg.mask_threshold)
                if self.cfg.save_sam_masks:
                    sam_mask_path = str(out_dir / "sam_masks" / f"instance_{idx:04d}.png")
                    save_mask(sam_mask_path, sam_masks[idx], self.cfg.mask_threshold)

            instances.append(
                InstanceResult(
                    instance_id=idx,
                    bbox_xyxy=det.bbox_xyxy,
                    score=det.score,
                    prompt=det.prompt,
                    mask_path=mask_path,
                    sam_mask_path=sam_mask_path,
                    area=area,
                )
            )

        result = {
            "image_path": str(image_path),
            "prompts": [d.prompt for d in detections],
            "num_detections": len(detections),
            "detections": [d.to_dict() for d in detections],
            "instances": [i.to_dict() for i in instances],
        }

        if save_outputs and out_dir is not None:
            draw_overlay(image_path, instances, out_dir / "overlay.jpg")
            result["output_dir"] = str(out_dir)
            result["overlay_path"] = str(out_dir / "overlay.jpg")
            save_json(out_dir / "result.json", result)
            save_json(out_dir / "detections.json", [d.to_dict() for d in detections])
        return result
