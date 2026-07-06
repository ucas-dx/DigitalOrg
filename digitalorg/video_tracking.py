from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from .config import DigitalOrgConfig
from .detector import TextPromptDetector


def build_geo_prompts_from_video(
    cfg: DigitalOrgConfig,
    video_path: str | Path,
    prompt: str,
    frame_indices: list[int],
    output_dir: str | Path,
    detect_conf: float | None = None,
    max_det: int | None = None,
    reuse_obj_ids_by_rank: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    frames_dir = output_dir / "prompt_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    old_conf = cfg.detect_conf
    old_max_det = cfg.max_det
    if detect_conf is not None:
        cfg.detect_conf = detect_conf
    if max_det is not None:
        cfg.max_det = max_det

    detector = TextPromptDetector(cfg)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    prompts = []
    try:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")

            frame_path = frames_dir / f"frame_{int(frame_idx):06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

            detections = detector.detect(frame_path, prompt)
            boxes = [d.bbox_xyxy for d in detections]
            scores = [d.score for d in detections]
            entry = {
                "frame_index": int(frame_idx),
                "boxes_xyxy": boxes,
                "scores": scores,
            }
            if reuse_obj_ids_by_rank:
                entry["obj_ids"] = list(range(1, len(boxes) + 1))
            prompts.append(entry)
    finally:
        cap.release()
        cfg.detect_conf = old_conf
        cfg.max_det = old_max_det

    geo_meta = {
        "video_path": str(video_path),
        "prompt": prompt,
        "detector": "DigitalOrgdet",
        "reuse_obj_ids_by_rank": bool(reuse_obj_ids_by_rank),
        "prompts": prompts,
    }
    geo_json = output_dir / "digitalorgdet_geo_prompts.json"
    geo_json.write_text(json.dumps(geo_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return geo_json


def track_video_with_geo_refine(
    cfg: DigitalOrgConfig,
    video_path: str | Path,
    geo_json: str | Path,
    output_dir: str | Path,
) -> None:
    sam3_repo = str(Path(cfg.sam3_repo).resolve())
    if sam3_repo not in sys.path:
        sys.path.insert(0, sam3_repo)
    from sam3.refine.refine_infer_video_text import geo_track_and_refine_video

    geo_track_and_refine_video(
        video_path=str(video_path),
        geo_json=str(geo_json),
        refine_ckpt=cfg.refine_checkpoint,
        device=cfg.device,
        roi_pad=cfg.roi_pad,
        out_dir=str(output_dir),
    )


def track_video_with_digitalorgdet_refine(
    cfg: DigitalOrgConfig,
    video_path: str | Path,
    prompt: str,
    frame_indices: list[int],
    output_dir: str | Path,
    detect_conf: float | None = None,
    max_det: int | None = None,
    reuse_obj_ids_by_rank: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geo_json = build_geo_prompts_from_video(
        cfg=cfg,
        video_path=video_path,
        prompt=prompt,
        frame_indices=frame_indices,
        output_dir=output_dir,
        detect_conf=detect_conf,
        max_det=max_det,
        reuse_obj_ids_by_rank=reuse_obj_ids_by_rank,
    )
    track_video_with_geo_refine(
        cfg=cfg,
        video_path=video_path,
        geo_json=geo_json,
        output_dir=output_dir,
    )
    return geo_json
