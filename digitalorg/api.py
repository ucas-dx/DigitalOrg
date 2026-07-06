from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile

from .config import DigitalOrgConfig
from .pipeline import DigitalOrgPipeline
from .video_tracking import (
    track_video_with_digitalorgdet_refine,
    track_video_with_geo_refine,
)


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = DigitalOrgConfig.from_yaml(config_path) if config_path else DigitalOrgConfig()
    pipeline = DigitalOrgPipeline(cfg)
    app = FastAPI(title="DigitalOrg", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(
        image: Annotated[UploadFile, File()],
        prompts: Annotated[str, Form()],
        output_dir: Annotated[str | None, Form()] = None,
        save_outputs: Annotated[bool, Form()] = True,
        detect_conf: Annotated[float | None, Form()] = None,
        max_det: Annotated[int | None, Form()] = None,
        digitalorgdet_model: Annotated[str | None, Form()] = None,
        digitalorgdet_weight: Annotated[str | None, Form()] = None,
    ) -> dict:
        suffix = Path(image.filename or "image.png").suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(await image.read())
            image_path = f.name
        out = output_dir or tempfile.mkdtemp(prefix="digitalorg_")
        if digitalorgdet_model or digitalorgdet_weight or detect_conf is not None or max_det is not None:
            request_cfg = _with_detector_overrides(
                cfg,
                digitalorgdet_model=digitalorgdet_model,
                digitalorgdet_weight=digitalorgdet_weight,
                detect_conf=detect_conf,
                max_det=max_det,
            )
            return DigitalOrgPipeline(request_cfg).predict_image(
                image_path, prompts, output_dir=out, save_outputs=save_outputs
            )
        return pipeline.predict_image(image_path, prompts, output_dir=out, save_outputs=save_outputs)

    @app.post("/predict/image")
    async def predict_image(
        image: Annotated[UploadFile, File()],
        prompts: Annotated[str, Form()],
        output_dir: Annotated[str | None, Form()] = None,
        save_outputs: Annotated[bool, Form()] = True,
        detect_conf: Annotated[float | None, Form()] = None,
        max_det: Annotated[int | None, Form()] = None,
        digitalorgdet_model: Annotated[str | None, Form()] = None,
        digitalorgdet_weight: Annotated[str | None, Form()] = None,
    ) -> dict:
        return await predict(
            image=image,
            prompts=prompts,
            output_dir=output_dir,
            save_outputs=save_outputs,
            detect_conf=detect_conf,
            max_det=max_det,
            digitalorgdet_model=digitalorgdet_model,
            digitalorgdet_weight=digitalorgdet_weight,
        )

    @app.post("/predict/video")
    async def predict_video(
        video: Annotated[UploadFile, File()],
        prompt: Annotated[str, Form()],
        prompt_frames: Annotated[str, Form()] = "0",
        output_dir: Annotated[str | None, Form()] = None,
        detect_conf: Annotated[float | None, Form()] = None,
        max_det: Annotated[int | None, Form()] = None,
        reuse_obj_ids_by_rank: Annotated[bool, Form()] = False,
        digitalorgdet_model: Annotated[str | None, Form()] = None,
        digitalorgdet_weight: Annotated[str | None, Form()] = None,
    ) -> dict:
        suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(await video.read())
            video_path = f.name

        out = output_dir or tempfile.mkdtemp(prefix="digitalorg_video_")
        frame_indices = [int(x.strip()) for x in prompt_frames.split(",") if x.strip()]
        request_cfg = _with_detector_overrides(cfg, digitalorgdet_model, digitalorgdet_weight)
        geo_json = track_video_with_digitalorgdet_refine(
            cfg=request_cfg,
            video_path=video_path,
            prompt=prompt,
            frame_indices=frame_indices,
            output_dir=out,
            detect_conf=detect_conf,
            max_det=max_det,
            reuse_obj_ids_by_rank=reuse_obj_ids_by_rank,
        )
        return _video_result(out, geo_json=str(geo_json), video_path=video_path)

    @app.post("/predict/video_geo")
    async def predict_video_geo(
        video: Annotated[UploadFile, File()],
        geo_json: Annotated[UploadFile, File()],
        output_dir: Annotated[str | None, Form()] = None,
    ) -> dict:
        video_suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix) as f:
            f.write(await video.read())
            video_path = f.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            f.write(await geo_json.read())
            geo_json_path = f.name

        out = output_dir or tempfile.mkdtemp(prefix="digitalorg_video_geo_")
        track_video_with_geo_refine(
            cfg=cfg,
            video_path=video_path,
            geo_json=geo_json_path,
            output_dir=out,
        )
        return _video_result(out, geo_json=geo_json_path, video_path=video_path)

    return app


def _video_result(output_dir: str | Path, geo_json: str, video_path: str) -> dict:
    out = Path(output_dir)
    return {
        "video_path": video_path,
        "output_dir": str(out),
        "geo_json": geo_json,
        "coarse_mask_index": str(out / "mask_index.json"),
        "coarse_overlay": str(out / "tracking_overlay_geo_boxes.mp4"),
        "refined_mask_index": str(out / "refine" / "refined_mask_index.json"),
        "refined_overlay": str(out / "refine" / "tracking_overlay_refined.mp4"),
    }


def _with_detector_overrides(
    cfg: DigitalOrgConfig,
    digitalorgdet_model: str | None = None,
    digitalorgdet_weight: str | None = None,
    detect_conf: float | None = None,
    max_det: int | None = None,
) -> DigitalOrgConfig:
    request_cfg = deepcopy(cfg)
    if digitalorgdet_model:
        request_cfg.digitalorgdet_model = digitalorgdet_model
    if digitalorgdet_weight:
        request_cfg.digitalorgdet_weight = digitalorgdet_weight
    if detect_conf is not None:
        request_cfg.detect_conf = detect_conf
    if max_det is not None:
        request_cfg.max_det = max_det
    return request_cfg


app = create_app()
