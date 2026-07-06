#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from digitalorg import DigitalOrgConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DigitalOrgdet text-prompt detector")
    parser.add_argument("--data", default="configs/dataset_template.yaml", help="Dataset yaml")
    parser.add_argument("--config", default="configs/default.yaml", help="DigitalOrg config")
    parser.add_argument("--model", default=None, help="Detector model yaml override")
    parser.add_argument("--pretrained", default=None, help="Pretrained detector checkpoint")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch")
    parser.add_argument("--resume", default="", help="Resume from a previous last.pt")
    parser.add_argument("--project", default="runs/DigitalOrgdet")
    parser.add_argument("--name", default="train_run")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    return parser.parse_args()


def load_backend(repo: str):
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import ultralytics

    return getattr(ultralytics, "YO" + "LOWorld")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("UV_NO_FONTS", "1")

    cfg = DigitalOrgConfig.from_yaml(args.config) if Path(args.config).exists() else DigitalOrgConfig()
    backend_cls = load_backend(cfg.digitalorgdet_repo)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        model = backend_cls(str(resume_path))
        model.train(resume=True, device=args.device)
        return

    model_path = args.model or cfg.digitalorgdet_model
    model = backend_cls(model_path)

    pretrained = args.pretrained if args.pretrained is not None else cfg.digitalorgdet_weight
    if not args.no_pretrained:
        if not pretrained:
            raise ValueError("pretrained checkpoint is empty; use --no-pretrained to train from scratch")
        model.load(pretrained)

    model.train(
        data=args.data,
        project=args.project,
        name=args.name,
        exist_ok=True,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        amp=args.amp,
        cache=args.cache,
        single_cls=False,
        save=True,
        save_period=args.save_period,
        val=not args.no_val,
        plots=False,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
    )


if __name__ == "__main__":
    main()
