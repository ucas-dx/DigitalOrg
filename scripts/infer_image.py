#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from digitalorg import DigitalOrgConfig, DigitalOrgPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DigitalOrg image inference")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompts", required=True, help="Comma-separated text prompts")
    parser.add_argument("--output-dir", default="outputs/infer_image", help="Directory for masks/json/overlay")
    parser.add_argument("--config", default="configs/default.yaml", help="DigitalOrg yaml config")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu")
    parser.add_argument("--digitalorgdet-weight", default=None, help="Override DigitalOrgdet text detector weight")
    parser.add_argument("--detect-conf", type=float, default=None, help="Override detector confidence threshold")
    parser.add_argument("--max-det", type=int, default=None, help="Override maximum detections before SAM3 refine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DigitalOrgConfig.from_yaml(args.config) if Path(args.config).exists() else DigitalOrgConfig()
    if args.device:
        cfg.device = args.device
    if args.digitalorgdet_weight:
        cfg.digitalorgdet_weight = args.digitalorgdet_weight
    if args.detect_conf is not None:
        cfg.detect_conf = args.detect_conf
    if args.max_det is not None:
        cfg.max_det = args.max_det
    pipeline = DigitalOrgPipeline(cfg)
    result = pipeline.predict_image(args.image, args.prompts, args.output_dir, save_outputs=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
