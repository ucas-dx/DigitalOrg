#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from digitalorg import DigitalOrgConfig
from digitalorg.video_tracking import (
    track_video_with_digitalorgdet_refine,
    track_video_with_geo_refine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DigitalOrgdet box-prompt video tracking with SAM3/SAM2 tracker and ROI refine"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/video_track_refine")
    parser.add_argument("--prompt", default="organoid")
    parser.add_argument(
        "--prompt-frames",
        default="0",
        help="Comma-separated frame indices where DigitalOrgdet will detect boxes",
    )
    parser.add_argument("--geo-json", default="", help="Use an existing box prompt json directly")
    parser.add_argument("--device", default=None)
    parser.add_argument("--detect-conf", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument(
        "--reuse-obj-ids-by-rank",
        action="store_true",
        help="For multi-frame prompts, assign same object ids by score rank across prompt frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DigitalOrgConfig.from_yaml(args.config) if Path(args.config).exists() else DigitalOrgConfig()
    if args.device:
        cfg.device = args.device

    if args.geo_json:
        track_video_with_geo_refine(
            cfg=cfg,
            video_path=args.video,
            geo_json=args.geo_json,
            output_dir=args.output_dir,
        )
        print(f"tracking/refine done: {args.output_dir}")
        return

    frame_indices = [int(x.strip()) for x in args.prompt_frames.split(",") if x.strip()]
    geo_json = track_video_with_digitalorgdet_refine(
        cfg=cfg,
        video_path=args.video,
        prompt=args.prompt,
        frame_indices=frame_indices,
        output_dir=args.output_dir,
        detect_conf=args.detect_conf,
        max_det=args.max_det,
        reuse_obj_ids_by_rank=args.reuse_obj_ids_by_rank,
    )
    print(f"geo prompts: {geo_json}")
    print(f"tracking/refine done: {args.output_dir}")


if __name__ == "__main__":
    main()
