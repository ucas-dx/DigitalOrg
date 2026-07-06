#!/usr/bin/env python3
from __future__ import annotations

import argparse

import uvicorn

from digitalorg.api import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DigitalOrg API server")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
