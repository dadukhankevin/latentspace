"""Restore target PNGs embedded in a species-vector benchmark result."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.result.read_text())
    images = payload.get("target_images")
    if not isinstance(images, dict) or not images:
        raise ValueError("result does not contain embedded target_images")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, uri in images.items():
        prefix = "data:image/png;base64,"
        if not isinstance(uri, str) or not uri.startswith(prefix):
            raise ValueError(f"target {name!r} is not an embedded PNG")
        (args.output / f"{name}.png").write_bytes(
            base64.b64decode(uri[len(prefix):]))
    print(f"restored {len(images)} targets to {args.output}")


if __name__ == "__main__":
    main()
