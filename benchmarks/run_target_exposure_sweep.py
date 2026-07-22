"""Run the exact per-target exposure sweep on nested image targets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets-from",
        type=Path,
        required=True,
        help="Existing benchmark JSON whose target list supplies the nesting.",
    )
    parser.add_argument("--counts", type=parse_ints,
                        default=parse_ints("32,64,96,128,168,256"))
    parser.add_argument("--seeds", type=parse_ints,
                        default=parse_ints("3,4,5"))
    parser.add_argument("--budget", type=int, default=30_000)
    parser.add_argument("--coefficient-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmark_results"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.targets_from.read_text())
    targets = [str(path) for path in source["targets"]]
    if max(args.counts) > len(targets):
        raise ValueError("requested target count exceeds the source target list")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.counts) * len(args.seeds)
    current = 0
    for seed in args.seeds:
        for count in args.counts:
            current += 1
            output = args.output_dir / (
                f"image_target_exposure_cifar_n{count}_s{seed}_"
                f"{args.budget}.json"
            )
            if output.exists() and not args.force:
                print(f"[{current}/{total}] keeping {output}", flush=True)
                continue
            print(
                f"\n[{current}/{total}] targets={count} seed={seed}",
                flush=True,
            )
            command = [
                sys.executable,
                "-m",
                "benchmarks.demo_image_fitness_scaling_rotating",
                *targets[:count],
                "--budget", str(args.budget),
                "--seed", str(seed),
                "--coefficient-dim", str(args.coefficient_dim),
                "--quality-exposure-step", "250",
                "--quality-exposure-limit", "3000",
                "--reports", "10",
                "--output", str(output),
            ]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
