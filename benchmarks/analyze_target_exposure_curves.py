"""Compare shared targets after exactly matched fitness-evaluation exposure."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    required = {
        "initial_records_mse",
        "target_quality_milestones",
        "quality_exposure_step",
        "quality_exposure_limit",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path} lacks exposure history: {sorted(missing)}")
    return payload


def _target_curve(run: dict, name: str,
                  milestones: list[int]) -> np.ndarray:
    initial = float(run["initial_records_mse"][name])
    if initial <= 0:
        raise ValueError(f"non-positive initial MSE for {name}")
    history = {
        int(exposure): float(mse)
        for exposure, mse in run["target_quality_milestones"][name]
    }
    missing = [value for value in milestones if value not in history]
    if missing:
        raise ValueError(
            f"target {name} in seed {run['seed']} lacks milestones {missing}")
    return np.asarray([
        100.0 * (initial - history[value]) / initial
        for value in milestones
    ])


def _bootstrap_target_mean(values: np.ndarray, rng: np.random.Generator,
                           samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap target identities after averaging repeated optimizer seeds."""
    target_means = values.mean(axis=1)
    indices = rng.integers(
        0, len(target_means), size=(samples, len(target_means)))
    boot = target_means[indices].mean(axis=1)
    return (
        np.quantile(boot, 0.025, axis=0),
        np.quantile(boot, 0.975, axis=0),
    )


def analyze(paths: list[Path], anchors: int, bootstrap: int,
            bootstrap_seed: int) -> dict:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for path in paths:
        payload = _load(path)
        grouped[int(payload["target_count"])].append(payload)
    if len(grouped) < 2:
        raise ValueError("at least two target-count conditions are required")
    for runs in grouped.values():
        runs.sort(key=lambda item: int(item["seed"]))

    counts = sorted(grouped)
    reference = counts[0]
    if anchors < 1 or anchors > reference:
        raise ValueError("anchors must fit in the smallest target condition")
    anchor_names = grouped[reference][0]["target_names"][:anchors]
    reference_seeds = [int(run["seed"]) for run in grouped[reference]]
    for count, runs in grouped.items():
        if [int(run["seed"]) for run in runs] != reference_seeds:
            raise ValueError("every condition must use the same seeds")
        for run in runs:
            if run["target_names"][:anchors] != anchor_names:
                raise ValueError(
                    f"condition {count} does not share the anchor targets")

    steps = {int(run["quality_exposure_step"])
             for runs in grouped.values() for run in runs}
    limits = {int(run["quality_exposure_limit"])
              for runs in grouped.values() for run in runs}
    if len(steps) != 1 or len(limits) != 1:
        raise ValueError("all runs must use the same exposure milestones")
    step = steps.pop()
    limit = limits.pop()
    milestones = list(range(0, limit + 1, step))
    rng = np.random.default_rng(bootstrap_seed)

    matrices: dict[int, np.ndarray] = {}
    curves: dict[str, list[dict]] = {}
    for count in counts:
        runs = grouped[count]
        values = np.stack([
            np.stack([
                _target_curve(run, name, milestones)
                for run in runs
            ], axis=0)
            for name in anchor_names
        ], axis=0)
        matrices[count] = values
        low, high = _bootstrap_target_mean(values, rng, bootstrap)
        seed_means = values.mean(axis=0)
        mean = values.mean(axis=(0, 1))
        seed_sd = seed_means.std(axis=0, ddof=1)
        curves[str(count)] = [
            {
                "exposure": exposure,
                "mean_improvement_pct": float(mean[index]),
                "target_ci_low": float(low[index]),
                "target_ci_high": float(high[index]),
                "seed_mean_sd": float(seed_sd[index]),
            }
            for index, exposure in enumerate(milestones)
        ]

    paired_at_limit: dict[str, dict] = {}
    reference_values = matrices[reference][:, :, -1]
    for count in counts:
        values = matrices[count][:, :, -1]
        delta = values - reference_values
        per_target_delta = delta.mean(axis=1)
        indices = rng.integers(
            0, anchors, size=(bootstrap, anchors))
        boot = per_target_delta[indices].mean(axis=1)
        paired_at_limit[str(count)] = {
            "mean_improvement_pct": float(values.mean()),
            "paired_delta_vs_reference_pct_points": float(delta.mean()),
            "paired_delta_ci_low": float(np.quantile(boot, 0.025)),
            "paired_delta_ci_high": float(np.quantile(boot, 0.975)),
            "targets_better_than_reference": int(np.sum(
                values.mean(axis=1) > reference_values.mean(axis=1))),
        }

    paired_adjacent_at_limit: dict[str, dict] = {}
    for previous, count in zip(counts, counts[1:]):
        previous_values = matrices[previous][:, :, -1]
        values = matrices[count][:, :, -1]
        delta = values - previous_values
        per_target_delta = delta.mean(axis=1)
        indices = rng.integers(
            0, anchors, size=(bootstrap, anchors))
        boot = per_target_delta[indices].mean(axis=1)
        paired_adjacent_at_limit[f"{previous}_to_{count}"] = {
            "from_target_count": previous,
            "to_target_count": count,
            "paired_delta_pct_points": float(delta.mean()),
            "paired_delta_ci_low": float(np.quantile(boot, 0.025)),
            "paired_delta_ci_high": float(np.quantile(boot, 0.975)),
            "targets_better_than_previous": int(np.sum(
                values.mean(axis=1) > previous_values.mean(axis=1))),
        }

    return {
        "metric": "percent_initial_target_mse_removed",
        "exposure_definition": (
            "times each fitness function scored a candidate phenotype"
        ),
        "anchor_count": anchors,
        "anchor_names": anchor_names,
        "seeds": reference_seeds,
        "target_counts": counts,
        "reference_target_count": reference,
        "quality_exposure_step": step,
        "quality_exposure_limit": limit,
        "bootstrap_target_samples": bootstrap,
        "curves": curves,
        "paired_at_limit": paired_at_limit,
        "paired_adjacent_at_limit": paired_adjacent_at_limit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--anchors", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_719)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        args.results,
        args.anchors,
        args.bootstrap,
        args.bootstrap_seed,
    )
    print(
        "targets  improvement at matched exposure  paired delta vs reference"
    )
    for count in result["target_counts"]:
        row = result["paired_at_limit"][str(count)]
        print(
            f"{count:>7}  {row['mean_improvement_pct']:>8.3f}%  "
            f"{row['paired_delta_vs_reference_pct_points']:>+8.3f} pp  "
            f"95% target CI "
            f"[{row['paired_delta_ci_low']:+.3f}, "
            f"{row['paired_delta_ci_high']:+.3f}]  "
            f"better targets "
            f"{row['targets_better_than_reference']}/{result['anchor_count']}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
