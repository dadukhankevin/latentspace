"""Summarize fixed-compute rotating-target benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    grouped: dict[int, list[dict]] = defaultdict(list)
    for path in args.results:
        payload = json.loads(path.read_text())
        grouped[int(payload["target_count"])].append(payload)
    counts = sorted(grouped)
    if not counts:
        raise ValueError("no results supplied")
    core_count = counts[0]
    core_names = list(grouped[core_count][0]["archive_records_mse"])

    rows: list[tuple[int, float, float, float, float, float]] = []
    print("targets  archive mean±SD  worst  shared-core mean±SD  exposure")
    for count in counts:
        runs = grouped[count]
        active = np.asarray([
            run["archive_metrics"]["mean_mse"] for run in runs])
        worst = np.asarray([
            run["archive_metrics"]["worst_mse"] for run in runs])
        core = np.asarray([
            np.mean([run["archive_records_mse"][name]
                     for name in core_names])
            for run in runs
        ])
        exposure = float(np.mean([
            np.mean(run["target_search_exposures"][:core_count])
            for run in runs
        ]))
        rows.append((
            count, float(active.mean()), float(active.std(ddof=1)),
            float(worst.mean()), float(core.mean()),
            float(core.std(ddof=1)),
        ))
        print(
            f"{count:>7}  {active.mean():.6f}±{active.std(ddof=1):.6f}  "
            f"{worst.mean():.6f}  "
            f"{core.mean():.6f}±{core.std(ddof=1):.6f}  {exposure:.0f}")

    large = [row for row in rows if row[0] >= 32]
    if len(large) >= 3:
        x = np.log([row[0] for row in large])
        y = np.log([row[4] for row in large])
        exponent, intercept = np.polyfit(x, y, 1)
        prediction = intercept + exponent * x
        r_squared = 1 - np.sum((y - prediction) ** 2) / np.sum(
            (y - y.mean()) ** 2)
        print(
            f"shared-core fixed-compute fit for N>=32: "
            f"MSE proportional to N^{exponent:.3f}, R^2={r_squared:.3f}")


if __name__ == "__main__":
    main()
