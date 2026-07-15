"""Render round-8b: PCA vs MLP decoders across family size K, per family.

Color encodes the family (blue = smooth 1-D, yellow = blob 2-D); line style
encodes the fitting method (solid = PCA closed form, dashed = MLP
autoencoder). Values are normalized to the direct GA on the same family at
the same 5,000-eval budget.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from benchmarks.plot_family_scaling import (
    FAMILY_SIZES,
    STYLE,
    WIDTH,
    X0,
    X1,
    Y0,
    Y1,
    x_pos,
    y_pos,
)

RESULTS = Path("benchmark_results/mps_round8b_mlp_scaling_5000.json")
ROUND8 = Path("benchmark_results/mps_round8_mlp_pretrain_5000.json")
OUTPUT = Path("benchmark_results/mlp_vs_pca_scaling.svg")

SERIES = [
    ("smooth1d_256", "pca", "Smooth 1-D — PCA", "s1", None),
    ("smooth1d_256", "mlp", "Smooth 1-D — MLP", "s1", "6 4"),
    ("blob2d_1024", "pca", "Blob 2-D — PCA", "s3", None),
    ("blob2d_1024", "mlp", "Blob 2-D — MLP", "s3", "6 4"),
]


def main():
    runs: dict[tuple[str, str], list[float]] = {}
    for path in (RESULTS, ROUND8):
        for run in json.loads(path.read_text())["runs"]:
            runs.setdefault((run["objective"], run["strategy"]), []).append(
                run["metric_at_budget"]
            )

    parts: list[str] = []
    add = parts.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} 560" '
        'font-size="10" role="img" aria-label="Log-log chart: PCA and MLP '
        'decoders both improve with family size K; PCA improves faster">')
    add(f"<style>{STYLE}</style>")
    add('<rect class="surface" width="100%" height="100%"/>')
    add('<text class="ink" x="20" y="30" font-size="14" font-weight="600">'
        "Same corpus, two fitting methods — both obey the law, PCA is steeper"
        "</text>")
    add('<text class="ink2" x="20" y="48" font-size="10.5">'
        "Solid = PCA (closed form) · dashed = MLP autoencoder (2,000 Adam steps) · "
        "identical elite corpus per K · 10 seeds, ±2 SE</text>")

    for tick in (0.25, 0.5, 1, 2, 4):
        py = y_pos(tick)
        add(f'<line class="grid" x1="{X0}" y1="{py:.1f}" x2="{X1}" y2="{py:.1f}"/>')
        add(f'<text class="ink2" x="{X0 - 8}" y="{py + 3.5:.1f}" text-anchor="end">{tick}×</text>')
    for k in FAMILY_SIZES:
        px = x_pos(k)
        add(f'<line class="grid" x1="{px:.1f}" y1="{Y0}" x2="{px:.1f}" y2="{Y1}"/>')
        add(f'<text class="ink2" x="{px:.1f}" y="{Y1 + 16}" text-anchor="middle">{k}</text>')
    add(f'<line class="axis" x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}"/>')
    add(f'<line class="axis" x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}"/>')
    parity = y_pos(1.0)
    add(f'<line class="parity" x1="{X0}" y1="{parity:.1f}" x2="{X1}" y2="{parity:.1f}"/>')
    add(f'<text class="ink2" x="{X0 + 6}" y="{parity - 6:.1f}" font-size="9.5">'
        "direct GA parity (equal budget)</text>")

    label_slots: list[float] = []
    for objective, kind, label, css, dash in SERIES:
        reference = float(np.mean(runs[(objective, "direct_ga")]))
        means, errors = [], []
        for k in FAMILY_SIZES:
            values = np.asarray(runs[(objective, f"{kind}{k}")]) / reference
            means.append(float(values.mean()))
            errors.append(2 * float(values.std(ddof=1)) / math.sqrt(len(values)))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        points = " ".join(
            f"{x_pos(k):.1f},{y_pos(m):.1f}" for k, m in zip(FAMILY_SIZES, means)
        )
        add(f'<polyline class="{css}" points="{points}" fill="none" '
            f'stroke-width="2" stroke-linejoin="round"{dash_attr}/>')
        for k, mean, error in zip(FAMILY_SIZES, means, errors):
            px = x_pos(k)
            top, bottom = y_pos(mean + error), y_pos(max(mean - error, 1e-9))
            add(f'<line class="{css}" x1="{px:.1f}" y1="{top:.1f}" '
                f'x2="{px:.1f}" y2="{bottom:.1f}" stroke-width="1"/>')
            add(f'<circle class="{css}f" cx="{px:.1f}" cy="{y_pos(mean):.1f}" r="4">'
                f"<title>{label}: K={k}, {mean:.2f}× direct GA</title></circle>")
        end_y = y_pos(means[-1])
        while any(abs(end_y - other) < 14 for other in label_slots):
            end_y += 14
        label_slots.append(end_y)
        add(f'<text class="ink" x="{X1 + 10}" y="{end_y + 3.5:.1f}" '
            f'font-weight="600">{means[-1]:.2f}×</text>')
        add(f'<text class="ink2" x="{X1 + 44}" y="{end_y + 3.5:.1f}" '
            f'font-size="9">{label}</text>')

    add(f'<text class="ink" x="{(X0 + X1) / 2:.0f}" y="{Y1 + 38}" '
        'text-anchor="middle" font-size="10.5">Pretraining instances in the family (K)</text>')
    add(f'<text class="ink" transform="translate(18 {(Y0 + Y1) / 2:.0f}) rotate(-90)" '
        'text-anchor="middle" font-size="10.5">Fresh-instance loss ÷ direct GA</text>')

    legend_y = Y1 - 78
    for index, (objective, kind, label, css, dash) in enumerate(SERIES):
        row = legend_y + index * 17
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        add(f'<line class="{css}" x1="{X0 + 12}" y1="{row - 2}" x2="{X0 + 34}" '
            f'y2="{row - 2}" stroke-width="2"{dash_attr}/>')
        add(f'<text class="ink" x="{X0 + 42}" y="{row + 2}">{label}</text>')

    add("</svg>")
    OUTPUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
