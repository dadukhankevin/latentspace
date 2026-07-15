"""Render the round-7 family-pretraining scaling law as a standalone SVG.

Pure-Python SVG generation (matplotlib is unavailable in this environment).
Loss on fresh instances, normalized to the direct GA at the same 5,000-eval
budget (1.0 = parity), versus pretraining instances K, log-log, ten seeds,
error bars are +/-2 SE. The SVG carries light and dark styles via
prefers-color-scheme.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RESULTS = Path("benchmark_results/mps_round7_scaling_5000.json")
OUTPUT = Path("benchmark_results/family_scaling.svg")

# Categorical slots 1-4 of the validated reference palette, fixed order.
SERIES = [
    ("smooth1d_256", "Smooth 1-D (256-d, 16-d manifold)", "s1"),
    ("rugged_smooth_256", "Rugged smooth (multimodal landscape)", "s2"),
    ("image2d_1024", "Image 2-D (1,024-d, 25-d manifold)", "s3"),
    ("rough1d_256", "Rough control (no shared structure)", "s4"),
]
FAMILY_SIZES = [8, 16, 32, 64, 128]

STYLE = """
  svg { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .surface { fill: #fcfcfb; }
  .ink  { fill: #0b0b0b; }
  .ink2 { fill: #52514e; }
  .grid { stroke: #e4e3df; stroke-width: 1; }
  .axis { stroke: #52514e; stroke-width: 1; }
  .parity { stroke: #52514e; stroke-width: 1.2; stroke-dasharray: 4 3; }
  .s1 { stroke: #2a78d6; } .s1f { fill: #2a78d6; }
  .s2 { stroke: #1baf7a; } .s2f { fill: #1baf7a; }
  .s3 { stroke: #eda100; } .s3f { fill: #eda100; }
  .s4 { stroke: #4a3aa7; } .s4f { fill: #4a3aa7; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink  { fill: #ffffff; }
    .ink2 { fill: #c3c2b7; }
    .grid { stroke: #34332f; }
    .axis { stroke: #c3c2b7; }
    .parity { stroke: #c3c2b7; }
    .s1 { stroke: #3987e5; } .s1f { fill: #3987e5; }
    .s2 { stroke: #199e70; } .s2f { fill: #199e70; }
    .s3 { stroke: #c98500; } .s3f { fill: #c98500; }
    .s4 { stroke: #9085e9; } .s4f { fill: #9085e9; }
  }
"""

WIDTH, HEIGHT = 880, 560
X0, X1 = 70, 620
Y0, Y1 = 78, 462
Y_MIN, Y_MAX = 0.22, 7.0


def x_pos(k: float) -> float:
    return X0 + (math.log2(k) - 3) / 4 * (X1 - X0)


def y_pos(v: float) -> float:
    span = math.log10(Y_MAX) - math.log10(Y_MIN)
    return Y1 - (math.log10(v) - math.log10(Y_MIN)) / span * (Y1 - Y0)


def main():
    data = json.loads(RESULTS.read_text())
    runs: dict[tuple[str, str], list[float]] = {}
    for run in data["runs"]:
        runs.setdefault((run["objective"], run["strategy"]), []).append(
            run["metric_at_budget"]
        )

    parts: list[str] = []
    add = parts.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'font-size="10">')
    add(f"<style>{STYLE}</style>")
    add(f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}"/>')

    add('<text class="ink" x="20" y="30" font-size="14" font-weight="600">'
        "Family-pretraining scaling law — learned genetic code vs direct search"
        "</text>")
    add('<text class="ink2" x="20" y="48" font-size="10.5">'
        "PCA-32 decoder pretrained on K instances, CMA-ES in its latent; "
        "5,000 fresh evaluations per instance; 10 seeds; error bars ±2 SE"
        "</text>")

    for tick in (0.25, 0.5, 1, 2, 4):
        py = y_pos(tick)
        add(f'<line class="grid" x1="{X0}" y1="{py:.1f}" x2="{X1}" y2="{py:.1f}"/>')
        add(f'<text class="ink2" x="{X0 - 8}" y="{py + 3.5:.1f}" '
            f'text-anchor="end">{tick}×</text>')
    for k in FAMILY_SIZES:
        px = x_pos(k)
        add(f'<line class="grid" x1="{px:.1f}" y1="{Y0}" x2="{px:.1f}" y2="{Y1}"/>')
        add(f'<text class="ink2" x="{px:.1f}" y="{Y1 + 16}" '
            f'text-anchor="middle">{k}</text>')
    add(f'<line class="axis" x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}"/>')
    add(f'<line class="axis" x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}"/>')

    parity = y_pos(1.0)
    add(f'<line class="parity" x1="{X0}" y1="{parity:.1f}" x2="{X1}" y2="{parity:.1f}"/>')
    add(f'<text class="ink2" x="{X0 + 6}" y="{parity - 6:.1f}" font-size="9.5">'
        "direct GA parity (equal budget)</text>")

    label_slots: list[float] = []
    for objective, label, css in SERIES:
        reference = float(np.mean(runs[(objective, "direct_ga")]))
        means, errors = [], []
        for k in FAMILY_SIZES:
            values = np.asarray(runs[(objective, f"family{k}")]) / reference
            means.append(float(values.mean()))
            errors.append(2 * float(values.std(ddof=1)) / math.sqrt(len(values)))

        points = " ".join(
            f"{x_pos(k):.1f},{y_pos(m):.1f}" for k, m in zip(FAMILY_SIZES, means)
        )
        add(f'<polyline class="{css}" points="{points}" fill="none" '
            f'stroke-width="2" stroke-linejoin="round"/>')
        for k, mean, error in zip(FAMILY_SIZES, means, errors):
            px, cap = x_pos(k), 2.5
            top, bottom = y_pos(mean + error), y_pos(max(mean - error, 1e-9))
            add(f'<line class="{css}" x1="{px:.1f}" y1="{top:.1f}" '
                f'x2="{px:.1f}" y2="{bottom:.1f}" stroke-width="1"/>')
            for py in (top, bottom):
                add(f'<line class="{css}" x1="{px - cap:.1f}" y1="{py:.1f}" '
                    f'x2="{px + cap:.1f}" y2="{py:.1f}" stroke-width="1"/>')
            add(f'<circle class="{css}f" cx="{px:.1f}" cy="{y_pos(mean):.1f}" r="4">'
                f"<title>{label}: K={k}, {mean:.2f}× direct GA</title></circle>")

        # Direct end labels, nudged apart if two series end close together.
        end_y = y_pos(means[-1])
        while any(abs(end_y - other) < 14 for other in label_slots):
            end_y += 14
        label_slots.append(end_y)
        add(f'<text class="ink" x="{X1 + 10}" y="{end_y + 3.5:.1f}" '
            f'font-weight="600">{means[-1]:.2f}×</text>')
        add(f'<text class="ink2" x="{X1 + 44}" y="{end_y + 3.5:.1f}" '
            f'font-size="9">{label.split(" (")[0]}</text>')

    add(f'<text class="ink" x="{(X0 + X1) / 2:.0f}" y="{Y1 + 38}" '
        'text-anchor="middle" font-size="10.5">'
        "Pretraining instances in the family (K)</text>")
    add(f'<text class="ink" transform="translate(18 {(Y0 + Y1) / 2:.0f}) rotate(-90)" '
        'text-anchor="middle" font-size="10.5">'
        "Fresh-instance loss ÷ direct GA</text>")

    legend_y = Y1 - 78
    for index, (objective, label, css) in enumerate(SERIES):
        row = legend_y + index * 17
        add(f'<rect class="{css}f" x="{X0 + 12}" y="{row - 8}" width="12" '
            f'height="12" rx="2"/>')
        add(f'<text class="ink" x="{X0 + 30}" y="{row + 2}">{label}</text>')

    add("</svg>")
    OUTPUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
