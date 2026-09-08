"""Render the six-panel reference-relative ablation figure.

Each bar is the signed percentage improvement over StyTr2 on the same
featured benchmark. Positive values consistently mean better performance.
Internal experiment IDs are intentionally replaced by functional labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures" / "ablation_metrics.png"

VARIANTS = [
    "Saliency-routed attention\n+ spatial α",
    "No structure guidance",
    "Residual saliency gating\n+ spatial α (proposed model)",
    "Saliency-routed attention\n+ scalar α",
    "Residual saliency gating\n+ scalar α",
]

# Colour encodes mechanism and alpha spatiality; the proposed model is outlined.
COLORS = ["#2F6B8A", "#B8B8B8", "#C45B32", "#8CBBD1", "#E6AA82"]
MAIN_INDEX = 2

METRICS = [
    ("SSIM", True, 0.5133, [0.5161, 0.5389, 0.5394, 0.5180, 0.5256]),
    ("OCR-sim", True, 0.3412, [0.4910, 0.5449, 0.5906, 0.5930, 0.5671]),
    ("E-SSIM", True, 0.4496, [0.4448, 0.4815, 0.4795, 0.4603, 0.4647]),
    ("Content loss", False, 2.0840, [1.7747, 1.6448, 1.6570, 1.7414, 1.6836]),
    ("Style loss", False, 2.1793, [1.7357, 1.7374, 1.7904, 1.6324, 1.5871]),
    ("LPIPS", False, 0.5649, [0.5493, 0.5172, 0.5218, 0.5481, 0.5426]),
]


def relative_improvement(
    values: np.ndarray, baseline: float, higher_is_better: bool
) -> np.ndarray:
    """Return signed percentage changes with positive always meaning better."""
    if higher_is_better:
        return (values / baseline - 1.0) * 100.0
    return (1.0 - values / baseline) * 100.0


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "axes.linewidth": 0.65,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.35))
    x = np.arange(len(VARIANTS))

    for panel, (ax, (metric, higher_is_better, baseline, raw_values)) in enumerate(
        zip(axes.flat, METRICS)
    ):
        values = np.asarray(raw_values, dtype=float)
        effects = relative_improvement(values, baseline, higher_is_better)
        best = int(np.argmax(effects))

        bars = ax.bar(
            x,
            effects,
            width=0.62,
            color=COLORS,
            edgecolor="none",
            zorder=2,
        )
        bars[MAIN_INDEX].set_edgecolor("#252525")
        bars[MAIN_INDEX].set_linewidth(1.0)

        upper = max(4.0, float(np.max(effects)) * 1.22)
        negative_extent = max(0.0, float(-np.min(effects)))
        lower = -max(upper * 0.26, negative_extent * 1.5)
        ax.set_ylim(lower, upper)
        ax.set_xlim(-0.65, len(values) - 0.35)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(
            f"({chr(97 + panel)}) {metric} "
            f"{'↑' if higher_is_better else '↓'}",
            loc="left",
            fontweight="bold",
            pad=5,
        )
        if panel % 3 == 0:
            ax.set_ylabel("Improvement (%)", labelpad=3)

        ax.axhline(0, color="#353535", linewidth=0.8, zorder=1)
        zero_fraction = (0.0 - lower) / (upper - lower)
        ax.text(
            0.985,
            zero_fraction - 0.014,
            "StyTr² baseline",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.2,
            color="#4A4A4A",
            zorder=4,
        )

        ax.grid(False)
        ax.tick_params(width=0.6, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        label_offset = (upper - lower) * 0.025
        for index, (bar, effect) in enumerate(zip(bars, effects)):
            positive = effect >= 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                effect + (label_offset if positive else -label_offset),
                f"{effect:+.1f}",
                ha="center",
                va="bottom" if positive else "top",
                fontsize=6.8,
                fontweight="bold" if index == best else "normal",
            )

    legend_handles = []
    for index, (color, label) in enumerate(zip(COLORS, VARIANTS)):
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="#252525" if index == MAIN_INDEX else "none",
                linewidth=1.0 if index == MAIN_INDEX else 0,
                label=label,
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.2,
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=1.6,
        handletextpad=0.55,
        labelspacing=0.75,
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.subplots_adjust(
        left=0.082,
        right=0.995,
        # Leave enough headroom for the top-row panel titles at high-DPI export.
        top=0.94,
        bottom=0.205,
        wspace=0.34,
        hspace=0.42,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=450, facecolor="white")
    plt.close(fig)
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
