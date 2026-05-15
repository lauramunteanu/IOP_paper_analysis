"""Standalone legend for the Fig6 2-D bias heatmaps.

The four Fig6 panels share the same overlay markers (16-84% band,
median line, mean line). Drop a single legend file into the LaTeX
subfigure environment instead of repeating it under every panel.

Output: scripts/Fig6_plots/Fig6_legend.pdf
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from FlatTreeMod import save_strip_legend  # noqa: E402

OUT_DIR = os.path.join(os.environ.get("OUTPUT_PLOTS_DIR", _HERE), "Fig6_plots")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    handles = [
        Patch(facecolor="#F5DEB3", alpha=0.30, edgecolor="#888888",
              linewidth=0.6, label=r"$1\sigma$ (16\%-84\%)"),
        Line2D([0], [0], color="#F5DEB3", lw=1.8, label="median"),
        Line2D([0], [0], color="#E69F00", lw=1.7, ls="--", label="mean"),
    ]
    save_strip_legend(handles, OUT_DIR, "Fig6_legend",
                      fig_w=4.0, fig_h=0.45)


if __name__ == "__main__":
    main()
