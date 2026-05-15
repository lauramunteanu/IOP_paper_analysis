"""Standalone colorbar(s) for the Fig6 2-D bias heatmaps.

Both Fig6_DUNE_EnuBias_2D_plot.py and Fig6_HK_EnuBias_2D_plot.py use the
same Z_CMAP / Z_VMAX / Z_VMIN, so a single shared colorbar applies to
every Fig6 panel. Drop into the LaTeX subfigure environment as the
common z-axis next to / under the panels.

Outputs (under scripts/Fig6_plots/):
  Fig6_colorbar.pdf            — vertical orientation, ~0.7" × 2.6"
  Fig6_colorbar_horizontal.pdf — horizontal orientation, ~5.0" × 0.6"
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import FlatTreeMod  # noqa: F401  -- side effect: rcParams
from Fig6_DUNE_EnuBias_2D_plot import Z_CMAP, Z_VMAX, Z_VMIN

OUT_DIR = os.path.join(os.environ.get("OUTPUT_PLOTS_DIR", _HERE), "Fig6_plots")
LABEL = r"Fraction per $E_{\nu}^{\rm true}$ bin"


def _make_cb(orientation, fig_w, fig_h, fname):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout='constrained')
    cmap = mpl.colormaps[Z_CMAP].copy()
    cmap.set_bad(cmap(0.0))
    norm = LogNorm(vmin=Z_VMIN, vmax=Z_VMAX)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                   orientation=orientation)
    cb.set_label(LABEL)
    out = f"{OUT_DIR}/{fname}"
    plt.savefig(f"{out}.png", dpi=200)
    plt.savefig(f"{out}.pdf")
    plt.close(fig)
    print(f"saved {out}.pdf")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Vertical: matches one panel's height (~2.6"), narrow
    _make_cb('vertical',   0.7, 2.6, "Fig6_colorbar")
    # Horizontal: spans most of \linewidth, short. Use this when placing
    # the colorbar once below all 6 panels in a 3x2 LaTeX grid.
    _make_cb('horizontal', 5.0, 0.6, "Fig6_colorbar_horizontal")


if __name__ == "__main__":
    main()
