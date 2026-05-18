"""Standalone colorbar(s) for the Fig6 2-D bias heatmaps.

Fig6_DUNE_EnuBias_2D_plot.py / Fig6_HK_EnuBias_2D_plot.py now use a
mode-dependent z-scale (Z_VMAX_BY_MODE -- 0.3 for abs, 0.15 for rel) so the
finer rel y-binning doesn't wash out the rel panels. This script emits one
colorbar per mode, in both vertical and horizontal orientations.

Outputs (under scripts/Fig6_plots/):
  Fig6_colorbar_abs.{pdf,png}            — vertical, abs scale  (vmax=0.3)
  Fig6_colorbar_abs_horizontal.{pdf,png} — horizontal, abs scale
  Fig6_colorbar_rel.{pdf,png}            — vertical, rel scale  (vmax=0.15)
  Fig6_colorbar_rel_horizontal.{pdf,png} — horizontal, rel scale
  Fig6_colorbar.{pdf,png}, Fig6_colorbar_horizontal.{pdf,png}
     — back-compat aliases of the abs versions.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import FlatTreeMod  # noqa: F401  -- side effect: rcParams
from Fig6_DUNE_EnuBias_2D_plot import Z_CMAP, Z_VMAX_BY_MODE

OUT_DIR = os.path.join(os.environ.get("OUTPUT_PLOTS_DIR", _HERE), "Fig6_plots")
LABEL = r"Fraction per $E_{\nu}^{\rm true}$ bin"


def _make_cb(orientation, fig_w, fig_h, fname, vmax):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout='constrained')
    cmap = mpl.colormaps[Z_CMAP].copy()
    cmap.set_bad(cmap(0.0))
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
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
    for mode, vmax in Z_VMAX_BY_MODE.items():
        # Vertical: matches one panel's height (~2.6"), narrow
        _make_cb('vertical',   0.7, 2.6, f"Fig6_colorbar_{mode}", vmax)
        # Horizontal: spans most of \linewidth, short.
        _make_cb('horizontal', 5.0, 0.6, f"Fig6_colorbar_{mode}_horizontal", vmax)
    # Back-compat aliases pointing at the abs colorbar (the previous default).
    _make_cb('vertical',   0.7, 2.6, "Fig6_colorbar",            Z_VMAX_BY_MODE["abs"])
    _make_cb('horizontal', 5.0, 0.6, "Fig6_colorbar_horizontal", Z_VMAX_BY_MODE["abs"])


if __name__ == "__main__":
    main()
