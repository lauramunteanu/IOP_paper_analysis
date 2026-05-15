"""Sanity plot: HK CC0pi Enu_QE bias from a single FSI NUISFLAT file.
Both functions select with `flagCC0pi`; the original two-function structure
is preserved for parity with Jake's commit."""
from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

BIN_SPECS = {
    "abs": dict(bin_width=10.0,  lo=-1000.0, hi=1000.0, xlim=(-900.0, 300.0)),
    "rel": dict(bin_width=0.005, lo=REL_BIAS_XLIM[0], hi=REL_BIAS_XLIM[1], xlim=REL_BIAS_XLIM),
}


def _diff_flag(filename, nEvents, mode):
    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
    return bias_arr(arr, "qe", kind=mode, vertex=False)


def plot_Enu_bias_numu_flag(ax, filename, nEvents, mode):
    diff_sel = _diff_flag(filename, nEvents, mode)
    spec = BIN_SPECS[mode]
    bin_width = spec["bin_width"]
    bins = np.arange(spec["lo"], spec["hi"] + bin_width, step=bin_width)
    ax.hist(diff_sel, bins=bins, histtype='step',
            weights=np.ones_like(diff_sel), color=dark_red, linestyle='--',
            linewidth=1.5, label="flag")
    custom_lines.append(Line2D([0], [0], color=dark_blue, lw=2, linestyle='-'))
    labels.append("flag")


def plot_Enu_bias_numu(ax, filename, nEvents, mode):
    diff_sel = _diff_flag(filename, nEvents, mode)
    spec = BIN_SPECS[mode]
    bin_width = spec["bin_width"]
    bins = np.arange(spec["lo"], spec["hi"] + bin_width, step=bin_width)
    ax.hist(diff_sel, bins=bins, histtype='step',
            weights=np.ones_like(diff_sel), color=dark_blue,
            linewidth=1.5, label="manual")
    custom_lines.append(Line2D([0], [0], color=dark_blue, lw=2, linestyle='-'))
    labels.append("manual")
    return ax


_fname = "../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root"

for mode in ("abs", "rel"):
    custom_lines, labels = [], []
    fig, ax = make_fig('single')
    ax = plot_Enu_bias_numu(ax, filename=_fname, nEvents=100000, mode=mode)
    plot_Enu_bias_numu_flag(ax, filename=_fname, nEvents=100000, mode=mode)
    ax.set_xlim(*BIN_SPECS[mode]["xlim"])
    ax.set_xlabel(bias_xlabel("qe", mode))
    plt.legend()
    plt.savefig(outpath("Fig2_plots", f"Fig2_HK_CC0pi_flag_{mode}.pdf"))
    plt.close(fig)
