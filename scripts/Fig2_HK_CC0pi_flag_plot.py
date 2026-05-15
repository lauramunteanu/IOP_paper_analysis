"""Sanity plot: HK CC0pi Enu_QE bias from a single FSI NUISFLAT file.
Both functions select with `flagCC0pi`; the original two-function structure
is preserved for parity with Jake's commit."""
from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)


def _diff_flag(filename, nEvents):
    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
    flag = is_cc0pi_arr(arr, vertex=False)
    diff = (np.asarray(arr['Enu_QE']) - np.asarray(arr['Enu_true']))[flag] * 1000.0
    return diff


def plot_Enu_bias_numu_flag(ax, filename, nEvents):
    diff_sel = _diff_flag(filename, nEvents)
    ax.hist(diff_sel, bins=np.arange(-1000, 1000, step=10), histtype='step',
            weights=np.ones_like(diff_sel), color=dark_red, linestyle='--',
            linewidth=1.5, label="flag")
    custom_lines.append(Line2D([0], [0], color=dark_blue, lw=2, linestyle='-'))
    labels.append("flag")


def plot_Enu_bias_numu(ax, filename, nEvents):
    diff_sel = _diff_flag(filename, nEvents)
    ax.hist(diff_sel, bins=np.arange(-1000, 1000, step=10), histtype='step',
            weights=np.ones_like(diff_sel), color=dark_blue,
            linewidth=1.5, label="manual")
    custom_lines.append(Line2D([0], [0], color=dark_blue, lw=2, linestyle='-'))
    labels.append("manual")
    return ax


fig, ax = make_fig('single')
fname = "../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root"
ax = plot_Enu_bias_numu(ax, filename=fname, nEvents=100000)
plot_Enu_bias_numu_flag(ax, filename=fname, nEvents=100000)
plt.legend()
plt.savefig("Fig2_plots/Fig2_HK_CC0pi_flag.pdf")
plt.close(fig)
