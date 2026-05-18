from FlatTreeMod import *
from collections import defaultdict
ROOT.gROOT.SetBatch(True)

# Per-mode bin spec for the DUNE CC bias histogram.
BIN_SPECS = {
    "abs": dict(bin_width=16.0,  lo=-1000.0, hi=1000.0, xlim=(-900.0, 300.0)),
    "rel": dict(bin_width=0.005, lo=REL_BIAS_XLIM[0], hi=REL_BIAS_XLIM[1], xlim=REL_BIAS_XLIM),
}


def plot_Enu_bias_numu(filename, nEvents, withPiCorr, plot_name, mode, vertex=False):
    """Caller passes a noFSI sample file when they want the noFSI line; we
    always read the post-FSI stack (vertex=False). withPiCorr=True picks
    Enu^had (charged-pi full E); False picks Enu^avail (charged-pi KE only)."""
    fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))
    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))

    observable = "had" if withPiCorr else "avail"
    bias = bias_arr(arr, observable, kind=mode, vertex=False)

    # neutron-content split on the same CC selection as bias_arr uses.
    _, _, cc_mask = enu_had_arr(arr, vertex=False)
    n_arr, pdg_arr, _, _, _, _ = particles_arr(arr, vertex=False)
    has_neutron_all = ak.to_numpy(ak.any(abs(pdg_arr) == 2112, axis=1))
    has_neutron = has_neutron_all[np.asarray(cc_mask, dtype=bool)]
    bias_by_n = {bool(b): bias[has_neutron == b] for b in (False, True)}

    spec = BIN_SPECS[mode]
    bin_width = spec["bin_width"]
    bins = np.arange(spec["lo"], spec["hi"] + bin_width, step=bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fScaleFactor = float(np.max(arr['fScaleFactor']))
    weights_total = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(bias)
    counts_total, _ = np.histogram(bias, bins=bins, weights=weights_total)

    ax.hist(bias, bins=bins, histtype='step', weights=weights_total,
            color=tol_dark, linewidth=1.8)

    sub_counts = {}
    for has_n, color, label in ((False, tol_teal,    "No neutrons"),
                                (True,  tol_magenta, "With neutrons")):
        vals = bias_by_n[has_n]
        if len(vals) == 0:
            sub_counts[has_n] = np.zeros_like(counts_total)
            continue
        w = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(vals)
        ax.hist(vals, bins=bins, histtype='step', weights=w,
                color=color, linewidth=1.4)
        c, _ = np.histogram(vals, bins=bins, weights=w)
        sub_counts[has_n] = c

    # Ratio panel: each subset / total. Strict division -- the two lines sum
    # to 1 in every bin with at least one event; bins with total=0 give NaN
    # and the step lines break there.
    for has_n, color in ((False, tol_teal), (True, tol_magenta)):
        ratio = np.divide(sub_counts[has_n], counts_total,
                          out=np.full_like(counts_total, np.nan, dtype=float),
                          where=counts_total > 0)
        ax_ratio.step(centers, ratio, where="mid", color=color, linewidth=1.4)
    ax_ratio.set_ylim(0, 1.05)
    ax_ratio.set_ylabel("fraction\nof total")

    ax.set_xlim(*spec["xlim"])
    ax.set_ylabel(bias_ylabel(mode))
    legend_handles = [
        Line2D([0], [0], color=tol_dark,    lw=1.8, label="Total CC inc."),
        Line2D([0], [0], color=tol_teal,    lw=1.4, label="No neutrons"),
        Line2D([0], [0], color=tol_magenta, lw=1.4, label="With neutrons"),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=13)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax_ratio.set_xlim(*spec["xlim"])
    ax_ratio.set_xlabel(bias_xlabel(observable, mode))

    plt.savefig(outpath("Fig2_plots", f"Fig2_DUNE_EnuRecoBias_{plot_name}_{mode}.pdf"))
    plt.close(fig)


_events = -1
NUMU = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root"
NUMUB = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root"

# noFSI line comes from the dedicated noFSI sample file (post-FSI stack of an
# FSI file is biased by NuWro binding-energy bookkeeping at cascade exit).
_CONFIGS = [
    dict(filename=noFSI_path(NUMU),  withPiCorr=True,  plot_name="WithPion_noFSI_numu"),
    dict(filename=noFSI_path(NUMUB), withPiCorr=True,  plot_name="WithPion_noFSI_numubar"),
    dict(filename=noFSI_path(NUMU),  withPiCorr=False, plot_name="WithoutPion_noFSI_numu"),
    dict(filename=noFSI_path(NUMUB), withPiCorr=False, plot_name="WithoutPion_noFSI_numubar"),
]

for mode in ("abs", "rel"):
    for cfg in _CONFIGS:
        plot_Enu_bias_numu(nEvents=_events, mode=mode, vertex=False, **cfg)
