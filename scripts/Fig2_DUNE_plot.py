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
    fig, ax = make_fig('single')
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

    fScaleFactor = float(np.max(arr['fScaleFactor']))
    if withPiCorr:
        line_color = dark_red
        line_label = "w/ pion mass"
    else:
        line_color = dark_blue
        line_label = "w/o pion mass correction"

    weights_total = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(bias)
    ax.hist(bias, bins=bins, histtype='step', weights=weights_total,
            color=line_color, linewidth=1.5, label=line_label)

    for has_n, color in ((True, tol_magenta), (False, tol_teal)):
        vals = bias_by_n[has_n]
        if len(vals) == 0:
            continue
        w = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(vals)
        ax.hist(vals, bins=bins, histtype='step', weights=w,
                color=color, linewidth=1.4, linestyle="--",
                label=("With neutron" if has_n else "No neutron"))

    ax.set_xlim(*spec["xlim"])
    ax.set_xlabel(bias_xlabel(observable, mode))
    ax.set_ylabel(bias_ylabel(mode))
    ax.legend(loc='best')
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
