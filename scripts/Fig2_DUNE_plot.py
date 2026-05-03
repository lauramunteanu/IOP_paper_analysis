from FlatTreeMod import *
from collections import defaultdict
ROOT.gROOT.SetBatch(True)


def plot_Enu_bias_numu(filename, nEvents, withPiCorr, plot_name, vertex=False):
    """Caller passes a noFSI sample file when they want the noFSI line; we
    always read the post-FSI stack (vertex=False). The `vertex` kwarg is kept
    for backward compatibility but should not be set to True for plots — the
    FSI vertex stack is biased by NuWro binding-energy bookkeeping.
    All energies in MeV."""
    fig, ax = plt.subplots()
    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
    bias_wo_list_GeV, bias_with_list_GeV, valid = enu_had_arr(arr, vertex=False)
    # convert to MeV
    bias_wo_list   = bias_wo_list_GeV   * 1000.0
    bias_with_list = bias_with_list_GeV * 1000.0
    fScaleFactor = float(np.max(arr['fScaleFactor']))

    n_arr, pdg_arr, _, _, _, _ = particles_arr(arr, vertex=False)
    has_neutron_all = ak.to_numpy(ak.any(abs(pdg_arr) == 2112, axis=1))
    has_neutron = has_neutron_all[valid]
    bias_wo_by_n   = {bool(b): bias_wo_list[has_neutron == b]   for b in (False, True)}
    bias_with_by_n = {bool(b): bias_with_list[has_neutron == b] for b in (False, True)}

    bin_width = 16.0   # MeV
    bins = np.arange(-1000, 1000 + bin_width, step=bin_width)

    if not withPiCorr:
        bias = bias_wo_list
        bias_by_n = bias_wo_by_n
        line_color = dark_blue
        line_label = "w/o pion mass correction"
        ax.set_xlabel(r"$E_{\nu}^{\rm avail} - E_{\nu}^{\rm true}$ [MeV]")
    else:
        bias = bias_with_list
        bias_by_n = bias_with_by_n
        line_color = dark_red
        line_label = "w/ pion mass"
        ax.set_xlabel(r"$E_{\nu}^{\rm had} - E_{\nu}^{\rm true}$ [MeV]")

    weights_total = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(bias)
    ax.hist(bias, bins=bins, histtype='step', weights=weights_total,
            color=line_color, linewidth=1.5, label=line_label)

    # Per-neutron-content breakdown as step lines (no fill).
    for has_n, color in ((True, tol_magenta), (False, tol_teal)):
        vals = bias_by_n[has_n]
        if len(vals) == 0:
            continue
        w = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(vals)
        ax.hist(vals, bins=bins, histtype='step', weights=w,
                color=color, linewidth=1.4, linestyle="--",
                label=("With neutron" if has_n else "No neutron"))

    ax.set_xlim(-1000, 1000)
    ax.set_ylabel(DSIGMA_DE_LABEL)
    ax.legend(loc='best')
    plt.savefig(f"Fig2_plots/Fig2_DUNE_EnuRecoBias_{plot_name}.pdf")


_events = -1
NUMU = "../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root"
NUMUB = "../../Remade_April26/DUNE/DUNE_numub_FSI.flat.root"

# noFSI line now comes from the dedicated noFSI sample file, not from the
# vertex stack of the FSI sample (vertex stack is biased by NuWro's binding-
# energy bookkeeping at cascade exit). vertex=False reads the post-FSI stack;
# in a noFSI file no cascade ran so post-FSI == no-FSI.
plot_Enu_bias_numu(filename=noFSI_path(NUMU),  nEvents=_events, withPiCorr=True,  plot_name="WithPion_noFSI_numu",     vertex=False)
plot_Enu_bias_numu(filename=noFSI_path(NUMUB), nEvents=_events, withPiCorr=True,  plot_name="WithPion_noFSI_numubar", vertex=False)

plot_Enu_bias_numu(filename=noFSI_path(NUMU),  nEvents=_events, withPiCorr=False, plot_name="WithoutPion_noFSI_numu",     vertex=False)
plot_Enu_bias_numu(filename=noFSI_path(NUMUB), nEvents=_events, withPiCorr=False, plot_name="WithoutPion_noFSI_numubar", vertex=False)
