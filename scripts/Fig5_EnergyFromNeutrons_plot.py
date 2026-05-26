from FlatTreeMod import *
from matplotlib.lines import Line2D
import ROOT

ROOT.gROOT.SetBatch(True)

Mneutron = 939.565  # MeV

# Bin definition shared across all panels
bin_width = 0.025
bins = np.arange(0, 1 + bin_width, step=bin_width)
centers = 0.5 * (bins[:-1] + bins[1:])

# Three neutron-multiplicity groups (in plot order, top → bottom in legend)
GROUPS = [
    (r"$N_n = 1$",   lambda nn: nn == 1),
    (r"$N_n = 2$",   lambda nn: nn == 2),
    (r"$N_n \geq 3$", lambda nn: nn >= 3),
]


def compute_neutron_data(filename, nEvents, lep_pdg=13):
    """Per-event (n_neutrons, x = ΣTn/q0, fScaleFactor, osc_w). After CC0pi-style cuts."""
    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
    fScaleFactor = float(arr["fScaleFactor"][0])

    n, pdg, E, _, _, _ = particles_arr(arr, vertex=False)
    apdg = abs(pdg)
    is_neutron = apdg == 2112
    nneutron_per_evt = ak.to_numpy(ak.sum(is_neutron, axis=1))
    neutron_E_MeV = E * 1000.0
    KE_per_neutron = ak.where(is_neutron, neutron_E_MeV - Mneutron, 0.0)
    neutron_KE = ak.to_numpy(ak.sum(KE_per_neutron, axis=1))
    bad_event = ak.to_numpy(ak.any(apdg > 3000, axis=1))
    Enu  = ak.to_numpy(arr["Enu_true"])
    ELep = ak.to_numpy(arr["ELep"])
    q0   = (Enu - ELep) * 1000.0

    keep = (~bad_event) & (neutron_KE != 0) & (q0 != 0)
    # Per-event νμ→νμ (or νμ→νe for lep_pdg=11) osc weight on the kept events.
    osc_fn = osc_weights_mue if lep_pdg == 11 else osc_weights_mumu
    osc_w = osc_fn(Enu[keep], filename=filename)
    return dict(
        n_neutrons=nneutron_per_evt[keep],
        x=neutron_KE[keep] / q0[keep],
        fScaleFactor=fScaleFactor,
        osc_w=osc_w,
    )


def make_grouped_hists(d):
    """Returns list of histogram arrays, one per group + total."""
    base = d["fScaleFactor"] / bin_width
    hists = []
    for label, sel_fn in GROUPS:
        mask = sel_fn(d["n_neutrons"])
        x = d["x"][mask]
        w = base * d["osc_w"][mask]
        h, _ = np.histogram(x, bins=bins, weights=w)
        hists.append(h)
    # Total = all events
    h_tot, _ = np.histogram(d["x"], bins=bins, weights=base * d["osc_w"])
    hists.append(h_tot)
    return hists


FIG5_COLORS     = list(TOL_MUTED[:3]) + [tol_dark]
FIG5_LABELS     = [g[0] for g in GROUPS] + ["Total"]
FIG5_LINEWIDTHS = [1.2, 1.2, 1.2, 1.8]


def plot_grouped(ax, ax_ratio, hists, ymax):
    for h, col, lw in zip(hists, FIG5_COLORS, FIG5_LINEWIDTHS):
        ax.step(centers, h, where="mid", color=col, linewidth=lw)

    # Ratio panel: each subgroup / total. Shows how much each Nn contributes
    # to the total at each x-bin (sums to 1 across the 3 subgroup lines).
    h_total = hists[-1]
    safe_total = np.where(h_total > 0, h_total, np.nan)
    for h, col, lw in zip(hists[:-1], FIG5_COLORS[:-1], FIG5_LINEWIDTHS[:-1]):
        ax_ratio.step(centers, h / safe_total, where="mid", color=col, linewidth=lw)
    ax_ratio.set_ylim(0, 1.05)
    ax_ratio.set_ylabel("fraction\nof total")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax_ratio.set_xlabel(r"$\sum T_n / q_0$")
    ax.set_ylabel(r"$\mathrm{d}\sigma / \mathrm{d}(\sum T_n/q_0)$ [cm$^2$/nucleon]")


_events = -1

SAMPLES = [
    # (exp, flav, FSI file path, lep_pdg)
    ("DUNE", "numu",    "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root",    13),
    ("DUNE", "numubar", "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root",   13),
    ("HK",   "numu",    "../../Remade_April26/nuwro_25031_morestats/HK/HK_numu_FSI.flat.root",        13),
    ("HK",   "numubar", "../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root",     13),
    ("DUNE", "nue",     "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_nue_FSI.flat.root",     11),
    ("DUNE", "nuebar",  "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_nueb_FSI.flat.root",    11),
    ("HK",   "nue",     "../../Remade_April26/nuwro_25031_morestats/HK/HK_nue_FSI.flat.root",         11),
    ("HK",   "nuebar",  "../../Remade_April26/nuwro_25031_morestats/HK/HK_nuebar_FSI.flat.root",      11),
]

# For each (exp, flav) pair: compute FSI + noFSI histograms, share y-max,
# emit two single-panel PDFs (FSI and noFSI) with matching y-scales.
for exp, flav, fname_FSI, lep_pdg in SAMPLES:
    fname_noFSI = noFSI_path(fname_FSI)

    # Pass 1: load + bin both files
    pair = {}
    for fsi_state, fname in (("FSI", fname_FSI), ("noFSI", fname_noFSI)):
        d = compute_neutron_data(fname, _events, lep_pdg=lep_pdg)
        pair[fsi_state] = make_grouped_hists(d)

    # Shared y-max from the larger of FSI / noFSI Total + group histograms
    ymax = max(h.max() for state in pair for h in pair[state]) * 1.05

    # Pass 2: plot each side with shared y-scale + ratio panel showing each
    # subgroup's contribution to the total. No per-plot legend -- the shared
    # legend is written once at the end.
    for fsi_state in ("FSI", "noFSI"):
        fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))
        plot_grouped(ax, ax_ratio, pair[fsi_state], ymax)
        plt.savefig(outpath("Fig5_plots", f"Fig5_{exp}_EnergyFromNeutrons_{flav}_{fsi_state}.pdf"))
        plt.close(fig)


# Single horizontal-strip legend shared by all 8 Fig5 PDFs.
_legend_handles = [
    Line2D([0], [0], color=c, linewidth=lw, label=lbl)
    for c, lw, lbl in zip(FIG5_COLORS, FIG5_LINEWIDTHS, FIG5_LABELS)
]
save_strip_legend(_legend_handles,
                  out_dir=os.path.dirname(outpath("Fig5_plots", "_")),
                  fname="legend_Fig5_EnergyFromNeutrons",
                  fig_w=7.0, fig_h=0.5)
