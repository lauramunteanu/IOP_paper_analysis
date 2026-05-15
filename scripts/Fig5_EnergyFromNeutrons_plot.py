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


def compute_neutron_data(filename, nEvents):
    """Per-event (n_neutrons, x = ΣTn/q0, fScaleFactor). After CC0pi-style cuts."""
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
    return dict(
        n_neutrons=nneutron_per_evt[keep],
        x=neutron_KE[keep] / q0[keep],
        fScaleFactor=fScaleFactor,
    )


def make_grouped_hists(d):
    """Returns list of histogram arrays, one per group + total."""
    weight = d["fScaleFactor"] / bin_width
    hists = []
    for label, sel_fn in GROUPS:
        mask = sel_fn(d["n_neutrons"])
        x = d["x"][mask]
        w = np.full(len(x), weight, dtype=float)
        h, _ = np.histogram(x, bins=bins, weights=w)
        hists.append(h)
    # Total = all events
    h_tot, _ = np.histogram(d["x"], bins=bins,
                            weights=np.full(len(d["x"]), weight, dtype=float))
    hists.append(h_tot)
    return hists


def plot_grouped(ax, ax_ratio, hists, fsi_label, ymax):
    colors = list(TOL_MUTED[:3]) + [tol_dark]
    labels = [g[0] for g in GROUPS] + ["Total"]
    linewidths = [1.2, 1.2, 1.2, 1.8]
    for h, col, lw in zip(hists, colors, linewidths):
        ax.step(centers, h, where="mid", color=col, linewidth=lw)

    # Ratio panel: each subgroup / total. Shows how much each Nn contributes
    # to the total at each x-bin (sums to 1 across the 3 subgroup lines).
    h_total = hists[-1]
    safe_total = np.where(h_total > 0, h_total, np.nan)
    for h, col, lw in zip(hists[:-1], colors[:-1], linewidths[:-1]):
        ax_ratio.step(centers, h / safe_total, where="mid", color=col, linewidth=lw)
    ax_ratio.set_ylim(0, 1.05)
    ax_ratio.set_ylabel("fraction\nof total")

    # Legend reordered for 2-col layout
    #   Total | N=2
    #   N=1   | N>=3
    # matplotlib fills column-major, so handles must be in column-major order:
    # [Total, N=1, N=2, N>=3].
    handles = [Line2D([0], [0], color=c, linewidth=lw)
               for c, lw in zip(colors, linewidths)]
    reorder = [3, 0, 1, 2]
    ax.legend([handles[i] for i in reorder], [labels[i] for i in reorder],
              title=fsi_label, loc="upper right", ncol=2, frameon=False)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax_ratio.set_xlabel(r"$\sum T_n / q_0$")
    ax.set_ylabel(r"$\mathrm{d}\sigma / \mathrm{d}(\sum T_n/q_0)$ [cm$^2$/nucleon]")


_events = -1

SAMPLES = [
    ("DUNE", "numu",    "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root"),
    ("DUNE", "numubar", "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root"),
    ("HK",   "numu",    "../../Remade_April26/nuwro_25031_morestats/HK/HK_numu_FSI.flat.root"),
    ("HK",   "numubar", "../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root"),
]

# For each (exp, flav) pair: compute FSI + noFSI histograms, share y-max,
# emit two single-panel PDFs (FSI and noFSI) with matching y-scales.
for exp, flav, fname_FSI in SAMPLES:
    fname_noFSI = noFSI_path(fname_FSI)

    # Pass 1: load + bin both files
    pair = {}
    for fsi_state, fname in (("FSI", fname_FSI), ("noFSI", fname_noFSI)):
        d = compute_neutron_data(fname, _events)
        pair[fsi_state] = make_grouped_hists(d)

    # Shared y-max from the larger of FSI / noFSI Total + group histograms
    ymax = max(h.max() for state in pair for h in pair[state]) * 1.05

    # Pass 2: plot each side with shared y-scale + ratio panel showing each
    # subgroup's contribution to the total.
    for fsi_state in ("FSI", "noFSI"):
        fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))
        legend_title = "FSI" if fsi_state == "FSI" else "no FSI"
        plot_grouped(ax, ax_ratio, pair[fsi_state], legend_title, ymax)
        plt.savefig(f"Fig5_plots/Fig5_{exp}_EnergyFromNeutrons_{flav}_{fsi_state}.pdf")
        plt.close(fig)
