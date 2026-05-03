from FlatTreeMod import *
from matplotlib.lines import Line2D
import ROOT

ROOT.gROOT.SetBatch(True)

Mneutron = 939.565  # MeV

def plot_neutron_energy_stacked(ax, filename, nEvents, title_label):
    """Caller passes either an FSI or a noFSI sample file. We always read the
    post-FSI stack (vertex=False); for a noFSI file no cascade ran so that's
    equivalent to the noFSI view. Avoids the binding-energy bookkeeping bias
    inherent in reading the vertex stack of an FSI file."""
    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
    fScaleFactor = float(arr['fScaleFactor'][0])  # constant per file

    n, pdg, E, _, _, _ = particles_arr(arr, vertex=False)
    apdg = abs(pdg)
    is_neutron = apdg == 2112
    nneutron_per_evt = ak.to_numpy(ak.sum(is_neutron, axis=1))
    # neutron KE per event: sum (E*1000 - Mneutron) over neutrons
    neutron_E_MeV = E * 1000.0
    KE_per_neutron = ak.where(is_neutron, neutron_E_MeV - Mneutron, 0.0)
    neutron_KE = ak.to_numpy(ak.sum(KE_per_neutron, axis=1))
    bad_event = ak.to_numpy(ak.any(apdg > 3000, axis=1))
    Enu  = ak.to_numpy(arr['Enu_true'])
    ELep = ak.to_numpy(arr['ELep'])
    q0   = (Enu - ELep) * 1000.0

    keep = (~bad_event) & (neutron_KE != 0) & (q0 != 0)
    xvals    = neutron_KE[keep] / q0[keep]
    n_neutrons = nneutron_per_evt[keep]

    # Bucket by neutron count
    values_by_n = {}
    for nn in np.unique(n_neutrons):
        values_by_n[int(nn)] = xvals[n_neutrons == nn]

    neutron_numbers = sorted(values_by_n.keys())
    data = [values_by_n[n] for n in neutron_numbers[0:8]]

    # Histogram setup
    bin_width = 0.01
    bins = np.arange(0, 1 + bin_width, step=bin_width)

    weights = [
        np.full(len(arr), fScaleFactor / bin_width, dtype=float)
        for arr in data
    ]

    # 8 CB-friendly colours from FlatTreeMod's Tol-muted palette.
    colors = TOL_MUTED[:len(data)]

    # Step lines per neutron-multiplicity (no fill, paper style).
    for d, w, c, n in zip(data, weights, colors, neutron_numbers):
        ax.hist(d, bins=bins, weights=w, histtype='step',
                color=c, linewidth=1.4, label=fr"$N_n={n}$")

    ax.set_xlim(0, 1)
    ax.set_ylabel(r"$\mathrm{d}\sigma / \mathrm{d}(\sum T_n/q_0)$ [cm$^2$/nucleon]")
    ax.text(
        0.98, 0.95, title_label,
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black")
    )

    return neutron_numbers


_events = -1

# All four samples: DUNE × {numu, numubar} and HK × {numu, numubar}.
# Each call produces a 2-panel (noFSI top, FSI bottom) PDF.
SAMPLES = [
    ("DUNE", "numu",    "../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root"),
    ("DUNE", "numubar", "../../Remade_April26/DUNE/DUNE_numub_FSI.flat.root"),
    ("HK",   "numu",    "../../Remade_April26/HK/HK_numu_FSI.flat.root"),
    ("HK",   "numubar", "../../Remade_April26/HK/HK_numubar_FSI.flat.root"),
]

for exp, flav, fname_FSI in SAMPLES:
    fname_noFSI = noFSI_path(fname_FSI)

    fig, (ax_top, ax_bot) = make_fig_stacked('double_stacked', sharex=True, sharey=True, hspace=0.08)

    plot_neutron_energy_stacked(
        ax=ax_top, filename=fname_noFSI,
        nEvents=_events, title_label="No FSI",
    )
    plot_neutron_energy_stacked(
        ax=ax_bot, filename=fname_FSI,
        nEvents=_events, title_label="FSI",
    )
    ax_bot.set_xlabel(r"$\sum T_n / q_0$")

    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.92), ncol=4)

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(f"Fig5_plots/Fig5_{exp}_EnergyFromNeutrons_{flav}_stacked.pdf")
    plt.close(fig)