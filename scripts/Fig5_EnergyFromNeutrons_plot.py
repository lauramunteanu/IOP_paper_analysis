from FlatTreeMod import *
from matplotlib.lines import Line2D
import ROOT

ROOT.gROOT.SetBatch(True)

Mneutron = 939.565  # MeV

def plot_neutron_energy_stacked(ax, filename, nEvents, title_label):
    Print(f"Reading: {filename}")

    fin = ROOT.TFile.Open(filename)
    tree = fin.Get("FlatTree_VARS")

    nentries = tree.GetEntries()
    nevs = nentries if nEvents == -1 else min(nEvents, nentries)

    tree.GetEntry(0)
    fScaleFactor = tree.fScaleFactor

    values_by_n = {}

    max_n = 0

    for i in range(nevs):
        tree.GetEntry(i)
        bad_event = False

        nfsp = tree.nfsp
        E = tree.E
        pdg = tree.pdg
        Enu = tree.Enu_true
        ELep = tree.ELep

        q0 = (Enu - ELep) * 1000.0

        neutron_KE = 0.0
        nneutron = 0

        for j in range(nfsp):
            apdg = abs(int(pdg[j]))
            Ej = float(E[j]) * 1000.0

            if apdg > 3000:
                bad_event = True
                continue

            if apdg == 2112:
                nneutron += 1
                neutron_KE += (Ej - Mneutron)

        if bad_event:
            continue

        if neutron_KE != 0 and q0 != 0:
            xval = neutron_KE / q0

            if nneutron not in values_by_n:
                values_by_n[nneutron] = []

            values_by_n[nneutron].append(xval)

        max_n = max(max_n, nneutron)

    fin.Close()

    # Convert to sorted lists for stacked histogram
    neutron_numbers = sorted(values_by_n.keys())
    data = [np.array(values_by_n[n]) for n in neutron_numbers[0:8]]

    # Histogram setup
    bin_width = 0.01
    bins = np.arange(0, 1 + bin_width, step=bin_width)

    weights = [
        np.full(len(arr), fScaleFactor / bin_width, dtype=float)
        for arr in data
    ]

    colors = [
        "#4C78A8",  
        "#F58518",  
        "#54A24B",  
        "#E45756",  
        "#B279A2",  
        "#FF9DA6",  
        "#9D755D",  
        "#BAB0AC",  
    ]
    colors = colors[:len(data)]

    ax.hist(
        data,
        bins=bins,
        weights=weights,
        stacked=True,
        histtype="barstacked",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        label=[fr"$N_n={n}$" for n in neutron_numbers],
    )

    ax.set_xlim(0, 1)
    ax.set_ylabel(r"$\mathrm{d}\sigma / \mathrm{d}(\sum T_n/q_0)$ [cm$^2$/nucleon]")
    ax.text(
        0.98, 0.95, title_label,
        transform=ax.transAxes,
        ha="right", va="top", fontsize=20,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black")
    )

    return neutron_numbers


_events = -1

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    sharex=True, sharey=True,
    figsize=(8, 8),
    gridspec_kw={"hspace": 0.08}
)

plot_neutron_energy_stacked(
    ax=ax_top,
    filename="../../Remade_April26/DUNE/DUNE_numu_noFSI.flat.root",
    nEvents=_events,
    title_label="No FSI"
)

plot_neutron_energy_stacked(
    ax=ax_bot,
    filename="../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root",
    nEvents=_events,
    title_label="FSI"
)

ax_bot.set_xlabel(r"$\sum T_n / q_0$")

handles, labels = ax_top.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5,0.85), fontsize=15)

plt.tight_layout()
plt.savefig("Fig5_plots/Fig6_DUNE_EnergyFromNeutrons_numu_stacked.pdf")
plt.show()