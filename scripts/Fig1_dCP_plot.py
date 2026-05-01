from FlatTreeMod import *

custom_lines = []
labels = []

bin_width = 20
bins = (np.arange(0, 2000, step=bin_width))

def plot_osc_reco(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom):
    ax.hist(diff_sel, bins=np.arange(0, 2000, step=bin_width), histtype='step', weights=weights, color=color,linewidth=1.5, label = label, linestyle = '-')
    custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
    labels.append(label)

    if(nominal == True):
        countsn, _ = np.histogram(diff_sel, weights=weights, bins=bins)
        ax_ratio.hlines(1, 0, 2000, linestyle='--', color = 'black')
        return countsn
    else:
        counts, edges = np.histogram(diff_sel, weights=weights, bins=bins)
        ratio = counts[1:]/counts_nom[1:]
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        ax_ratio.step(edges[1:-1], ratio, color=color, linestyle='-', where="mid")
        return
        

def plot_osc_true(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom):
    ax.hist(diff_sel, bins=np.arange(0, 2000, step=bin_width), histtype='step', weights=weights, color=color,linewidth=1.5, label = label, linestyle = '-')
    custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
    labels.append(label)

    if(nominal == True):
        countsn, _ = np.histogram(diff_sel, weights=weights, bins=bins)
        ax_ratio.hlines(1, 0, 2000, linestyle='--', color = 'black')
        return countsn
    else:
        counts, edges = np.histogram(diff_sel, weights=weights, bins=bins)
        ratio = counts[1:]/counts_nom[1:]
        ax_ratio.step(edges[1:-1], ratio, color=color, linestyle='-', where="mid")
        return
   


def plot_EnuReco(filename: str, nEvents: int, IsReco: bool):
    ## Set axis
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.07)
    ax = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax)
    plt.sca(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    filename = filename
    Print(f"Reading: {filename}")
    fin = ROOT.TFile.Open(filename)
    tree = fin.Get("FlatTree_VARS")

    # ---------------------------------
    # Event loop
    # ---------------------------------
    nentries    = tree.GetEntries()
    diff_sel    = []
    Enu_t_sel   = []
    Enu_QE_sel   = []
    counts_nom  = []
    nevs = 0
    if(nEvents == -1):
        nevs = nentries
    else:
        nevs = nEvents

    for i in range(nEvents):
        tree.GetEntry(i)

        Enu_true = tree.Enu_true * 1000
        Enu_QE   = tree.Enu_QE * 1000
        pdg      = tree.pdg
        nfsp     = tree.nfsp
        mode     = tree.Mode
        isCC0pi     = tree.flagCC0pi

        if(isCC0pi == True):
            diff = Enu_QE - Enu_true
            diff_sel.append(diff)
            Enu_t_sel.append(Enu_true)
            Enu_QE_sel.append(Enu_QE)

    diff_sel    = np.array(diff_sel) 
    Enu_t_sel   = np.array(Enu_t_sel)
    Enu_QE_sel  = np.array(Enu_QE_sel)

    # ----------------------------------------
    # Arrays to hold oscillation probs for
    # different oscillation parameters
    # ----------------------------------------
    prob_plus_dcp    = []
    prob_minus_dcp   = []

    # ----------------------------------------
    # Compute oscillation weights PER EVENT
    # (convert MeV -> GeV for OscProb)
    # ----------------------------------------
    prob_default_nue = np.array([pmns.Prob(1, 0, E/1000.0, L) for E in Enu_t_sel])  # νμ → νe

    ## Increase dCP by + 20deg
    dCP_new = deltaCP+(20*np.pi/180)
    pmns.SetMix(theta12, theta23, theta13, dCP_new)
    prob_plus_dcp = np.array([pmns.Prob(1, 0, E/1000.0, L) for E in Enu_t_sel])  # νμ → νe

    ## Decrease dCP by  20deg
    dCP_new = deltaCP-(20*np.pi/180)
    pmns.SetMix(theta12, theta23, theta13, dCP_new)
    prob_minus_dcp = np.array([pmns.Prob(1, 0, E/1000.0, L) for E in Enu_t_sel])  # νμ → νe 

    if(IsReco == True):
        counts_nom = plot_osc_reco(ax, ax_ratio, Enu_QE_sel, "default PMNS", vivid_purple, prob_default_nue, True, counts_nom)
        plot_osc_reco(ax, ax_ratio, Enu_QE_sel, "Inc dCP", light_green, prob_plus_dcp, False, counts_nom)
        plot_osc_reco(ax, ax_ratio, Enu_QE_sel, "Dec dCP", dark_green, prob_minus_dcp, False, counts_nom)
        plot_osc_reco(ax, ax_ratio, Enu_QE_sel+5, "default PMNS, 5MeV shift", dark_blue, prob_default_nue, False, counts_nom)
        plot_osc_reco(ax, ax_ratio, Enu_QE_sel-5, "default PMNS, -5MeV shift", dark_red, prob_default_nue, False, counts_nom)

        # ax.legend(custom_lines, labels, loc = 'lower right')
        ax.legend(loc = 'upper right')
        ax_ratio.set_xlabel(r"$E_{\nu}^{\text{\text{QE}}}$ [MeV]")
        ax.set_ylabel("Number of events")

        ax.set_xlim(0,1200)
        ax_ratio.set_xlim(0,1200)
        ax_ratio.set_ylim(0.90,1.1)
        plt.savefig("Fig1_plots/Fig1_EnuQE_dCP.pdf")

    else:
        counts_nom = plot_osc_true(ax, ax_ratio, Enu_t_sel, "default PMNS", vivid_purple, prob_default_nue, True, counts_nom)
        plot_osc_true(ax, ax_ratio, Enu_t_sel, "Inc dCP", light_green, prob_plus_dcp, False, counts_nom)
        plot_osc_true(ax, ax_ratio, Enu_t_sel, "Dec dCP", dark_green, prob_minus_dcp, False, counts_nom)

        # ax.legend(custom_lines, labels, loc = 'lower right')
        ax.legend(loc = 'upper right')
        ax_ratio.set_xlabel(r"$E_{\nu}^{\text{\text{True}}}$ [MeV]")
        ax.set_ylabel("Number of events")
        ax.set_xlim(0,1200)
        ax_ratio.set_xlim(0,1200)
        ax_ratio.set_ylim(0.90,1.1)
        plt.savefig("Fig1_plots/Fig1_EnuTrue_dCP.pdf")
    plt.show()

    return



# plot_EnuReco("../../Remade_April26/HK/HK_numu_FSI.flat2.root", nEvents = 2000000, IsReco = True)
plot_EnuReco("../../Remade_April26/HK/HK_numu_FSI.flat2.root", nEvents = 1000000, IsReco = False)
# plot_EnuReco(nEvents = 5000000, IsReco = True)





