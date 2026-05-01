from FlatTreeMod import *
from collections import defaultdict

custom_lines = []
labels = []

bin_width = 0.1

def plot_osc_reco(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom):
    bins = np.arange(0, 8, step=bin_width)
    ax.hist(diff_sel, bins=bins, histtype='step', weights=weights, color=color,linewidth=1.5, label = label, linestyle = '-')
    # custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
    # labels.append(label)

    if(nominal == True):
        countsn, _ = np.histogram(diff_sel, weights=weights, bins=bins)
        ax_ratio.hlines(1, 0, 8, linestyle='--', color = 'black')
        return countsn
    else:
        counts, edges = np.histogram(diff_sel, weights=weights, bins=bins)
        ratio = counts[1:]/counts_nom[1:]
        ax_ratio.step(edges[1:-1], ratio, color=color, linestyle='-', where="mid")
        return
        

def plot_osc_true(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom):
    bins = np.arange(0, 5, step=bin_width)
    ax.hist(diff_sel, bins=bins, histtype='step', weights=weights, color=color,linewidth=1.5, label = label, linestyle = '-')
    # custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
    # labels.append(label)

    if(nominal == True):
        countsn, _ = np.histogram(diff_sel, weights=weights, bins=bins)
        ax_ratio.hlines(1, 0, 5, linestyle='--', color = 'black')
        return countsn
    else:
        counts, edges = np.histogram(diff_sel, weights=weights, bins=bins)
        ratio = counts[1:]/counts_nom[1:]
        ax_ratio.step(edges[1:-1], ratio, color=color, linestyle='-', where="mid")
        return
   


def plot_EnuReco(nEvents: int, IsReco: bool, IsdCP: bool):
    ## Set axis
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.07)
    ax = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax)
    plt.sca(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    filename = "../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root"
    fin = ROOT.TFile.Open(filename)
    tree = fin.Get("FlatTree_VARS")

    # ---------------------------------
    # Event loop
    # ---------------------------------
    nentries    = tree.GetEntries()
    Enu_t_sel      = []
    bias_wo_list   = []
    bias_with_list = []
    counts_nom  = []
    nevs = 0
    if(nEvents == -1):
        nevs = nentries
    else:
        nevs = nEvents

    for i in range(nevs):
        tree.GetEntry(i)
        has_neutron = False
        bad_event = False
        ELep     = tree.ELep
        Enu_true = tree.Enu_true
        nfsp     = tree.nfsp
        mode     = tree.Mode

        E  = tree.E
        px = tree.px
        py = tree.py
        pz = tree.pz
        pdg = tree.pdg

        # -------------------------
        # Lepton energy
        # -------------------------
        enuhad_wo   = ELep
        enuhad_with = ELep

        # Loop over final state particles
        for j in range(nfsp):

            apdg = abs(int(pdg[j]))
            Ej   = float(E[j])
            pxj  = float(px[j])
            pyj  = float(py[j])
            pzj  = float(pz[j])

            p2 = pxj*pxj + pyj*pyj + pzj*pzj

            # Check neutron
            if apdg == 2112:
                has_neutron = True

            # -------------------------
            # Remove heavy stuff
            # -------------------------
            if apdg > 3000:
                bad_event = True
                continue

            # -------------------------
            # Heavy baryons
            # -------------------------
            if 2300 < apdg < 3000:
                enuhad_wo   += Ej
                enuhad_with += Ej
                continue

            # -------------------------
            # Definition 1
            # -------------------------
            if (apdg == 11 or (17 < apdg < 2000)) and apdg != 211:

                enuhad_wo += Ej

            elif apdg in (2212, 211):

                mass2 = Ej*Ej - p2

                if mass2 > 0:
                    enuhad_wo += Ej - np.sqrt(mass2)

            # -------------------------
            # Definition 2
            # -------------------------
            if (apdg == 11 or (17 < apdg < 2000)):

                enuhad_with += Ej

            elif apdg == 2212:

                mass2 = Ej*Ej - p2

                if mass2 > 0:
                    enuhad_with += Ej - np.sqrt(mass2)

        # -------------------------
        # Fill
        # -------------------------
        bias_wo   = enuhad_wo   #- Enu_true
        bias_with = enuhad_with #- Enu_true

        if(bad_event == False):
            # Total
            bias_wo_list.append(bias_wo)
            bias_with_list.append(bias_with)
            Enu_t_sel.append(Enu_true)

        else:
            continue

        # By neutron content
    # ---------------------------------
    # Write output
    # ---------------------------------
    Enu_t_sel = np.array(Enu_t_sel)
    bias_wo_list = np.array(bias_wo_list)
    bias_with_list = np.array(bias_with_list)
    
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
    L = 1285.0
    pmns.SetPath(L, 2.8)
    prob_default_nue = np.array([pmns.Prob(1, 0, E, L) for E in Enu_t_sel])  # νμ → νe
    # prob_default_nue = np.array([1.0 for E in Enu_t_sel])  # νμ → νe

    ## Increase dCP by + 20deg
    dCP_new = deltaCP+(20*np.pi/180)
    pmns.SetMix(theta12, theta23, theta13, dCP_new)
    prob_plus_dcp = np.array([pmns.Prob(1, 0, E, L) for E in Enu_t_sel])  # νμ → νe

    ## Decrease dCP by  20deg
    dCP_new = deltaCP-(20*np.pi/180)
    pmns.SetMix(theta12, theta23, theta13, dCP_new)
    prob_minus_dcp = np.array([pmns.Prob(1, 0, E, L) for E in Enu_t_sel])  # νμ → νe 

    pmns.SetMix(theta12, theta23, theta13, deltaCP)
    prob_default_numu = np.array([pmns.Prob(1, 1, E, L) for E in Enu_t_sel])  # νμ → νe

    ## Increase dm23^2 by +0.4%
    dm32_new = dm32*1.004
    pmns.SetDeltaMsqrs(dm21, dm32_new)
    prob_plus_dm2 = np.array([pmns.Prob(1, 1, E, L) for E in Enu_t_sel])  # νμ → νμ survival

    ## Decrease dm23^2 by +0.4%
    dm32_new = dm32*0.996
    pmns.SetDeltaMsqrs(dm21, dm32_new)
    prob_minus_dm2 = np.array([pmns.Prob(1, 1, E, L) for E in Enu_t_sel])  # νμ → νμ survival

    if(IsReco == True):
        if(IsdCP == True):
            counts_nom = plot_osc_reco(ax, ax_ratio, bias_with_list, "default PMNS", vivid_purple, prob_default_nue, True, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list, "Inc dCP", light_green, prob_plus_dcp, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list, "Dec dCP", dark_green, prob_minus_dcp, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list+(15/1000), "default PMNS, +15MeV shift", dark_blue, prob_default_nue, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list-(15/1000), "default PMNS, -15MeV shift", dark_red, prob_default_nue, False, counts_nom)

            ax.legend(loc = 'upper right')
            ax_ratio.set_xlabel(r"$E_{\nu}^{\text{\text{had}}}$ [MeV]")
            ax.set_ylabel("Number of events")
            ax_ratio.set_ylim(0.90,1.1)
            plt.savefig("Fig1_plots/Fig1_DUNE_Enuhad_dCP.pdf")

        else:
            counts_nom = plot_osc_reco(ax, ax_ratio, bias_with_list, "default PMNS", vivid_purple, prob_default_numu, True, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list, "Inc dm32", light_green, prob_plus_dm2, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list, "Dec dm32", dark_green, prob_minus_dm2, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list+(15/1000), "default PMNS, +15MeV shift", dark_blue, prob_default_numu, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list-(15/1000), "default PMNS, -15MeV shift", dark_red, prob_default_numu, False, counts_nom)

            ax.legend(loc = 'upper right')
            ax_ratio.set_xlabel(r"$E_{\nu}^{\text{\text{had}}}$ [MeV]")
            ax.set_ylabel("Number of events")
            ax_ratio.set_ylim(0.90,1.1)
            plt.savefig("Fig1_plots/Fig1_DUNE_Enuhad_dm32.pdf")
    else:
        if(IsdCP == True):
            counts_nom = plot_osc_reco(ax, ax_ratio, Enu_t_sel, "default PMNS", vivid_purple, prob_default_nue, True, counts_nom)
            plot_osc_reco(ax, ax_ratio, Enu_t_sel, "Inc dCP", light_green, prob_plus_dcp, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, Enu_t_sel, "Dec dCP", dark_green, prob_minus_dcp, False, counts_nom)

            ax.legend(loc = 'upper right')
            ax_ratio.set_xlabel(r"$E_{\nu}^{\text{\text{had}}}$ [MeV]")
            ax.set_ylabel("Number of events")
            ax_ratio.set_ylim(0.90,1.1)
            plt.savefig("Fig1_plots/Fig1_DUNE_EnuTrue_dCP.pdf")

        else:
            counts_nom = plot_osc_reco(ax, ax_ratio, Enu_t_sel, "default PMNS", vivid_purple, prob_default_numu, True, counts_nom)
            plot_osc_reco(ax, ax_ratio, Enu_t_sel, "Inc dm32", light_green, prob_plus_dm2, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, Enu_t_sel, "Dec dm32", dark_green, prob_minus_dm2, False, counts_nom)

            ax.legend(loc = 'upper right')
            ax_ratio.set_xlabel(r"$E_{\nu}^{\text{\text{true}}}$ [MeV]")
            ax.set_ylabel("Number of events")
            ax_ratio.set_ylim(0.90,1.1)
            plt.savefig("Fig1_plots/Fig1_DUNE_EnuTrue_dm32.pdf")
    # plt.show()

    return



# plot_EnuReco(nEvents = -1, IsdCP = True)
# plot_EnuReco(nEvents = -1, IsReco=False, IsdCP = False)
# plot_EnuReco(nEvents = -1, IsReco=False, IsdCP = True)
plot_EnuReco(nEvents = 1000000, IsReco=True, IsdCP = False) # issue
# plot_EnuReco(nEvents = -1, IsReco=True, IsdCP = True)





