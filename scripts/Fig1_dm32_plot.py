from FlatTreeMod import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

custom_lines = []
labels = []

bin_width = 20
bins = np.arange(0, 2000, step=bin_width)

def plot_osc_reco(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom):
    ax.hist(diff_sel, bins=bins, histtype='step', weights=weights, color=color,linewidth=1.5, label = label, linestyle = '-')
    custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
    labels.append(label)

    if(nominal == True):
        countsn, _ = np.histogram(diff_sel, weights=weights, bins=(bins))
        ax_ratio.hlines(1, 0, 2000, linestyle='--', color = 'black')
        return countsn
    else:
        counts, edges = np.histogram(diff_sel, weights=weights, bins=(bins))
        ratio = counts[1:]/counts_nom[1:]
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        ax_ratio.step(edges[1:-1], ratio, color=color, linestyle='-', where="mid")
        return
        

def plot_osc_true(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom):
    ax.hist(diff_sel, bins=bins, histtype='step', weights=weights, color=color,linewidth=1.5, label = label, linestyle = '-')
    custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
    labels.append(label)

    if(nominal == True):
        countsn, _ = np.histogram(diff_sel, weights=weights, bins=(bins))
        ax_ratio.hlines(1, 0, 2000, linestyle='--', color = 'black')
        return countsn
    else:
        counts, edges = np.histogram(diff_sel, weights=weights, bins=(bins))
        ratio = counts[1:]/counts_nom[1:]
        ax_ratio.step(edges[1:-1], ratio, color=color, linestyle='-', where="mid")
        return
   


def plot_EnuReco(filename: str, nEvents: int, IsReco: bool):
    ## Set axis
    fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(1, 1), hspace=0.07)
    plt.sca(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
    flag = is_cc0pi_arr(arr, vertex=False)
    Enu_t_sel  = np.asarray(arr['Enu_true'])[flag] * 1000.0
    Enu_QE_sel = np.asarray(arr['Enu_QE'])  [flag] * 1000.0
    diff_sel   = Enu_QE_sel - Enu_t_sel
    Enu_QE_sel_p5 = Enu_QE_sel + 5
    counts_nom = []

    # ----------------------------------------
    # Arrays to hold oscillation probs for
    # different oscillation parameters
    # ----------------------------------------
    prob_plus_dm2    = []
    prob_minus_dm2   = []

    # ----------------------------------------
    # Compute oscillation weights PER EVENT
    # (convert MeV -> GeV for OscProb)
    # ----------------------------------------
    L = 295.0
    pmns.SetPath(L, 2.8)  # HK baseline; reset path in case a prior call (e.g. DUNE) re-pathed pmns
    pmns.SetMix(theta12, theta23, theta13, deltaCP)
    pmns.SetDeltaMsqrs(dm21, dm32)  # reset, in case a previous call left pmns mutated
    prob_default_numu = np.array([pmns.Prob(1, 1, E/1000.0, L) for E in Enu_t_sel])  # νμ → νe

    ## Increase dm23^2 by +0.4%
    dm32_new = dm32*1.004
    pmns.SetDeltaMsqrs(dm21, dm32_new)
    prob_plus_dm2 = np.array([pmns.Prob(1, 1, E/1000.0, L) for E in Enu_t_sel])  # νμ → νμ survival

    ## Decrease dm23^2 by +0.4%
    dm32_new = dm32*0.996
    pmns.SetDeltaMsqrs(dm21, dm32_new)
    prob_minus_dm2 = np.array([pmns.Prob(1, 1, E/1000.0, L) for E in Enu_t_sel])  # νμ → νμ survival

    # ----------------------------------------
    # Scale to expected event yield: HK νμ→νμ survival channel.
    # ----------------------------------------
    target = expected_events(filename, channel='numu')
    sum_prob = float(prob_default_numu.sum())
    scale = target / sum_prob / bin_width
    prob_default_numu = prob_default_numu * scale
    prob_plus_dm2     = prob_plus_dm2     * scale
    prob_minus_dm2    = prob_minus_dm2    * scale

    if(IsReco == True):
        counts_nom = plot_osc_reco(ax, ax_ratio, Enu_QE_sel, "default PMNS", vivid_purple, prob_default_numu, True, counts_nom)
        plot_osc_reco(ax, ax_ratio, Enu_QE_sel, "Inc dm32", light_green, prob_plus_dm2, False, counts_nom)
        plot_osc_reco(ax, ax_ratio, Enu_QE_sel, "Dec dm32", dark_green, prob_minus_dm2, False, counts_nom)
        plot_osc_shift_e(ax, ax_ratio, shift=+5, label="+5 MeV shift",
                         color=dark_blue, counts_nom=counts_nom, bins=bins,
                         x_unshifted=Enu_QE_sel, weights=prob_default_numu)
        plot_osc_shift_e(ax, ax_ratio, shift=-5, label="-5 MeV shift",
                         color=dark_red,  counts_nom=counts_nom, bins=bins,
                         x_unshifted=Enu_QE_sel, weights=prob_default_numu)

        ax.legend(custom_lines, labels, loc = 'upper right')
        ax_ratio.set_xlabel(r"$E_{\nu}^{\text{\text{QE}}}$ [MeV]")
        ax.set_ylabel(EVENT_RATE_LABEL)

        ax.set_xlim(0,1200)
        ax_ratio.set_xlim(0,1200)
        ax_ratio.set_ylim(0.95,1.05)
        plt.savefig("Fig1_plots/Fig1_EnuQE_dm32.pdf")
        plt.close(fig)

    else:
        counts_nom = plot_osc_true(ax, ax_ratio, Enu_t_sel, "default PMNS", vivid_purple, prob_default_numu, True, counts_nom)
        plot_osc_true(ax, ax_ratio, Enu_t_sel, "Inc dm32", light_green, prob_plus_dm2, False, counts_nom)
        plot_osc_true(ax, ax_ratio, Enu_t_sel, "Dec dm32", dark_green, prob_minus_dm2, False, counts_nom)

        ax.legend(custom_lines, labels, loc = 'upper right')
        ax_ratio.set_xlabel(r"$E_{\nu}^{\text{\text{True}}}$ [MeV]")
        ax.set_ylabel(EVENT_RATE_LABEL)
        ax.set_xlim(0,1200)
        ax_ratio.set_xlim(0,1200)
        ax_ratio.set_ylim(0.95,1.05)
        plt.savefig("Fig1_plots/Fig1_EnuTrue_dm32.pdf")
        plt.close(fig)
    return



# plot_EnuReco("../../Remade_April26/HK/HK_numu_FSI.flat.root", nEvents = 2000000, IsReco = True)
plot_EnuReco("../../Remade_April26/HK/HK_numu_FSI.flat.root", nEvents = 1000000, IsReco = False)
# plot_EnuReco(nEvents = 5000000, IsReco = True)





