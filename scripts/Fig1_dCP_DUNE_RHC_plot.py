"""DUNE dCP / dm32 spectrum -- RHC sibling of Fig1_dCP_DUNE_plot.py.

Uses the DUNE ν̄μ beam file and computes the antineutrino appearance
P(ν̄μ → ν̄e) for the dCP plots and the antineutrino disappearance
P(ν̄μ → ν̄μ) for the dm32 plots, via pmns.SetIsNuBar(True). Same osc
variations and same +-15 MeV reco-energy shift; output PDF names carry a
`_RHC` suffix and the plots are titled "RHC".
"""
from FlatTreeMod import *
from collections import defaultdict

custom_lines = []
labels = []

bin_width = 100.0  # MeV (DUNE Enu plot)

def plot_osc_reco(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom):
    bins = np.arange(0, 6000 + bin_width, step=bin_width)
    ax.hist(diff_sel, bins=bins, histtype='step', weights=weights, color=color, linewidth=1.5, label=label, linestyle='-')
    if nominal:
        countsn, _ = np.histogram(diff_sel, weights=weights, bins=bins)
        ax_ratio.hlines(1, 0, 6000, linestyle='--', color='black')
        return countsn
    else:
        counts, edges = np.histogram(diff_sel, weights=weights, bins=bins)
        ratio = counts[1:]/np.maximum(counts_nom[1:], 1e-30)
        ax_ratio.step(edges[1:-1], ratio, color=color, linestyle='-', where="mid")
        return


def plot_osc_true(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom):
    bins = np.arange(0, 6000 + bin_width, step=bin_width)
    ax.hist(diff_sel, bins=bins, histtype='step', weights=weights, color=color, linewidth=1.5, label=label, linestyle='-')
    if nominal:
        countsn, _ = np.histogram(diff_sel, weights=weights, bins=bins)
        ax_ratio.hlines(1, 0, 6000, linestyle='--', color='black')
        return countsn
    else:
        counts, edges = np.histogram(diff_sel, weights=weights, bins=bins)
        ratio = counts[1:]/np.maximum(counts_nom[1:], 1e-30)
        ax_ratio.step(edges[1:-1], ratio, color=color, linestyle='-', where="mid")
        return



def plot_EnuReco(nEvents: int, IsReco: bool, IsdCP: bool):
    fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(1, 1), hspace=0.07)
    plt.sca(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    filename = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root"
    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
    bias_wo_GeV, bias_with_GeV, valid = enu_had_arr(arr, vertex=False)
    Enu_t_GeV  = np.asarray(arr['Enu_true'])[valid]
    Enu_t_sel  = Enu_t_GeV * 1000.0   # MeV
    bias_wo_list   = (bias_wo_GeV   + Enu_t_GeV) * 1000.0  # MeV
    bias_with_list = (bias_with_GeV + Enu_t_GeV) * 1000.0  # MeV
    counts_nom = []

    L = 1285.0
    pmns.SetPath(L, 2.8)
    pmns.SetIsNuBar(True)   # RHC: antineutrino probabilities
    pmns.SetMix(theta12, theta23, theta13, deltaCP)
    pmns.SetDeltaMsqrs(dm21, dm32)
    prob_default_nue = np.array([pmns.Prob(1, 0, E, L) for E in Enu_t_GeV])  # ν̄μ → ν̄e

    dCP_new = deltaCP+(20*np.pi/180)
    pmns.SetMix(theta12, theta23, theta13, dCP_new)
    prob_plus_dcp = np.array([pmns.Prob(1, 0, E, L) for E in Enu_t_GeV])

    dCP_new = deltaCP-(20*np.pi/180)
    pmns.SetMix(theta12, theta23, theta13, dCP_new)
    prob_minus_dcp = np.array([pmns.Prob(1, 0, E, L) for E in Enu_t_GeV])

    pmns.SetMix(theta12, theta23, theta13, deltaCP)
    prob_default_numu = np.array([pmns.Prob(1, 1, E, L) for E in Enu_t_GeV])  # ν̄μ → ν̄μ

    dm32_new = dm32*1.004
    pmns.SetDeltaMsqrs(dm21, dm32_new)
    prob_plus_dm2 = np.array([pmns.Prob(1, 1, E, L) for E in Enu_t_GeV])

    dm32_new = dm32*0.996
    pmns.SetDeltaMsqrs(dm21, dm32_new)
    prob_minus_dm2 = np.array([pmns.Prob(1, 1, E, L) for E in Enu_t_GeV])

    if(IsReco == True):
        bins = np.arange(0, 6000 + bin_width, step=bin_width)
        if(IsdCP == True):
            target = expected_events(filename, channel='nuebar')
            scale = target / float(prob_default_nue.sum())
            prob_default_nue = prob_default_nue * scale
            prob_plus_dcp    = prob_plus_dcp    * scale
            prob_minus_dcp   = prob_minus_dcp   * scale
            counts_nom = plot_osc_reco(ax, ax_ratio, bias_with_list, r"Nominal $\delta_{CP} = -\pi/2$", tol_dark, prob_default_nue, True, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list, r"$\delta_{CP} + 20^{\circ}$", osc_inc_color, prob_plus_dcp, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list, r"$\delta_{CP} - 20^{\circ}$", osc_dec_color, prob_minus_dcp, False, counts_nom)
            plot_osc_shift_e(ax, ax_ratio, shift=+15.0, label=r"$E_{\nu}^{\rm had} + 15$ MeV",
                             color=pastel_red, counts_nom=counts_nom, bins=bins,
                             x_unshifted=bias_with_list, weights=prob_default_nue)
            plot_osc_shift_e(ax, ax_ratio, shift=-15.0, label=r"$E_{\nu}^{\rm had} - 15$ MeV",
                             color=pastel_blue, counts_nom=counts_nom, bins=bins,
                             x_unshifted=bias_with_list, weights=prob_default_nue)
            ax.set_xlim(0, 6000); ax_ratio.set_xlim(0, 6000)
            ax_ratio.set_xlabel(r"$E_{\nu}^{\rm had}$ [MeV]")
            ax.set_ylabel(EVENT_RATE_LABEL)
            ax_ratio.set_ylim(0.90, 1.1)
            plt.savefig(outpath("Fig1_plots", "Fig1_DUNE_Enuhad_dCP_RHC.pdf"))

        else:
            target = expected_events(filename, channel='numubar')
            scale = target / float(prob_default_numu.sum())
            prob_default_numu = prob_default_numu * scale
            prob_plus_dm2     = prob_plus_dm2     * scale
            prob_minus_dm2    = prob_minus_dm2    * scale
            counts_nom = plot_osc_reco(ax, ax_ratio, bias_with_list, r"Nominal $\Delta m^{2}_{32} = 2.437 \times 10^{-3}$ eV$^{2}$", tol_dark, prob_default_numu, True, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list, r"$\Delta m^{2}_{32} + 0.4\%$", osc_inc_color, prob_plus_dm2, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, bias_with_list, r"$\Delta m^{2}_{32} - 0.4\%$", osc_dec_color, prob_minus_dm2, False, counts_nom)
            plot_osc_shift_e(ax, ax_ratio, shift=+15.0, label=r"$E_{\nu}^{\rm had} + 15$ MeV",
                             color=pastel_red, counts_nom=counts_nom, bins=bins,
                             x_unshifted=bias_with_list, weights=prob_default_numu)
            plot_osc_shift_e(ax, ax_ratio, shift=-15.0, label=r"$E_{\nu}^{\rm had} - 15$ MeV",
                             color=pastel_blue, counts_nom=counts_nom, bins=bins,
                             x_unshifted=bias_with_list, weights=prob_default_numu)
            ax.set_xlim(0, 6000); ax_ratio.set_xlim(0, 6000)
            ax_ratio.set_xlabel(r"$E_{\nu}^{\rm had}$ [MeV]")
            ax.set_ylabel(EVENT_RATE_LABEL)
            ax_ratio.set_ylim(0.90, 1.1)
            plt.savefig(outpath("Fig1_plots", "Fig1_DUNE_Enuhad_dm32_RHC.pdf"))
    else:
        if(IsdCP == True):
            counts_nom = plot_osc_reco(ax, ax_ratio, Enu_t_sel, r"Nominal $\delta_{CP} = -\pi/2$", tol_dark, prob_default_nue, True, counts_nom)
            plot_osc_reco(ax, ax_ratio, Enu_t_sel, r"$\delta_{CP} + 20^{\circ}$", osc_inc_color, prob_plus_dcp, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, Enu_t_sel, r"$\delta_{CP} - 20^{\circ}$", osc_dec_color, prob_minus_dcp, False, counts_nom)

            ax_ratio.set_xlabel(r"$E_{\nu}^{\rm true}$ [MeV]")
            ax.set_ylabel(EVENT_RATE_LABEL)
            ax_ratio.set_ylim(0.90,1.1)
            plt.savefig(outpath("Fig1_plots", "Fig1_DUNE_EnuTrue_dCP_RHC.pdf"))

        else:
            counts_nom = plot_osc_reco(ax, ax_ratio, Enu_t_sel, r"Nominal $\Delta m^{2}_{32} = 2.437 \times 10^{-3}$ eV$^{2}$", tol_dark, prob_default_numu, True, counts_nom)
            plot_osc_reco(ax, ax_ratio, Enu_t_sel, r"$\Delta m^{2}_{32} + 0.4\%$", osc_inc_color, prob_plus_dm2, False, counts_nom)
            plot_osc_reco(ax, ax_ratio, Enu_t_sel, r"$\Delta m^{2}_{32} - 0.4\%$", osc_dec_color, prob_minus_dm2, False, counts_nom)

            ax_ratio.set_xlabel(r"$E_{\nu}^{\rm true}$ [MeV]")
            ax.set_ylabel(EVENT_RATE_LABEL)
            ax_ratio.set_ylim(0.90,1.1)
            plt.savefig(outpath("Fig1_plots", "Fig1_DUNE_EnuTrue_dm32_RHC.pdf"))
    plt.close(fig)
    return



plot_EnuReco(nEvents = 1000000, IsReco=False, IsdCP = False)
plot_EnuReco(nEvents = 1000000, IsReco=False, IsdCP = True)
plot_EnuReco(nEvents = 1000000, IsReco=True,  IsdCP = False)
plot_EnuReco(nEvents = 1000000, IsReco=True,  IsdCP = True)
