"""HK dCP spectrum -- RHC + relative-energy-shift sibling.

Combines the RHC channel (ν̄μ beam, P(ν̄μ → ν̄e)) of Fig1_dCP_RHC_plot.py
with the per-event ±0.5%·E_ν^true shift of Fig1_dCP_relshift_plot.py.
Output PDF names carry both `_RHC` and `_relshift` suffixes; plots are
titled "RHC".
"""
from FlatTreeMod import *

custom_lines = []
labels = []

bin_width = 20
bins = np.arange(0, 2000, step=bin_width)
centers = 0.5 * (bins[:-1] + bins[1:])
SHIFT_FRAC = 0.005   # 0.5% * Enu_true


def plot_osc_reco(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom):
    ax.hist(diff_sel, bins=bins, histtype='step', weights=weights, color=color,
            linewidth=1.5, label=label, linestyle='-')
    custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
    labels.append(label)
    if nominal:
        countsn, _ = np.histogram(diff_sel, weights=weights, bins=bins)
        ax_ratio.hlines(1, 0, 2000, linestyle='--', color='black')
        return countsn
    counts, edges = np.histogram(diff_sel, weights=weights, bins=bins)
    ratio = counts[1:] / counts_nom[1:]
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ax_ratio.step(edges[1:-1], ratio, color=color, linestyle='-', where="mid")


def plot_osc_true(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom):
    return plot_osc_reco(ax, ax_ratio, diff_sel, label, color, weights, nominal, counts_nom)


def plot_EnuReco(filename, nEvents, IsReco):
    # Bin width: 20 MeV for Eν^QE (Fig 1 paper convention), 50 MeV for
    # Eν^true (matches T2K flux native binning).
    global bin_width, bins
    bin_width = 20 if IsReco else 50
    bins = np.arange(0, 2000, step=bin_width)
    fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(1, 1), hspace=0.07)
    plt.sca(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
    flag = is_cc0pi_arr(arr, vertex=False, lep_pdg=11)
    Enu_t_sel  = np.asarray(arr['Enu_true'])[flag] * 1000.0
    Enu_QE_sel = np.asarray(arr['Enu_QE'])  [flag] * 1000.0
    counts_nom = []

    L = 295.0
    pmns.SetPath(L, 2.8)
    pmns.SetIsNuBar(True)   # RHC: P(ν̄μ → ν̄e)
    pmns.SetMix(theta12, theta23, theta13, deltaCP)
    pmns.SetDeltaMsqrs(dm21, dm32)
    prob_default_nue = np.array([pmns.Prob(1, 0, E / 1000.0, L) for E in Enu_t_sel])

    pmns.SetMix(theta12, theta23, theta13, deltaCP + 20 * np.pi / 180)
    prob_plus_dcp = np.array([pmns.Prob(1, 0, E / 1000.0, L) for E in Enu_t_sel])

    pmns.SetMix(theta12, theta23, theta13, deltaCP - 20 * np.pi / 180)
    prob_minus_dcp = np.array([pmns.Prob(1, 0, E / 1000.0, L) for E in Enu_t_sel])

    target = expected_events(filename, channel='nuebar')
    scale = target / float(prob_default_nue.sum())
    prob_default_nue = prob_default_nue * scale
    prob_plus_dcp    = prob_plus_dcp    * scale
    prob_minus_dcp   = prob_minus_dcp   * scale

    if IsReco:
        counts_nom = plot_osc_reco(ax, ax_ratio, Enu_QE_sel, r"Nominal $\delta_{CP} = -\pi/2$", tol_dark,       prob_default_nue, True, counts_nom)
        plot_osc_reco(ax, ax_ratio, Enu_QE_sel, r"$\delta_{CP} + 20^{\circ}$", osc_inc_color, prob_plus_dcp,  False, counts_nom)
        plot_osc_reco(ax, ax_ratio, Enu_QE_sel, r"$\delta_{CP} - 20^{\circ}$", osc_dec_color, prob_minus_dcp, False, counts_nom)
        plot_osc_shift_e(ax, ax_ratio, shift=+SHIFT_FRAC * centers,
                         label=r"$E_{\nu}^{\rm QE} + 0.5\%\, E_{\nu}^{\rm true}$",
                         color=pastel_red, counts_nom=counts_nom, bins=bins,
                         x_unshifted=Enu_QE_sel, weights=prob_default_nue)
        plot_osc_shift_e(ax, ax_ratio, shift=-SHIFT_FRAC * centers,
                         label=r"$E_{\nu}^{\rm QE} - 0.5\%\, E_{\nu}^{\rm true}$",
                         color=pastel_blue, counts_nom=counts_nom, bins=bins,
                         x_unshifted=Enu_QE_sel, weights=prob_default_nue)
        ax_ratio.set_xlabel(r"$E_{\nu}^{\rm QE}$ [MeV]")
        ax.set_ylabel(EVENT_RATE_LABEL)
        ax.set_xlim(0, 1200)
        ax_ratio.set_xlim(0, 1200)
        ax_ratio.set_ylim(0.90, 1.1)
        plt.savefig(outpath("Fig1_plots", "Fig1_EnuQE_dCP_RHC_relshift.pdf"))
        plt.close(fig)
    else:
        counts_nom = plot_osc_true(ax, ax_ratio, Enu_t_sel, r"Nominal $\delta_{CP} = -\pi/2$", tol_dark,       prob_default_nue, True, counts_nom)
        plot_osc_true(ax, ax_ratio, Enu_t_sel, r"$\delta_{CP} + 20^{\circ}$", osc_inc_color, prob_plus_dcp,  False, counts_nom)
        plot_osc_true(ax, ax_ratio, Enu_t_sel, r"$\delta_{CP} - 20^{\circ}$", osc_dec_color, prob_minus_dcp, False, counts_nom)
        ax_ratio.set_xlabel(r"$E_{\nu}^{\rm True}$ [MeV]")
        ax.set_ylabel(EVENT_RATE_LABEL)
        ax.set_xlim(0, 1200)
        ax_ratio.set_xlim(0, 1200)
        ax_ratio.set_ylim(0.90, 1.1)
        plt.savefig(outpath("Fig1_plots", "Fig1_EnuTrue_dCP_RHC_relshift.pdf"))
        plt.close(fig)


plot_EnuReco("../../Remade_April26/nuwro_25031_morestats/HK/HK_nuebar_FSI.flat.root", nEvents=1_000_000, IsReco=False)
plot_EnuReco("../../Remade_April26/nuwro_25031_morestats/HK/HK_nuebar_FSI.flat.root", nEvents=1_000_000, IsReco=True)
