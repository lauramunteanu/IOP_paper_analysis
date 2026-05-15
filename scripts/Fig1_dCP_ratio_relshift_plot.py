"""HK nue / nuebar appearance-ratio with relative-energy-shift systematic
(+-0.5% * Enu_true). Sibling of Fig1_dCP_ratio_plot.py: identical dCP
variants, but the +-5 MeV constant shift is replaced by a per-event
fractional shift via smooth_shift_ratio with a per-bin shift array.
"""
from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

bin_width = 20
bins = np.arange(0, 2000, step=bin_width)
centers = 0.5 * (bins[:-1] + bins[1:])
L_HK = 295.0
SHIFT_FRAC = 0.005


def _osc_prob(Enu_t_MeV, dCP_used, is_nubar):
    pmns.SetPath(L_HK, 2.8)
    pmns.SetMix(theta12, theta23, theta13, dCP_used)
    pmns.SetDeltaMsqrs(dm21, dm32)
    pmns.SetIsNuBar(bool(is_nubar))
    return np.array([pmns.Prob(1, 0, E / 1000.0, L_HK) for E in Enu_t_MeV])


def _scaled_counts(x_MeV, Enu_t_MeV, target, dCP_used, is_nubar):
    prob = _osc_prob(Enu_t_MeV, dCP_used, is_nubar)
    scale = target / float(prob.sum())
    counts, _ = np.histogram(x_MeV, weights=prob * scale, bins=bins)
    return counts


def _ratio(c_nue, c_nuebar):
    safe = np.where(c_nuebar > 0, c_nuebar, np.nan)
    return c_nue / safe


def _step_double_ratio(ax_ratio, variant_ratio, nominal_ratio, color):
    safe_nom = np.where(np.isfinite(nominal_ratio) & (nominal_ratio > 0),
                        nominal_ratio, np.nan)
    dr = variant_ratio / safe_nom
    dr = np.nan_to_num(dr, nan=1.0, posinf=1.0, neginf=1.0)
    ax_ratio.step(centers, dr, where="mid", color=color, lw=1.5)


def plot_ratio(IsReco):
    fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(1, 1), hspace=0.07)
    plt.sca(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    f_numu  = "../../Remade_April26/nuwro_25031_morestats/HK/HK_numu_FSI.flat.root"
    f_nubar = "../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root"
    arr_numu  = load_arrays(f_numu,  max_events=1_000_000)
    arr_nubar = load_arrays(f_nubar, max_events=1_000_000)

    flag_numu  = is_cc0pi_arr(arr_numu,  vertex=False)
    flag_nubar = is_cc0pi_arr(arr_nubar, vertex=False)

    Enu_t_numu  = np.asarray(arr_numu['Enu_true'])[flag_numu]  * 1000.0
    Enu_t_nubar = np.asarray(arr_nubar['Enu_true'])[flag_nubar] * 1000.0

    if IsReco:
        x_numu  = np.asarray(arr_numu['Enu_QE'])[flag_numu]   * 1000.0
        x_nubar = np.asarray(arr_nubar['Enu_QE'])[flag_nubar] * 1000.0
        xlabel_str = r"$E_{\nu}^{\rm QE}$ [MeV]"
        save_tag = "EnuQE"
    else:
        x_numu  = Enu_t_numu
        x_nubar = Enu_t_nubar
        xlabel_str = r"$E_{\nu}^{\rm true}$ [MeV]"
        save_tag = "EnuTrue"

    target_nue    = expected_events(f_numu,  channel='nue')
    target_nuebar = expected_events(f_nubar, channel='nuebar')

    def counts_pair(dCP_used):
        c_nue    = _scaled_counts(x_numu,  Enu_t_numu,  target_nue,    dCP_used, is_nubar=False)
        c_nuebar = _scaled_counts(x_nubar, Enu_t_nubar, target_nuebar, dCP_used, is_nubar=True)
        return c_nue, c_nuebar

    c_nue_nom, c_nuebar_nom = counts_pair(deltaCP)
    r_nom = _ratio(c_nue_nom, c_nuebar_nom)

    c_p_n, c_p_nb = counts_pair(deltaCP + 20 * np.pi / 180)
    c_m_n, c_m_nb = counts_pair(deltaCP - 20 * np.pi / 180)
    r_plus_dcp  = _ratio(c_p_n, c_p_nb)
    r_minus_dcp = _ratio(c_m_n, c_m_nb)

    ax.step(centers, r_nom,       where='mid', color=tol_dark,      lw=1.6, label=r"Nominal $\delta_{CP} = -\pi/2$")
    ax.step(centers, r_plus_dcp,  where='mid', color=osc_inc_color, lw=1.5, label=r"$\delta_{CP} + 20^{\circ}$")
    ax.step(centers, r_minus_dcp, where='mid', color=osc_dec_color, lw=1.5, label=r"$\delta_{CP} - 20^{\circ}$")
    ax_ratio.hlines(1, 0, 2000, linestyle='--', color='black', lw=0.7)
    _step_double_ratio(ax_ratio, r_plus_dcp,  r_nom, osc_inc_color)
    _step_double_ratio(ax_ratio, r_minus_dcp, r_nom, osc_dec_color)

    if IsReco:
        # Per-bin Taylor shift: shift_array = SHIFT_FRAC * centers (in MeV).
        sh = SHIFT_FRAC * centers
        sh_nue_p = c_nue_nom    * smooth_shift_ratio(c_nue_nom,    bins, +sh)
        sh_nub_p = c_nuebar_nom * smooth_shift_ratio(c_nuebar_nom, bins, +sh)
        sh_nue_m = c_nue_nom    * smooth_shift_ratio(c_nue_nom,    bins, -sh)
        sh_nub_m = c_nuebar_nom * smooth_shift_ratio(c_nuebar_nom, bins, -sh)
        r_plus_shift  = _ratio(sh_nue_p, sh_nub_p)
        r_minus_shift = _ratio(sh_nue_m, sh_nub_m)
        ax.step(centers, r_plus_shift,  where='mid', color=pastel_red,  lw=1.4, label=r"$E_{\nu}^{\rm QE} + 0.5\%\, E_{\nu}^{\rm true}$")
        ax.step(centers, r_minus_shift, where='mid', color=pastel_blue, lw=1.4, label=r"$E_{\nu}^{\rm QE} - 0.5\%\, E_{\nu}^{\rm true}$")
        _step_double_ratio(ax_ratio, r_plus_shift,  r_nom, pastel_red)
        _step_double_ratio(ax_ratio, r_minus_shift, r_nom, pastel_blue)

    ax.set_xlim(0, 1200)
    ax_ratio.set_xlim(0, 1200)
    ax_ratio.set_ylim(0.90, 1.10)
    ax.set_ylabel(r"$N_{\nu_{e}} / N_{\bar{\nu}_{e}}$")
    ax_ratio.set_xlabel(xlabel_str)
    ax_ratio.set_ylabel("variant / nominal")

    plt.savefig(outpath("Fig1_plots", f"Fig1_HK_ratio_dCP_relshift_{save_tag}.pdf"))
    plt.close(fig)


plot_ratio(IsReco=False)
plot_ratio(IsReco=True)
