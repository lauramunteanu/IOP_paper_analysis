"""HK nue / nuebar appearance-channel ratio under dCP and energy-shift
variations. Two PDFs (Enu_true and Enu_reco x-axes), each showing:

  Top panel:  R(E) = N_{nu_e}(E) / N_{nu_e bar}(E)  for nominal + variants.
              N_{nu_e}    = HK numu sample weighted by P(numu->nue), scaled to
                            EXPECTED_EVENTS[('HK','nue')].
              N_{nu_e bar} = HK numubar sample weighted by P(numubar->nuebar)
                            (with IsNuBar=True), scaled to ...['nuebar'].
              Variants: dCP +-20deg, Enu^QE shift +-5 MeV (Reco panel only).

  Bottom:     R_variant(E) / R_nominal(E) (double-ratio) -- shows how each
              systematic skews the appearance asymmetry that drives dCP.

Mirrors Fig1_dCP_plot.py's layout/conventions; reuses make_Fig1_legends
strip legends for the standalone-legend slot in LaTeX.
"""
from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

# HK binning (matches Fig1_dCP_plot.py)
bin_width = 20
bins = np.arange(0, 2000, step=bin_width)
centers = 0.5 * (bins[:-1] + bins[1:])
L_HK = 295.0


def _osc_prob(Enu_t_MeV, dCP_used, is_nubar):
    """Per-event P(numu(bar) -> nue(bar)) at the HK baseline, given dCP."""
    pmns.SetPath(L_HK, 2.8)
    pmns.SetMix(theta12, theta23, theta13, dCP_used)
    pmns.SetDeltaMsqrs(dm21, dm32)
    pmns.SetIsNuBar(bool(is_nubar))
    return np.array([pmns.Prob(1, 0, E / 1000.0, L_HK) for E in Enu_t_MeV])


def _scaled_counts(x_MeV, Enu_t_MeV, target, dCP_used, is_nubar):
    """Histogram of `x_MeV` weighted by per-event osc prob, scaled to
    `target` events (so the histogram integrates to `target`)."""
    prob = _osc_prob(Enu_t_MeV, dCP_used, is_nubar)
    scale = target / float(prob.sum())
    counts, _ = np.histogram(x_MeV, weights=prob * scale, bins=bins)
    return counts


def _ratio(counts_nue, counts_nuebar):
    """Bin-wise nue / nuebar ratio with safe NaN on empty bins."""
    safe = np.where(counts_nuebar > 0, counts_nuebar, np.nan)
    return counts_nue / safe


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

    def ratio_for(dCP_used, shift_MeV=0.0):
        xn  = x_numu  + shift_MeV if (IsReco and shift_MeV) else x_numu
        xnb = x_nubar + shift_MeV if (IsReco and shift_MeV) else x_nubar
        c_nue    = _scaled_counts(xn,  Enu_t_numu,  target_nue,    dCP_used, is_nubar=False)
        c_nuebar = _scaled_counts(xnb, Enu_t_nubar, target_nuebar, dCP_used, is_nubar=True)
        return _ratio(c_nue, c_nuebar)

    r_nom        = ratio_for(deltaCP)
    r_plus_dcp   = ratio_for(deltaCP + 20 * np.pi / 180)
    r_minus_dcp  = ratio_for(deltaCP - 20 * np.pi / 180)

    # Top panel — nominal + dCP variants
    ax.step(centers, r_nom,       where='mid', color=tol_dark,      lw=1.6, label=r"Nominal $\delta_{CP} = -\pi/2$")
    ax.step(centers, r_plus_dcp,  where='mid', color=osc_inc_color, lw=1.5, label=r"$\delta_{CP} + 20^{\circ}$")
    ax.step(centers, r_minus_dcp, where='mid', color=osc_dec_color, lw=1.5, label=r"$\delta_{CP} - 20^{\circ}$")
    ax_ratio.hlines(1, 0, 2000, linestyle='--', color='black', lw=0.7)
    _step_double_ratio(ax_ratio, r_plus_dcp,  r_nom, osc_inc_color)
    _step_double_ratio(ax_ratio, r_minus_dcp, r_nom, osc_dec_color)

    if IsReco:
        r_plus_shift  = ratio_for(deltaCP, shift_MeV=+5.0)
        r_minus_shift = ratio_for(deltaCP, shift_MeV=-5.0)
        ax.step(centers, r_plus_shift,  where='mid', color=pastel_red,  lw=1.4, label=r"$E_{\nu}^{\rm QE} + 5$ MeV")
        ax.step(centers, r_minus_shift, where='mid', color=pastel_blue, lw=1.4, label=r"$E_{\nu}^{\rm QE} - 5$ MeV")
        _step_double_ratio(ax_ratio, r_plus_shift,  r_nom, pastel_red)
        _step_double_ratio(ax_ratio, r_minus_shift, r_nom, pastel_blue)

    ax.set_xlim(0, 1200)
    ax_ratio.set_xlim(0, 1200)
    ax_ratio.set_ylim(0.90, 1.10)
    ax.set_ylabel(r"$N_{\nu_{e}} / N_{\bar{\nu}_{e}}$")
    ax_ratio.set_xlabel(xlabel_str)
    ax_ratio.set_ylabel("variant / nominal")

    plt.savefig(outpath("Fig1_plots", f"Fig1_HK_ratio_dCP_{save_tag}.pdf"))
    plt.close(fig)


plot_ratio(IsReco=False)
plot_ratio(IsReco=True)
