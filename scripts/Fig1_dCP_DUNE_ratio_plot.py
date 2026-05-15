"""DUNE nue / nuebar appearance-channel ratio under dCP and energy-shift
variations. Two PDFs (Enu_true and Enu_had x-axes), each showing the
ratio R(E) = N_{nu_e} / N_{nu_e bar} for nominal + variants, plus a
variant/nominal double-ratio. Mirrors Fig1_dCP_DUNE_plot.py's layout
and reuses its bin spec.
"""
from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

# DUNE binning (matches Fig1_dCP_DUNE_plot.py)
bin_width = 100.0  # MeV
bins = np.arange(0, 6000 + bin_width, step=bin_width)
centers = 0.5 * (bins[:-1] + bins[1:])
L_DUNE = 1285.0


def _osc_prob(Enu_t_GeV, dCP_used, is_nubar):
    """Per-event P(numu(bar) -> nue(bar)) at the DUNE baseline (E in GeV)."""
    pmns.SetPath(L_DUNE, 2.8)
    pmns.SetMix(theta12, theta23, theta13, dCP_used)
    pmns.SetDeltaMsqrs(dm21, dm32)
    pmns.SetIsNuBar(bool(is_nubar))
    return np.array([pmns.Prob(1, 0, E, L_DUNE) for E in Enu_t_GeV])


def _scaled_counts(x_MeV, Enu_t_GeV, target, dCP_used, is_nubar):
    prob = _osc_prob(Enu_t_GeV, dCP_used, is_nubar)
    scale = target / float(prob.sum())
    counts, _ = np.histogram(x_MeV, weights=prob * scale, bins=bins)
    return counts


def _ratio(counts_nue, counts_nuebar):
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

    f_numu  = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root"
    f_nubar = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root"
    arr_numu  = load_arrays(f_numu,  max_events=1_000_000)
    arr_nubar = load_arrays(f_nubar, max_events=1_000_000)

    # For DUNE the Eν^had is built from the particle stack; use enu_had_arr.
    bias_wo_numu,  bias_with_numu,  cc_numu  = enu_had_arr(arr_numu,  vertex=False)
    bias_wo_nubar, bias_with_nubar, cc_nubar = enu_had_arr(arr_nubar, vertex=False)
    Enu_t_GeV_numu  = ak.to_numpy(arr_numu['Enu_true'])[np.asarray(cc_numu,  dtype=bool)]
    Enu_t_GeV_nubar = ak.to_numpy(arr_nubar['Enu_true'])[np.asarray(cc_nubar, dtype=bool)]
    Enu_t_MeV_numu  = Enu_t_GeV_numu  * 1000.0
    Enu_t_MeV_nubar = Enu_t_GeV_nubar * 1000.0

    if IsReco:
        # Enu_had (with charged-pion full E) — matches Fig1_dCP_DUNE_plot.py.
        x_numu_GeV  = bias_with_numu  + Enu_t_GeV_numu
        x_nubar_GeV = bias_with_nubar + Enu_t_GeV_nubar
        x_numu  = np.asarray(x_numu_GeV)  * 1000.0
        x_nubar = np.asarray(x_nubar_GeV) * 1000.0
        xlabel_str = r"$E_{\nu}^{\rm had}$ [MeV]"
        save_tag = "Enuhad"
    else:
        x_numu  = Enu_t_MeV_numu
        x_nubar = Enu_t_MeV_nubar
        xlabel_str = r"$E_{\nu}^{\rm true}$ [MeV]"
        save_tag = "EnuTrue"

    target_nue    = expected_events(f_numu,  channel='nue')
    target_nuebar = expected_events(f_nubar, channel='nuebar')

    def ratio_for(dCP_used, shift_MeV=0.0):
        xn  = x_numu  + shift_MeV if (IsReco and shift_MeV) else x_numu
        xnb = x_nubar + shift_MeV if (IsReco and shift_MeV) else x_nubar
        c_nue    = _scaled_counts(xn,  Enu_t_GeV_numu,  target_nue,    dCP_used, is_nubar=False)
        c_nuebar = _scaled_counts(xnb, Enu_t_GeV_nubar, target_nuebar, dCP_used, is_nubar=True)
        return _ratio(c_nue, c_nuebar)

    r_nom        = ratio_for(deltaCP)
    r_plus_dcp   = ratio_for(deltaCP + 20 * np.pi / 180)
    r_minus_dcp  = ratio_for(deltaCP - 20 * np.pi / 180)

    ax.step(centers, r_nom,       where='mid', color=tol_dark,      lw=1.6, label=r"Nominal $\delta_{CP} = -\pi/2$")
    ax.step(centers, r_plus_dcp,  where='mid', color=osc_inc_color, lw=1.5, label=r"$\delta_{CP} + 20^{\circ}$")
    ax.step(centers, r_minus_dcp, where='mid', color=osc_dec_color, lw=1.5, label=r"$\delta_{CP} - 20^{\circ}$")
    ax_ratio.hlines(1, 0, 6000, linestyle='--', color='black', lw=0.7)
    _step_double_ratio(ax_ratio, r_plus_dcp,  r_nom, osc_inc_color)
    _step_double_ratio(ax_ratio, r_minus_dcp, r_nom, osc_dec_color)

    if IsReco:
        r_plus_shift  = ratio_for(deltaCP, shift_MeV=+15.0)
        r_minus_shift = ratio_for(deltaCP, shift_MeV=-15.0)
        ax.step(centers, r_plus_shift,  where='mid', color=pastel_red,  lw=1.4, label=r"$E_{\nu}^{\rm had} + 15$ MeV")
        ax.step(centers, r_minus_shift, where='mid', color=pastel_blue, lw=1.4, label=r"$E_{\nu}^{\rm had} - 15$ MeV")
        _step_double_ratio(ax_ratio, r_plus_shift,  r_nom, pastel_red)
        _step_double_ratio(ax_ratio, r_minus_shift, r_nom, pastel_blue)

    ax.set_xlim(0, 6000)
    ax_ratio.set_xlim(0, 6000)
    ax_ratio.set_ylim(0.90, 1.10)
    ax.set_ylabel(r"$N_{\nu_{e}} / N_{\bar{\nu}_{e}}$")
    ax_ratio.set_xlabel(xlabel_str)
    ax_ratio.set_ylabel("variant / nominal")

    plt.savefig(outpath("Fig1_plots", f"Fig1_DUNE_ratio_dCP_{save_tag}.pdf"))
    plt.close(fig)


plot_ratio(IsReco=False)
plot_ratio(IsReco=True)
