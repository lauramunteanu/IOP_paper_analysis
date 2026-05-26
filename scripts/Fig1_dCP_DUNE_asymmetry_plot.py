"""DUNE nue / nuebar appearance-channel asymmetry under dCP and energy-shift
variations. Two PDFs (Enu_true and Enu_had x-axes), each showing the
asymmetry A(E) = (N_{nu_e} - N_{nu_e bar}) / (N_{nu_e} + N_{nu_e bar}) for
nominal + variants, plus a variant−nominal ΔA panel. Mirrors
Fig1_dCP_DUNE_ratio_plot.py's layout and reuses its bin spec.
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


def _asymmetry(counts_nue, counts_nuebar):
    """Bin-wise asymmetry (N_νe - N_ν̄e)/(N_νe + N_ν̄e)."""
    denom = counts_nue + counts_nuebar
    safe = np.where(denom > 0, denom, np.nan)
    return (counts_nue - counts_nuebar) / safe


def _step_delta_asym(ax_delta, variant_asym, nominal_asym, color):
    """Bottom panel: ΔA = A_variant − A_nominal (additive)."""
    delta = variant_asym - nominal_asym
    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    ax_delta.step(centers, delta, where="mid", color=color, lw=1.5)


def plot_asymmetry(IsReco):
    # Bin width: 100 MeV for Eν^reco (Fig 1 paper convention), 125 MeV
    # for Eν^true (matches DUNE flux native binning).
    global bin_width, bins, centers
    bin_width = 100.0 if IsReco else 125.0
    bins = np.arange(0, 6000 + bin_width, step=bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, (ax, ax_delta) = make_fig_ratio('single_ratio', height_ratios=(1, 1), hspace=0.07)
    plt.sca(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    f_numu  = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_nue_FSI.flat.root"
    f_nubar = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_nueb_FSI.flat.root"
    arr_numu  = load_arrays(f_numu,  max_events=1_000_000)
    arr_nubar = load_arrays(f_nubar, max_events=1_000_000)

    # For DUNE the Eν^had is built from the particle stack; use enu_had_arr.
    bias_wo_numu,  bias_with_numu,  cc_numu  = enu_had_arr(arr_numu,  vertex=False, lep_pdg=11)
    bias_wo_nubar, bias_with_nubar, cc_nubar = enu_had_arr(arr_nubar, vertex=False, lep_pdg=11)
    Enu_t_GeV_numu  = ak.to_numpy(arr_numu['Enu_true'])[np.asarray(cc_numu,  dtype=bool)]
    Enu_t_GeV_nubar = ak.to_numpy(arr_nubar['Enu_true'])[np.asarray(cc_nubar, dtype=bool)]
    Enu_t_MeV_numu  = Enu_t_GeV_numu  * 1000.0
    Enu_t_MeV_nubar = Enu_t_GeV_nubar * 1000.0

    if IsReco:
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

    def counts_pair(dCP_used):
        c_nue    = _scaled_counts(x_numu,  Enu_t_GeV_numu,  target_nue,    dCP_used, is_nubar=False)
        c_nuebar = _scaled_counts(x_nubar, Enu_t_GeV_nubar, target_nuebar, dCP_used, is_nubar=True)
        return c_nue, c_nuebar

    c_nue_nom, c_nuebar_nom = counts_pair(deltaCP)
    a_nom = _asymmetry(c_nue_nom, c_nuebar_nom)

    c_p_n, c_p_nb = counts_pair(deltaCP + 20 * np.pi / 180)
    c_m_n, c_m_nb = counts_pair(deltaCP - 20 * np.pi / 180)
    a_plus_dcp  = _asymmetry(c_p_n, c_p_nb)
    a_minus_dcp = _asymmetry(c_m_n, c_m_nb)

    ax.step(centers, a_nom,       where='mid', color=tol_dark,      lw=1.6, label=r"Nominal $\delta_{CP} = -\pi/2$")
    ax.step(centers, a_plus_dcp,  where='mid', color=osc_inc_color, lw=1.5, label=r"$\delta_{CP} + 20^{\circ}$")
    ax.step(centers, a_minus_dcp, where='mid', color=osc_dec_color, lw=1.5, label=r"$\delta_{CP} - 20^{\circ}$")
    ax_delta.hlines(0, 0, 6000, linestyle='--', color='black', lw=0.7)
    _step_delta_asym(ax_delta, a_plus_dcp,  a_nom, osc_inc_color)
    _step_delta_asym(ax_delta, a_minus_dcp, a_nom, osc_dec_color)

    if IsReco:
        # Smooth-shift via Taylor (same approach as Fig1_dCP_DUNE_ratio_plot.py).
        shift = 15.0  # MeV
        sh_nue_p = c_nue_nom    * smooth_shift_ratio(c_nue_nom,    bins, +shift)
        sh_nub_p = c_nuebar_nom * smooth_shift_ratio(c_nuebar_nom, bins, +shift)
        sh_nue_m = c_nue_nom    * smooth_shift_ratio(c_nue_nom,    bins, -shift)
        sh_nub_m = c_nuebar_nom * smooth_shift_ratio(c_nuebar_nom, bins, -shift)
        a_plus_shift  = _asymmetry(sh_nue_p, sh_nub_p)
        a_minus_shift = _asymmetry(sh_nue_m, sh_nub_m)
        ax.step(centers, a_plus_shift,  where='mid', color=pastel_red,  lw=1.4, label=r"$E_{\nu}^{\rm had} + 15$ MeV")
        ax.step(centers, a_minus_shift, where='mid', color=pastel_blue, lw=1.4, label=r"$E_{\nu}^{\rm had} - 15$ MeV")
        _step_delta_asym(ax_delta, a_plus_shift,  a_nom, pastel_red)
        _step_delta_asym(ax_delta, a_minus_shift, a_nom, pastel_blue)

    ax.set_xlim(0, 6000)
    ax_delta.set_xlim(0, 6000)
    ax_delta.set_ylim(-0.05, 0.05)
    ax.set_ylabel(r"$(N_{\nu_e} - N_{\bar\nu_e}) / (N_{\nu_e} + N_{\bar\nu_e})$")
    ax_delta.set_xlabel(xlabel_str)
    ax_delta.set_ylabel(r"variant $-$ nominal")

    plt.savefig(outpath("Fig1_plots", f"Fig1_DUNE_asymmetry_dCP_{save_tag}.pdf"))
    plt.close(fig)


plot_asymmetry(IsReco=False)
plot_asymmetry(IsReco=True)
