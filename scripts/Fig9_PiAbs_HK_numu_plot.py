"""Fig 10 — HK FHC νμ disappearance, comparing pi_abs ±31% variants and
Δm²₃₂ ±0.4% osc-parameter variations against the NuWro nominal CC0π
dσ/dE distribution.

  · Top:    dσ/dE for nominal + 2 pi_abs variants + 2 osc-param variants
  · Bottom: variant / nominal ratio

CC0π selection (first FS particle = μ); per-event weight =
fScaleFactor × P(νμ→νμ) / ⟨P_osc⟩_Φ × DSIGMA_DE_SCALE / bin_width.
Osc-param variations re-evaluate P(νμ→νμ) at Δm²₃₂ × (1 ± 0.4%).
"""
from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

# Match laura_extras Fig9_PiAbs palette + Fig 1 osc colours.
COL_PIABS_PLUS  = "#D55E00"   # COL_VERMILION  (+31%)
COL_PIABS_MINUS = "#0072B2"   # COL_BLUE       (-31%)

# Binning matches Fig 1 / Fig 9 HK (0-2000 MeV, 20 MeV bins).
BIN_WIDTH = 20.0
BINS = np.arange(0, 2000, step=BIN_WIDTH)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])
EDGES = BINS

L_HK = 295.0

# ⟨P_osc⟩_Φ is computed once at nominal parameters and reused as the
# denominator for every variant (matching the convention validated in
# Bias_level_study/validate_osc_fscalefactor.py).
POSC_AVG = posc_flux_avg('HK', 'numu', lep_pdg=13)


def _spectrum_with_osc(filename, *, dm32_factor=1.0):
    """dσ/dE [10⁻⁴² cm²/nucleon/MeV] for HK CC0π νμ events from ``filename``,
    weighted by P(νμ→νμ) at Δm²₃₂ × dm32_factor."""
    arr = load_arrays(filename, max_events=None)
    sel = is_cc0pi_arr(arr, vertex=False, lep_pdg=13)
    Enu_QE_MeV = np.asarray(arr['Enu_QE'])[sel] * 1000.0
    Enu_t_GeV  = np.asarray(arr['Enu_true'])[sel]
    fSF        = np.asarray(arr['fScaleFactor'])[sel]

    # Set pmns to the desired Δm²₃₂ scaling and evaluate per-event prob.
    pmns.SetPath(L_HK, 2.8)
    pmns.SetIsNuBar(False)
    pmns.SetMix(theta12, theta23, theta13, deltaCP)
    pmns.SetDeltaMsqrs(dm21, dm32 * dm32_factor)
    prob = np.array([pmns.Prob(1, 1, float(E), L_HK) for E in Enu_t_GeV])

    w = fSF * prob / POSC_AVG * DSIGMA_DE_SCALE / BIN_WIDTH
    counts, _ = np.histogram(Enu_QE_MeV, bins=BINS, weights=w)
    return counts


# ----------------------------------------------------------------------
BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/nuwro_25031_morestats/HK"
F_NOM   = f"{BASE}/HK_numu_FSI.flat.root"
F_PLUS  = f"{BASE}/piabs/HK_numu_piabs131_FSI.flat.root"   # piabs +31%
F_MINUS = f"{BASE}/piabs/HK_numu_piabs069_FSI.flat.root"   # piabs -31%

c_nom         = _spectrum_with_osc(F_NOM)
c_piabs_plus  = _spectrum_with_osc(F_PLUS)
c_piabs_minus = _spectrum_with_osc(F_MINUS)
c_dm_plus     = _spectrum_with_osc(F_NOM, dm32_factor=1.004)
c_dm_minus    = _spectrum_with_osc(F_NOM, dm32_factor=0.996)

# ----------------------------------------------------------------------
custom_lines, labels = [], []
fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))


def _draw(counts, color, label, ls='-', nominal=False, counts_nom=None):
    ax.hist(CENTERS, bins=BINS, histtype='step', weights=counts,
            color=color, linewidth=1.5, linestyle=ls, label=label)
    custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle=ls))
    labels.append(label)
    if nominal:
        ax_ratio.hlines(1, BINS[0], BINS[-1], linestyle='--', color='black')
        return counts
    ratio = counts / np.where(counts_nom > 0, counts_nom, np.nan)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ax_ratio.step(EDGES, np.r_[ratio, ratio[-1]],
                  color=color, linestyle=ls, where='post')


# Nominal first so the ratio reference is correct.
cn = _draw(c_nom,         tol_dark,       "Nominal", nominal=True)
_draw(c_piabs_plus,       COL_PIABS_PLUS, r"$\pi_{\rm abs}$ +31\%", counts_nom=cn)
_draw(c_piabs_minus,      COL_PIABS_MINUS, r"$\pi_{\rm abs}$ -31\%", counts_nom=cn)
_draw(c_dm_plus,          osc_inc_color,  r"$\Delta m^2_{32}\,+0.4\%$", counts_nom=cn)
_draw(c_dm_minus,         osc_dec_color,  r"$\Delta m^2_{32}\,-0.4\%$", counts_nom=cn)

ax.legend(custom_lines, labels, loc='upper right', fontsize=11)
ax.set_xlim(0, 1200)
ax_ratio.set_xlim(0, 1200)
ax.set_ylim(bottom=0)
ax.set_ylabel(DSIGMA_DE_LABEL)
ax_ratio.set_xlabel(r"$E_\nu^{\rm QE}$ [MeV]")
ax_ratio.set_ylabel("variant / nominal")
ax_ratio.set_ylim(0.9, 1.1)

plt.savefig(outpath("Fig9_plots", "Fig9_HK_PiAbs_numu.pdf"))
plt.close(fig)
