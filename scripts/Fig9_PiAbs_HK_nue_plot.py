"""Fig 10 — HK FHC νμ→νe appearance, comparing pi_abs ±31% variants and
δ_CP ±20° osc-parameter variations against the NuWro nominal CC0π
dσ/dE distribution.

Mirror of Fig10_PiAbs_HK_numu_plot.py, but using the νe sample and the
νμ→νe appearance osc weight. Selection: CC0π with lep_pdg=11 (first FS
particle = e). Δm²₃₂ variations replaced with δ_CP ±20° since that is the
osc parameter the appearance probability is most sensitive to at HK.
"""
from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

COL_PIABS_PLUS  = "#D55E00"   # COL_VERMILION  (+31%)
COL_PIABS_MINUS = "#0072B2"   # COL_BLUE       (-31%)

BIN_WIDTH = 20.0
BINS = np.arange(0, 2000, step=BIN_WIDTH)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])
EDGES = BINS

L_HK = 295.0
POSC_AVG = posc_flux_avg('HK', 'nue', lep_pdg=11)


def _spectrum_with_osc(filename, *, dCP_off_deg=0.0):
    """dσ/dE [10⁻⁴² cm²/nucleon/MeV] for HK CC0π νe events from ``filename``,
    weighted by P(νμ→νe) at δ_CP + dCP_off_deg."""
    arr = load_arrays(filename, max_events=None)
    sel = is_cc0pi_arr(arr, vertex=False, lep_pdg=11)
    Enu_QE_MeV = np.asarray(arr['Enu_QE'])[sel] * 1000.0
    Enu_t_GeV  = np.asarray(arr['Enu_true'])[sel]
    fSF        = np.asarray(arr['fScaleFactor'])[sel]

    pmns.SetPath(L_HK, 2.8)
    pmns.SetIsNuBar(False)
    pmns.SetMix(theta12, theta23, theta13, deltaCP + dCP_off_deg * np.pi / 180.0)
    pmns.SetDeltaMsqrs(dm21, dm32)
    prob = np.array([pmns.Prob(1, 0, float(E), L_HK) for E in Enu_t_GeV])

    w = fSF * prob / POSC_AVG * DSIGMA_DE_SCALE / BIN_WIDTH
    counts, _ = np.histogram(Enu_QE_MeV, bins=BINS, weights=w)
    return counts


BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/nuwro_25031_morestats/HK"
F_NOM   = f"{BASE}/HK_nue_FSI.flat.root"
F_PLUS  = f"{BASE}/piabs/HK_nue_piabs131_FSI.flat.root"   # piabs +31%
F_MINUS = f"{BASE}/piabs/HK_nue_piabs069_FSI.flat.root"   # piabs -31%

c_nom         = _spectrum_with_osc(F_NOM)
c_piabs_plus  = _spectrum_with_osc(F_PLUS)
c_piabs_minus = _spectrum_with_osc(F_MINUS)
c_dcp_plus    = _spectrum_with_osc(F_NOM, dCP_off_deg=+20)
c_dcp_minus   = _spectrum_with_osc(F_NOM, dCP_off_deg=-20)

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


cn = _draw(c_nom,         tol_dark,       "Nominal", nominal=True)
_draw(c_piabs_plus,       COL_PIABS_PLUS, r"$\pi_{\rm abs}$ +31\%", counts_nom=cn)
_draw(c_piabs_minus,      COL_PIABS_MINUS, r"$\pi_{\rm abs}$ -31\%", counts_nom=cn)
_draw(c_dcp_plus,         osc_inc_color,  r"$\delta_{CP}\,+20^{\circ}$", counts_nom=cn)
_draw(c_dcp_minus,        osc_dec_color,  r"$\delta_{CP}\,-20^{\circ}$", counts_nom=cn)

ax.legend(custom_lines, labels, loc='upper right', fontsize=11)
ax.set_xlim(0, 1200)
ax_ratio.set_xlim(0, 1200)
ax.set_ylim(bottom=0)
ax.set_ylabel(DSIGMA_DE_LABEL)
ax_ratio.set_xlabel(r"$E_\nu^{\rm QE}$ [MeV]")
ax_ratio.set_ylabel("variant / nominal")
ax_ratio.set_ylim(0.9, 1.1)

plt.savefig(outpath("Fig9_plots", "Fig9_HK_PiAbs_nue.pdf"))
plt.close(fig)
