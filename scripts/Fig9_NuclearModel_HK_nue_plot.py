"""Fig 9 — HK FHC νμ→νe appearance, comparing NEUT EDRMF vs RPWIA CCQE
dσ/dE distributions.

Same conventions as Fig9_NuclearModel_HK_numu_plot.py; the only difference
is the osc factor (appearance instead of survival).  Events still have a
final-state muon (NEUT EDRMF/RPWIA files are νμ-CC) — we proxy through
them and reweight by P(νμ→νe).  ⟨P_osc⟩_Φ uses the appearance flux average.

  · X-axis: E_ν^QE [MeV]
  · Y-axis: dσ/dE [10⁻⁴² cm²/nucleon/MeV]
  · Top:    EDRMF and RPWIA CCQE
  · Bottom: EDRMF / RPWIA ratio
"""
from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

COL_EDRMF = "#444444"
COL_RPWIA = "#E69F00"
NEUT_O_PER_H2O = 16.0 / 18.0

BIN_WIDTH = 20.0
BINS = np.arange(0, 2000, step=BIN_WIDTH)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])
EDGES = BINS


def _ccqe_dxsec_appearance(filename):
    """Same CCQE-only filter as the νμ panel, but reweight per-event with
    P(νμ→νe) and divide by the appearance ⟨P_osc⟩_Φ."""
    arr = load_arrays(filename, max_events=None)

    cc      = np.asarray(arr['cc'], dtype=bool)
    is_ccqe = np.abs(np.asarray(arr['Mode'])) == 1
    first_pdg = ak.to_numpy(ak.fill_none(ak.firsts(abs(arr['pdg'])), -1))
    sel = cc & is_ccqe & (first_pdg == 13)

    Enu_QE_MeV = np.asarray(arr['Enu_QE'])[sel] * 1000.0
    Enu_t_GeV  = np.asarray(arr['Enu_true'])[sel]
    fSF        = np.asarray(arr['fScaleFactor'])[sel]

    osc_w = osc_weights_mue(Enu_t_GeV, filename=filename)
    avg = posc_flux_avg('HK', 'numu', lep_pdg=11)

    w = fSF * osc_w / avg * DSIGMA_DE_SCALE / BIN_WIDTH * NEUT_O_PER_H2O
    counts, _ = np.histogram(Enu_QE_MeV, bins=BINS, weights=w)
    return counts


F_EDRMF = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/HK/NEUT_HK_EDRMF_numu.flat.root"
F_RPWIA = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/HK/NEUT_HK_RPWIA_numu.flat.root"

c_RPWIA = _ccqe_dxsec_appearance(F_RPWIA)
c_EDRMF = _ccqe_dxsec_appearance(F_EDRMF)

custom_lines, labels = [], []
fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))


def _draw(counts, color, label, nominal=False, counts_nom=None):
    ax.hist(CENTERS, bins=BINS, histtype='step', weights=counts,
            color=color, linewidth=1.5, label=label)
    custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
    labels.append(label)
    if nominal:
        ax_ratio.hlines(1, BINS[0], BINS[-1], linestyle='--', color='black')
        return counts
    ratio = counts / np.where(counts_nom > 0, counts_nom, np.nan)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ax_ratio.step(EDGES, np.r_[ratio, ratio[-1]],
                  color=color, linestyle='-', where='post')


counts_rpwia = _draw(c_RPWIA, COL_RPWIA, "RPWIA", nominal=True)
_draw(c_EDRMF, COL_EDRMF, "ED-RMF", nominal=False, counts_nom=counts_rpwia)

ax.legend(custom_lines, labels, loc='upper right', fontsize=10)
ax.set_xlim(0, 1200)
ax_ratio.set_xlim(0, 1200)
ax.set_ylim(bottom=0)
ax.set_ylabel(DSIGMA_DE_LABEL)
ax_ratio.set_xlabel(r"$E_\nu^{\rm QE}$ [MeV]")
ax_ratio.set_ylabel("ED-RMF / RPWIA")
auto_ratio_ylim(ax_ratio, counts_rpwia)

plt.savefig(outpath("Fig9_plots", "Fig9_HK_NuclearModel_nue.pdf"))
plt.close(fig)
