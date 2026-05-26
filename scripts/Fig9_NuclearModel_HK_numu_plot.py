"""Fig 9 — HK FHC νμ disappearance, comparing NEUT EDRMF vs RPWIA CCQE
dσ/dE distributions.

  · X-axis: E_ν^QE [MeV] (oscillation-weighted reco-energy spectrum)
  · Y-axis: dσ/dE [10⁻⁴² cm²/nucleon/MeV]
  · Top panel:   EDRMF and RPWIA CCQE-only differential cross-section
  · Bottom panel: EDRMF / RPWIA ratio

CCQE-only filter: cc & |Mode|==1 & first-FS-particle PDG == 13.
Per-event weight: fScaleFactor × P(νμ→νμ) / ⟨P_osc⟩_Φ × DSIGMA_DE_SCALE / bin
then × 16/18 to convert NEUT's per-O-nucleon to a per-H2O-nucleon axis
(matches the NuWro nominal samples used elsewhere in the paper).

Style matches Fig 7: custom_lines/labels legend, step-post ratio panel,
auto_ratio_ylim.
"""
from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

# Match the laura_extras BW Fig12 / Fig 7 palette for EDRMF vs RPWIA.
COL_EDRMF = "#444444"   # COL_GREY
COL_RPWIA = "#E69F00"   # COL_ORANGE

# NEUT samples are generated on Oxygen; multiply by 16/18 to put them on
# the same "per H2O nucleon" axis as the NuWro samples used elsewhere.
NEUT_O_PER_H2O = 16.0 / 18.0

# Binning matches Fig 1 HK (0-2000 MeV, 20 MeV bins).
BIN_WIDTH = 20.0
BINS = np.arange(0, 2000, step=BIN_WIDTH)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])
EDGES = BINS


def _ccqe_dxsec(filename, *, channel):
    """Return dσ/dE [10⁻⁴² cm²/nucleon/MeV] for CCQE-only events from a
    NEUT νμ flat-tree, with the requested oscillation channel applied
    (``"surv"`` → P(νμ→νμ),  ``"app"`` → P(νμ→νe)). Includes the 16/18
    O→H2O nucleon correction."""
    arr = load_arrays(filename, max_events=-1 if False else None)

    cc      = np.asarray(arr['cc'], dtype=bool)
    is_ccqe = np.abs(np.asarray(arr['Mode'])) == 1
    # First-FS-particle convention (matches is_cc0pi_arr post-FSI branch).
    first_pdg = ak.to_numpy(ak.fill_none(ak.firsts(abs(arr['pdg'])), -1))
    sel = cc & is_ccqe & (first_pdg == 13)

    Enu_QE_MeV = np.asarray(arr['Enu_QE'])[sel] * 1000.0
    Enu_t_GeV  = np.asarray(arr['Enu_true'])[sel]
    fSF        = np.asarray(arr['fScaleFactor'])[sel]

    if channel == "surv":
        osc_w = osc_weights_mumu(Enu_t_GeV, filename=filename)
        # ⟨P_osc⟩_Φ for the appropriate channel: lep_pdg drives the
        # appearance-vs-survival lookup inside posc_flux_avg.
        avg = posc_flux_avg('HK', 'numu', lep_pdg=13)
    elif channel == "app":
        osc_w = osc_weights_mue(Enu_t_GeV, filename=filename)
        avg = posc_flux_avg('HK', 'numu', lep_pdg=11)
    else:
        raise ValueError(f"unknown channel {channel!r}")

    w = fSF * osc_w / avg * DSIGMA_DE_SCALE / BIN_WIDTH * NEUT_O_PER_H2O
    counts, _ = np.histogram(Enu_QE_MeV, bins=BINS, weights=w)
    return counts


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
F_EDRMF = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/HK/NEUT_HK_EDRMF_numu.flat.root"
F_RPWIA = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/HK/NEUT_HK_RPWIA_numu.flat.root"

c_RPWIA = _ccqe_dxsec(F_RPWIA, channel="surv")
c_EDRMF = _ccqe_dxsec(F_EDRMF, channel="surv")

custom_lines, labels = [], []
fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))


def _draw(counts, color, label, nominal=False, counts_nom=None):
    """Top-panel histogram + bottom-panel step.  Returns counts (for the
    nominal call) or None.  Mirrors the Fig 7 helper layout."""
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


# RPWIA first so it becomes the ratio reference.
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

plt.savefig(outpath("Fig9_plots", "Fig9_HK_NuclearModel_numu.pdf"))
plt.close(fig)
