from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

# Match the laura_extras BW palette for the FSI / no-FSI distinction so this
# figure and the BW summary plots share a colour key.
COL_FSI   = "#444444"  # COL_GREY — with FSI
COL_NOFSI = "#D55E00"  # COL_VERMILION — no FSI

# Per-mode bin spec for the DUNE FSI-vs-noFSI bias histogram.
BIN_SPECS = {
    "abs": dict(bin_width=16.0,  lo=-1000.0,           hi=1000.0,            xlim=(-900.0, 300.0)),
    "rel": dict(bin_width=0.02,  lo=REL_BIAS_XLIM[0],  hi=REL_BIAS_XLIM[1],  xlim=REL_BIAS_XLIM),
}


def plot_Enu_bias_numu(ax, ax_ratio, filename, nEvents, withPion, mode, nominal=False, counts_nom=None):
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  observable = "had" if withPion else "avail"
  bias = bias_arr(arr, observable, kind=mode, vertex=False)
  fScaleFactor = float(np.max(arr['fScaleFactor']))

  spec = BIN_SPECS[mode]
  bin_width = spec["bin_width"]
  bins = np.arange(spec["lo"], spec["hi"] + bin_width, step=bin_width)

  weights = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(bias)

  # Label intentionally drops the pion-mass modifier: the with/without-pion
  # variant is encoded in the output filename, and the user-facing legend
  # only needs the FSI / no-FSI distinction.
  if nominal:
    color = COL_NOFSI
    label = "no FSI"
  else:
    color = COL_FSI
    label = "FSI"

  ax.hist(
      bias,
      bins=bins,
      histtype='step',
      weights=weights,
      color=color,
      linewidth=1.5,
      label=label
  )

  custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
  labels.append(label)

  counts, edges = np.histogram(bias, weights=weights, bins=bins)

  if nominal:
      ax_ratio.hlines(1, bins[0], bins[-1], linestyle='--', color='black')
      Print(f"Done: {filename}")
      return counts
  else:
      ratio = counts / counts_nom
      ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
      ax_ratio.step(
        edges,
        np.r_[ratio, ratio[-1]],
        color=color,
        linestyle='-',
        where='post'
        )

  Print(f"Done: {filename}")
  return counts


_events = 10000

_PLOT_CONFIGS = [
    {"withPion": True,  "flavor": "numu",
     "file": "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root",
     "title": r"$\nu_{\mu}$, w/ pion mass",
     "stem": "Fig4_DUNE_EnuRecoFSIBias_WithPion_numu"},
    {"withPion": True,  "flavor": "numubar",
     "file": "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root",
     "title": r"$\bar{\nu}_{\mu}$, w/ pion mass",
     "stem": "Fig4_DUNE_EnuRecoFSIBias_WithPion_numubar"},
    {"withPion": False, "flavor": "numu",
     "file": "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root",
     "title": r"$\nu_{\mu}$, w/o pion mass",
     "stem": "Fig4_DUNE_EnuRecoFSIBias_WithoutPion_numu"},
    {"withPion": False, "flavor": "numubar",
     "file": "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root",
     "title": r"$\bar{\nu}_{\mu}$, w/o pion mass",
     "stem": "Fig4_DUNE_EnuRecoFSIBias_WithoutPion_numubar"},
]

for mode in ("abs", "rel"):
    for cfg in _PLOT_CONFIGS:
        custom_lines, labels = [], []
        fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))
        counts_nom = plot_Enu_bias_numu(
            ax=ax, ax_ratio=ax_ratio, filename=noFSI_path(cfg["file"]),
            nEvents=_events, withPion=cfg["withPion"], mode=mode,
            nominal=True,
        )
        counts_fsi = plot_Enu_bias_numu(
            ax=ax, ax_ratio=ax_ratio, filename=cfg["file"],
            nEvents=_events, withPion=cfg["withPion"], mode=mode,
            nominal=False, counts_nom=counts_nom,
        )

        spec = BIN_SPECS[mode]
        ax.legend(custom_lines, labels, loc='upper left', fontsize=13)
        ax.set_xlim(*spec["xlim"])
        peak = max(float(counts_nom.max()), float(counts_fsi.max()))
        ax.set_ylim(0, peak * 1.15)
        ax.set_ylabel(bias_ylabel(mode))

        observable = "had" if cfg["withPion"] else "avail"
        ax_ratio.set_xlabel(bias_xlabel(observable, mode))
        ax_ratio.set_ylabel("FSI/noFSI")
        ax_ratio.set_xlim(*spec["xlim"])
        auto_ratio_ylim(ax_ratio, counts_nom)

        plt.savefig(outpath("Fig4_plots", f"{cfg['stem']}_{mode}.pdf"))
        plt.close(fig)
