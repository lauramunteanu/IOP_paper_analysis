from FlatTreeMod import *

# Per-mode bin spec for the DUNE EDRMF/RPWIA bias histogram. abs mode keeps
# Jake's original GeV-scaled binning (bias is computed in GeV here, not MeV),
# rel mode uses dimensionless REL_BIAS_XLIM.
BIN_SPECS = {
    "abs": dict(bin_width=0.05,  lo=-0.9,              hi=0.3,               xlim=(-0.9, 0.3)),
    "rel": dict(bin_width=0.005, lo=REL_BIAS_XLIM[0],  hi=REL_BIAS_XLIM[1],  xlim=REL_BIAS_XLIM),
}


def plot_Enu_bias_numu(ax, ax_ratio, filename, label, nEvents, withPion, mode,
                       nominal=False, counts_nom=None):
  """abs mode: bias in GeV (preserving Jake's original DUNE Fig7 y-scale).
  rel mode: dimensionless. The y-axis label adjusts via bias_ylabel(mode);
  for abs the GeV-scaled label is set explicitly below to match the prior
  Fig7 convention."""
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  observable = "had" if withPion else "avail"
  # abs mode is in GeV here (matches Jake's DUNE Fig7 plot); bias_arr abs
  # returns MeV, so divide by 1000 for the abs path. rel mode is dimensionless
  # and passes through unchanged.
  bias_MeV_or_rel = bias_arr(arr, observable, kind=mode, vertex=False)
  bias = bias_MeV_or_rel / 1000.0 if mode == "abs" else bias_MeV_or_rel
  fScaleFactor = float(np.max(arr['fScaleFactor']))

  spec = BIN_SPECS[mode]
  bin_width = spec["bin_width"]
  bins = np.arange(spec["lo"], spec["hi"] + bin_width, step=bin_width)
  weights = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(bias)

  if label == "ED-RMF":
      color = dark_red
  elif label == "RPWIA":
      color = dark_blue
  else:
      color = "black"

  pion_label = "w/ pion mass" if withPion else "w/o pion mass"
  plot_label = f"{label} {pion_label}"

  ax.hist(
      bias,
      bins=bins,
      histtype='step',
      weights=weights,
      color=color,
      linewidth=1.5,
      label=plot_label
  )

  custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
  labels.append(plot_label)

  counts, edges = np.histogram(bias, weights=weights, bins=bins)

  if nominal:
      ax_ratio.hlines(1, bins[0], bins[-1], linestyle='--', color=dark_blue)
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

  return counts


_events = 100000

_PLOT_CONFIGS = [
    dict(withPion=False, stem="Fig7_Ar40_EnuRecoBias_EDRMF_RPWIA_WithoutPion_ratio",
         xlabel_observable="avail"),
    dict(withPion=True,  stem="Fig7_Ar40_EnuRecoBias_EDRMF_RPWIA_WithPion_ratio",
         xlabel_observable="had"),
]

# abs-mode y-axis label preserves Jake's original GeV-scaled convention; rel
# is dimensionless, use the standard bias_ylabel(rel) helper.
_YLAB_ABS = (r"$\mathrm{d}\sigma/\mathrm{d}E$ "
             r"[10$^{-42}$ cm$^{2}$/nucleon/GeV]")

for mode in ("abs", "rel"):
    for cfg in _PLOT_CONFIGS:
        custom_lines, labels = [], []
        fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))

        counts_rpwia = plot_Enu_bias_numu(
            ax=ax, ax_ratio=ax_ratio,
            filename="/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/DUNE/DUNE_RPWIA_numu.root",
            label="RPWIA", nEvents=_events, withPion=cfg["withPion"], mode=mode,
            nominal=True,
        )
        plot_Enu_bias_numu(
            ax=ax, ax_ratio=ax_ratio,
            filename="/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/DUNE/DUNE_EDRMF_numu.root",
            label="ED-RMF", nEvents=_events, withPion=cfg["withPion"], mode=mode,
            nominal=False, counts_nom=counts_rpwia,
        )

        spec = BIN_SPECS[mode]
        ax.legend(custom_lines, labels, loc='best')
        ax.set_ylabel(_YLAB_ABS if mode == "abs" else bias_ylabel("rel"))

        # abs xlabel keeps GeV units; rel uses the standard dimensionless label.
        if mode == "abs":
            sym = r"E_{\nu}^{\rm had}" if cfg["withPion"] else r"E_{\nu}^{\rm avail}"
            ax_ratio.set_xlabel(rf"${sym} - E_{{\nu}}^{{\rm true}}$ [GeV]")
        else:
            ax_ratio.set_xlabel(bias_xlabel(cfg["xlabel_observable"], mode))
        ax_ratio.set_ylabel("ED-RMF/RPWIA")
        ax.set_xlim(*spec["xlim"])
        ax_ratio.set_xlim(*spec["xlim"])
        ax_ratio.set_ylim(0.5, 1.5)

        plt.savefig(outpath("Fig7_plots", f"{cfg['stem']}_{mode}.pdf"))
        plt.close(fig)
