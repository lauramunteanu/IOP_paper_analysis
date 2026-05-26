from FlatTreeMod import *

# Match the laura_extras BW Fig12 palette for the EDRMF / RPWIA distinction
# (and the BW summary "Nuc. Pot." colour key). Defined locally — see
# scripts/laura_extras/fsi_iop_plots.py for the originals.
COL_EDRMF = "#444444"  # COL_GREY
COL_RPWIA = "#E69F00"  # COL_ORANGE

# Per-mode bin spec for the DUNE EDRMF/RPWIA bias histogram. Both modes now
# use the standard units shared with Fig 4 DUNE (MeV for abs, dimensionless
# for rel) so the units / labels come from bias_xlabel / bias_ylabel.
BIN_SPECS = {
    "abs": dict(bin_width=20.0,  lo=-1000.0,           hi=1000.0,            xlim=(-900.0, 300.0)),
    "rel": dict(bin_width=0.02,  lo=REL_BIAS_XLIM[0],  hi=REL_BIAS_XLIM[1],  xlim=REL_BIAS_XLIM),
}


def plot_Enu_bias_numu(ax, ax_ratio, filename, label, nEvents, withPion, mode,
                       nominal=False, counts_nom=None):
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  observable = "had" if withPion else "avail"
  bias = bias_arr(arr, observable, kind=mode, vertex=False)
  fScaleFactor = float(np.max(arr['fScaleFactor']))

  spec = BIN_SPECS[mode]
  bin_width = spec["bin_width"]
  bins = np.arange(spec["lo"], spec["hi"] + bin_width, step=bin_width)
  weights = make_weights_dxsec_osc(arr, bin_width, observable, filename,
                                    vertex=False, fScaleFactor=fScaleFactor)

  if label == "ED-RMF":
      color = COL_EDRMF
  elif label == "RPWIA":
      color = COL_RPWIA
  else:
      color = "black"

  # The with/without-pion variant is in the output filename; the legend
  # carries only the EDRMF / RPWIA distinction (matches Fig 4).
  ax.hist(
      bias,
      bins=bins,
      histtype='step',
      weights=weights,
      color=color,
      linewidth=1.5,
      label=label,
  )

  custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
  labels.append(label)

  counts, edges = np.histogram(bias, weights=weights, bins=bins)

  if nominal:
      ax_ratio.hlines(1, bins[0], bins[-1], linestyle='--', color='black')
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
        ax.legend(custom_lines, labels, loc='upper left', fontsize=13)
        ax.set_ylabel(bias_ylabel(mode))

        ax_ratio.set_xlabel(bias_xlabel(cfg["xlabel_observable"], mode))
        ax_ratio.set_ylabel("ED-RMF/RPWIA")
        ax.set_xlim(*spec["xlim"])
        ax_ratio.set_xlim(*spec["xlim"])
        auto_ratio_ylim(ax_ratio, counts_rpwia)

        plt.savefig(outpath("Fig7_plots", f"{cfg['stem']}_{mode}.pdf"))
        plt.close(fig)
