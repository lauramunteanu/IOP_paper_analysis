from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

# Match the laura_extras BW Fig12 palette for the EDRMF / RPWIA distinction.
COL_EDRMF = "#444444"  # COL_GREY
COL_RPWIA = "#E69F00"  # COL_ORANGE

# NEUT EDRMF/RPWIA samples are generated on an O target; convert to the
# per-H2O-nucleon convention used by every other figure in the paper
# (NuWro/GENIE on water). Pure normalisation; ratio panel and any
# weighted-statistic downstream (BW, variation table) are invariant.
NEUT_O_PER_H2O = 16.0 / 18.0

# Per-mode bin spec for the HK EDRMF/RPWIA bias histogram. abs xlim (-900, 500)
# matches the wider HK abs range used by Fig 2 / Fig 3 / Fig 6.
BIN_SPECS = {
    "abs": dict(bin_width=20.0,  lo=-1000.0,           hi=1000.0,            xlim=(-900.0, 500.0)),
    "rel": dict(bin_width=0.02,  lo=REL_BIAS_XLIM[0],  hi=REL_BIAS_XLIM[1],  xlim=REL_BIAS_XLIM),
}


def plot_Enu_bias_numu(ax, ax_ratio, filename, label, nEvents, mode, nominal=False, counts_nom=None):
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  diff_sel = bias_arr(arr, "qe", kind=mode, vertex=False)
  fScaleFactor = float(np.max(arr['fScaleFactor']))

  spec = BIN_SPECS[mode]
  bin_width = spec["bin_width"]
  bins = np.arange(spec["lo"], spec["hi"] + bin_width, step=bin_width)
  weights = make_weights_dxsec_osc(arr, bin_width, "qe", filename,
                                    vertex=False, fScaleFactor=fScaleFactor) * NEUT_O_PER_H2O

  if label == "ED-RMF":
    color = COL_EDRMF
  elif label == "RPWIA":
    color = COL_RPWIA
  else:
    color = "black"

  ax.hist(
      diff_sel,
      bins=bins,
      histtype='step',
      weights=weights,
      color=color,
      linewidth=1.5,
      label=label
  )

  custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
  labels.append(label)

  counts, edges = np.histogram(diff_sel, weights=weights, bins=bins)

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


_events = -1

for mode in ("abs", "rel"):
    custom_lines, labels = [], []
    fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))

    counts_rpwia = plot_Enu_bias_numu(
        ax=ax, ax_ratio=ax_ratio,
        filename="/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/HK/NEUT_HK_RPWIA_numu.flat.root",
        label="RPWIA", nEvents=_events, mode=mode, nominal=True,
    )
    plot_Enu_bias_numu(
        ax=ax, ax_ratio=ax_ratio,
        filename="/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/HK/NEUT_HK_EDRMF_numu.flat.root",
        label="ED-RMF", nEvents=_events, mode=mode, nominal=False,
        counts_nom=counts_rpwia,
    )

    spec = BIN_SPECS[mode]
    ax.vlines(x=0, ymin=0, ymax=ax.get_ylim()[1], color='black', linestyles='--')
    ax.legend(custom_lines, labels, loc='upper left', fontsize=13)
    ax.set_xlim(*spec["xlim"])
    ax_ratio.set_xlim(*spec["xlim"])
    ax.set_ylabel(bias_ylabel(mode))

    ax_ratio.set_xlabel(bias_xlabel("qe", mode))
    ax_ratio.set_ylabel("ED-RMF/RPWIA")
    auto_ratio_ylim(ax_ratio, counts_rpwia)

    plt.savefig(outpath("Fig7_plots", f"Fig7_HK_EnuRecoFSIBias_EDRMF_numu_{mode}.pdf"))
    plt.close(fig)
