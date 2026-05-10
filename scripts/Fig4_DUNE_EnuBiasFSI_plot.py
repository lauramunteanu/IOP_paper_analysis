from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

def plot_Enu_bias_numu(ax, ax_ratio, filename, nEvents, withPion, nominal=False, counts_nom=None):
  # noFSI vs FSI line is now selected by the caller passing a noFSI or FSI
  # file. We always read the post-FSI stack (vertex=False) — vertex stack of
  # an FSI file would be biased by NuWro binding-energy bookkeeping. For a
  # noFSI file no cascade ran, so post-FSI == no-FSI. All energies in MeV.
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  bias_wo_GeV, bias_with_GeV, _ = enu_had_arr(arr, vertex=False)
  bias_wo_list   = bias_wo_GeV   * 1000.0
  bias_with_list = bias_with_GeV * 1000.0
  fScaleFactor = float(np.max(arr['fScaleFactor']))

  bin_width = 16.0  # MeV
  bins = np.arange(-1000, 1000 + bin_width, step=bin_width)

  weights_with = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(bias_with_list)
  weights_wo   = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(bias_wo_list)

  # ---------------------------------
  # Choose which definition to plot
  # ---------------------------------
  if(withPion == True):
    bias = bias_with_list
    weights = weights_with
    pion_label = "w/ pion mass"
  else:
    bias = bias_wo_list
    weights = weights_wo
    pion_label = "w/o pion mass"

  # ---------------------------------
  # Choose FSI / noFSI (label & colour driven by `nominal`, not `vertex`)
  # ---------------------------------
  if nominal:
    color = dark_blue
    label = pion_label + " noFSI"
  else:
    color = dark_red
    label = pion_label + " FSI"

  # ---------------------------------
  # Main histogram
  # ---------------------------------
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

  # ---------------------------------
  # Ratio histogram
  # ---------------------------------
  counts, edges = np.histogram(bias, weights=weights, bins=bins)

  if nominal == True:
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

plot_configs = [
    {"withPion": True,  "flavor": "numu",
     "file": "../../Remade_April26/nuwro_25031/DUNE/DUNE_numu_FSI.flat.root",
     "title": r"$\nu_{\mu}$, w/ pion mass",
     "outfile": "Fig4_plots/Fig4_DUNE_EnuRecoFSIBias_WithPion_numu.pdf"},
    {"withPion": True,  "flavor": "numubar",
     "file": "../../Remade_April26/nuwro_25031/DUNE/DUNE_numub_FSI.flat.root",
     "title": r"$\bar{\nu}_{\mu}$, w/ pion mass",
     "outfile": "Fig4_plots/Fig4_DUNE_EnuRecoFSIBias_WithPion_numubar.pdf"},
    {"withPion": False, "flavor": "numu",
     "file": "../../Remade_April26/nuwro_25031/DUNE/DUNE_numu_FSI.flat.root",
     "title": r"$\nu_{\mu}$, w/o pion mass",
     "outfile": "Fig4_plots/Fig4_DUNE_EnuRecoFSIBias_WithoutPion_numu.pdf"},
    {"withPion": False, "flavor": "numubar",
     "file": "../../Remade_April26/nuwro_25031/DUNE/DUNE_numub_FSI.flat.root",
     "title": r"$\bar{\nu}_{\mu}$, w/o pion mass",
     "outfile": "Fig4_plots/Fig4_DUNE_EnuRecoFSIBias_WithoutPion_numubar.pdf"},
]

for cfg in plot_configs:
    custom_lines, labels = [], []
    fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))
    # noFSI line from the dedicated noFSI sample (sibling of cfg["file"]).
    counts_nom = plot_Enu_bias_numu(
        ax=ax, ax_ratio=ax_ratio, filename=noFSI_path(cfg["file"]),
        nEvents=_events, withPion=cfg["withPion"],
        nominal=True,
    )
    counts_fsi = plot_Enu_bias_numu(
        ax=ax, ax_ratio=ax_ratio, filename=cfg["file"],
        nEvents=_events, withPion=cfg["withPion"],
        nominal=False, counts_nom=counts_nom,
    )

    ax.legend(custom_lines, labels, loc='best')
    ax.set_title(cfg["title"])
    ax.set_xlim(-1000, 1000)
    peak = max(float(counts_nom.max()), float(counts_fsi.max()))
    ax.set_ylim(0, peak * 1.15)
    ax.set_ylabel(DSIGMA_DE_LABEL)

    ax_ratio.set_xlabel(r"$E_{\nu}^{\rm reco} - E_{\nu}^{\rm true}$ [MeV]")
    ax_ratio.set_ylabel("FSI/noFSI")
    ax_ratio.set_xlim(-1000, 1000)
    # y-range left to matplotlib auto-scale

    plt.savefig(cfg["outfile"])
    plt.close(fig)
