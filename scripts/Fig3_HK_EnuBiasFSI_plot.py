from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

# Per-mode bin spec for the HK CC0pi bias histogram.
#   abs:  20 MeV bins over [-1000, +1000] MeV; visible window (-900, +300) MeV
#   rel:  0.005 bins over REL_BIAS_XLIM (= (-0.9, +0.3) dimensionless)
BIN_SPECS = {
    "abs": dict(bin_width=20.0,  lo=-1000.0,           hi=1000.0,            xlim=(-900.0, 300.0)),
    "rel": dict(bin_width=0.005, lo=REL_BIAS_XLIM[0],  hi=REL_BIAS_XLIM[1],  xlim=REL_BIAS_XLIM),
}


def plot_Enu_bias_numu(ax, ax_ratio, filename, isNub, nEvents, mode, nominal=False, counts_nom=None):
  # noFSI vs FSI no longer toggled via vertex flag — caller passes a noFSI
  # file when they want the noFSI line (nominal=True) and an FSI file for the
  # FSI overlay. Vertex stack of an FSI file would be biased by NuWro binding-
  # energy bookkeeping at cascade exit, so we read the post-FSI stack of the
  # appropriate file instead.
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  diff_sel = bias_arr(arr, "qe", kind=mode)
  fScaleFactor = float(np.max(arr['fScaleFactor']))
  Log(f"  pass {len(diff_sel)} / total {len(arr['Enu_true'])}  (mode={mode} file={filename})")

  spec = BIN_SPECS[mode]
  bin_width = spec["bin_width"]
  bins = np.arange(spec["lo"], spec["hi"] + bin_width, step=bin_width)
  weights = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(diff_sel)

  if nominal:
    color = dark_blue
    label = "noFSI"
  else:
    color = dark_red
    label = "FSI"

  ax.hist(
      diff_sel,
      bins=bins,
      histtype='step',
      weights=weights,
      color=color,
      linewidth=1.8,
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


def _draw_pair(fname_FSI, flav_tag, isNub, mode, n_events):
  """Render one (flavour, mode) combination — produces a single PDF."""
  global custom_lines, labels
  custom_lines, labels = [], []
  fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))

  fname_noFSI = noFSI_path(fname_FSI)
  counts_nom = plot_Enu_bias_numu(
      ax, ax_ratio, filename=fname_noFSI, isNub=isNub,
      nEvents=n_events, mode=mode, nominal=True,
  )
  counts_fsi = plot_Enu_bias_numu(
      ax, ax_ratio, filename=fname_FSI, isNub=isNub,
      nEvents=n_events, mode=mode, nominal=False, counts_nom=counts_nom,
  )

  spec = BIN_SPECS[mode]
  ax.axvline(0, color='black', linestyle='--', lw=0.7)
  ax.legend(custom_lines, labels, loc='upper right')
  ax.set_xlim(*spec["xlim"])
  peak = max(float(counts_nom.max()), float(counts_fsi.max()))
  ax.set_ylim(0, peak * 1.15)
  ax.set_ylabel(bias_ylabel(mode))

  ax_ratio.set_xlabel(bias_xlabel("qe", mode))
  ax_ratio.set_ylabel("FSI/noFSI")
  ax_ratio.set_xlim(*spec["xlim"])

  plt.savefig(outpath("Fig3_plots", f"Enu_bias_FSIvsNoFSI_{flav_tag}_{mode}.pdf"))
  plt.close(fig)


_events = -1
_FSI_FILES = {
    "numubar": "../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root",
    "numu":    "../../Remade_April26/nuwro_25031_morestats/HK/HK_numu_FSI.flat.root",
}

for mode in ("abs", "rel"):
    for flav_tag, fname_FSI in _FSI_FILES.items():
        _draw_pair(fname_FSI, flav_tag, isNub=(flav_tag == "numubar"), mode=mode, n_events=_events)
