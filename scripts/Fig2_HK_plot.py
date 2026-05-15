from FlatTreeMod import *
from collections import defaultdict
ROOT.gROOT.SetBatch(True)

# Per-mode bin spec for the HK CC0pi bias histogram (with Mode breakdown).
BIN_SPECS = {
    "abs": dict(bin_width=10.0,  lo=-1000.0,           hi=1000.0,            xlim=(-900.0, 300.0)),
    "rel": dict(bin_width=0.005, lo=REL_BIAS_XLIM[0],  hi=REL_BIAS_XLIM[1],  xlim=REL_BIAS_XLIM),
}


def plot_Enu_bias(filename, label, isNuBar, nEvents, plot_name, mode, vertex=False):
  fig, ax = make_fig('single')
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  diff_sel = bias_arr(arr, "qe", kind=mode, vertex=vertex)
  # Need the Mode array on the same CC0pi selection as diff_sel.
  sel = is_cc0pi_arr(arr, vertex=vertex)
  modes = np.asarray(arr['Mode'])
  modes_sel = modes[sel]
  diff_by_mode = {int(m): diff_sel[modes_sel == m] for m in np.unique(modes_sel)}

  spec = BIN_SPECS[mode]
  bin_width = spec["bin_width"]
  bins = np.arange(spec["lo"], spec["hi"] + bin_width, step=bin_width)
  fScaleFactor = float(np.max(arr['fScaleFactor']))
  weights = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(diff_sel)

  ax.hist(diff_sel, bins=bins, histtype='step', weights=weights,
          color=dark_blue, linewidth=1.5, label=label)

  if not isNuBar:
    mode_colors = {1: tol_red, 2: tol_blue}
  else:
    mode_colors = {-1: tol_red, -2: tol_blue}

  for m, diffs in diff_by_mode.items():
    if abs(m) not in (1, 2) or len(diffs) == 0 or m == 0:
      continue
    w = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(diffs)
    ax.hist(diffs, bins=bins, histtype='step', weights=w,
            color=mode_colors.get(m, "gray"), linewidth=1.4,
            linestyle="--", label=f"Mode {m}")

  ax.set_xlim(*spec["xlim"])
  ax.set_xlabel(bias_xlabel("qe", mode))
  ax.set_ylabel(bias_ylabel(mode))
  ax.legend(loc='best')
  plt.savefig(outpath("Fig2_plots", f"Fig2_HK_EnuRecoBias_{plot_name}_{mode}.pdf"))
  plt.close(fig)


_events = -1
# noFSI line now from the dedicated noFSI file with vertex=False (the FSI
# vertex stack is biased by NuWro binding-energy bookkeeping at cascade exit;
# loading the noFSI sample sidesteps that).
_CONFIGS = [
    dict(filename=noFSI_path("../../Remade_April26/nuwro_25031_morestats/HK/HK_numu_FSI.flat.root"),
         label=r"no FSI $\nu_{\mu}$", isNuBar=False, plot_name="noFSI_numu"),
    dict(filename=noFSI_path("../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root"),
         label=r"no FSI $\bar{\nu}_{\mu}$", isNuBar=True, plot_name="noFSI_numubar"),
]

for mode in ("abs", "rel"):
    for cfg in _CONFIGS:
        plot_Enu_bias(nEvents=_events, mode=mode, vertex=False, **cfg)
