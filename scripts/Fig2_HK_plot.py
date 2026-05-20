from FlatTreeMod import *
from collections import defaultdict
ROOT.gROOT.SetBatch(True)

# Per-mode bin spec for the HK CC0pi bias histogram (with Mode breakdown).
# abs bin_width matches Fig 3 HK so the two figures share the same x-axis
# granularity.
BIN_SPECS = {
    "abs": dict(bin_width=20.0,  lo=-1000.0,           hi=1000.0,            xlim=(-900.0, 500.0)),
    "rel": dict(bin_width=0.02,  lo=REL_BIAS_XLIM[0],  hi=REL_BIAS_XLIM[1],  xlim=REL_BIAS_XLIM),
}


def plot_Enu_bias(filename, label, isNuBar, nEvents, plot_name, mode, vertex=False):
  fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  diff_sel = bias_arr(arr, "qe", kind=mode, vertex=vertex)

  # CCQE vs non-CCQE split on the same CC0pi selection as bias_arr uses.
  # CCQE is the dominant signal channel at HK (~80% of CC0pi); the
  # non-CCQE tail is mostly 2p2h and CC-RES events whose pion was absorbed
  # via FSI -- splitting by Mode (rather than by final-state neutron
  # content) makes that physics directly visible.
  sel = is_cc0pi_arr(arr, vertex=vertex)
  modes_all = np.abs(np.asarray(arr['Mode']))
  is_ccqe_all = (modes_all == 1)
  is_ccqe = is_ccqe_all[np.asarray(sel, dtype=bool)]
  bias_by_mode = {True: diff_sel[is_ccqe], False: diff_sel[~is_ccqe]}

  spec = BIN_SPECS[mode]
  bin_width = spec["bin_width"]
  bins = np.arange(spec["lo"], spec["hi"] + bin_width, step=bin_width)
  centers = 0.5 * (bins[:-1] + bins[1:])
  fScaleFactor = float(np.max(arr['fScaleFactor']))
  weights = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(diff_sel)
  counts_total, _ = np.histogram(diff_sel, bins=bins, weights=weights)

  ax.hist(diff_sel, bins=bins, histtype='step', weights=weights,
          color=tol_dark, linewidth=1.8)

  sub_counts = {}
  for is_qe, color, lbl in ((True,  tol_teal,    "CCQE"),
                            (False, tol_magenta, "non-CCQE")):
    vals = bias_by_mode[is_qe]
    if len(vals) == 0:
      sub_counts[is_qe] = np.zeros_like(counts_total)
      continue
    w = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(vals)
    ax.hist(vals, bins=bins, histtype='step', weights=w,
            color=color, linewidth=1.4)
    c, _ = np.histogram(vals, bins=bins, weights=w)
    sub_counts[is_qe] = c

  # Ratio panel: each subset / total. Strict division -- the two lines sum
  # to 1 in every bin with at least one event; bins with total=0 give NaN
  # and the step lines break there.
  for is_qe, color in ((True, tol_teal), (False, tol_magenta)):
    ratio = np.divide(sub_counts[is_qe], counts_total,
                      out=np.full_like(counts_total, np.nan, dtype=float),
                      where=counts_total > 0)
    ax_ratio.step(centers, ratio, where="mid", color=color, linewidth=1.4)
  ax_ratio.set_ylim(0, 1.05)
  ax_ratio.set_ylabel("fraction\nof total")

  ax.set_xlim(*spec["xlim"])
  ax.set_ylabel(bias_ylabel(mode))
  legend_handles = [
      Line2D([0], [0], color=tol_dark,    lw=1.8, label=r"Total CC0$\pi$"),
      Line2D([0], [0], color=tol_teal,    lw=1.4, label="CCQE"),
      Line2D([0], [0], color=tol_magenta, lw=1.4, label="non-CCQE"),
  ]
  ax.legend(handles=legend_handles, loc='upper left', fontsize=10)
  plt.setp(ax.get_xticklabels(), visible=False)

  ax_ratio.set_xlim(*spec["xlim"])
  ax_ratio.set_xlabel(bias_xlabel("qe", mode))

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
