from FlatTreeMod import *
from collections import defaultdict
ROOT.gROOT.SetBatch(True)


def plot_Enu_bias(filename, label, isNuBar, nEvents, plot_name, vertex=False):
  fig, ax = make_fig('single')
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  sel = is_cc0pi_arr(arr, vertex=vertex)
  diff_all = (np.asarray(arr['Enu_QE']) - np.asarray(arr['Enu_true'])) * 1000.0
  modes = np.asarray(arr['Mode'])
  diff_sel = diff_all[sel]
  modes_sel = modes[sel]
  diff_by_mode = {int(m): diff_sel[modes_sel == m] for m in np.unique(modes_sel)}

  bin_width = 10.0
  bins = np.arange(-1000, 1000, step=bin_width)
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

  ax.set_xlim(-1000, 1000)
  ax.set_xlabel(r"$E_{\nu}^{\rm QE} - E_{\nu}^{\rm true}$ [MeV]")
  ax.set_ylabel(DSIGMA_DE_LABEL)
  ax.legend(loc='best')
  plt.savefig(f"Fig2_plots/Fig2_HK_EnuRecoBias_{plot_name}.pdf")


_events = -1
# noFSI line now from the dedicated noFSI file with vertex=False (the FSI
# vertex stack is biased by NuWro binding-energy bookkeeping at cascade exit;
# loading the noFSI sample sidesteps that).
plot_Enu_bias(filename=noFSI_path("../../Remade_April26/HK/HK_numu_FSI.flat.root"),
              label=r"no FSI $\nu_{\mu}$", isNuBar=False, nEvents=_events,
              plot_name="noFSI_numu", vertex=False)
plot_Enu_bias(filename=noFSI_path("../../Remade_April26/HK/HK_numubar_FSI.flat.root"),
              label=r"no FSI $\bar{\nu}_{\mu}$", isNuBar=True, nEvents=_events,
              plot_name="noFSI_numubar", vertex=False)
