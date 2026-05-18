from FlatTreeMod import *
from matplotlib.colors import LogNorm
ROOT.gROOT.SetBatch(True)

# Shared z-scale across every Fig6 panel — see Fig6_DUNE script header.
Z_CMAP = "RdPu"
Z_VMAX = 0.3
Z_VMIN = Z_VMAX * 1e-4

# Per-mode y-axis bin specs for the 2-D heatmap. x-axis (Enu_true) stays in MeV
# in both modes.
YBIN_SPECS = {
    "abs": dict(bin_width=16.0,  lo=-1000.0,           hi=1000.0,            ylim=(-900.0, 300.0)),
    "rel": dict(bin_width=0.005, lo=-1.0,              hi=1.0,               ylim=(-1.0, 1.0)),
}


def plot_Enu_bias(filename, isNub, nEvents, plot_name, mode, xbins):
  fig, ax = make_fig('single')
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  diff_sel = bias_arr(arr, "qe", kind=mode, vertex=False)
  # Enu_true on the same CC0pi selection (always MeV — this is the x-axis).
  flag = is_cc0pi_arr(arr, vertex=False)
  Enu_t_sel = np.asarray(arr['Enu_true'])[flag] * 1000.0

  yspec = YBIN_SPECS[mode]
  ybins = np.arange(yspec["lo"], yspec["hi"] + yspec["bin_width"], step=yspec["bin_width"])

  H, xedges, yedges = np.histogram2d(
      Enu_t_sel,
      diff_sel,
      bins=[xbins, ybins],
  )

  # Normalise so each Enu_true slice sums to 1 (per-row probability).
  denom = H.sum(axis=1, keepdims=True)
  H_plot = np.divide(H, denom, out=np.zeros_like(H), where=denom != 0)

  import matplotlib as _mpl
  cmap = _mpl.colormaps[Z_CMAP].copy()
  cmap.set_bad(cmap(0.0))
  H_plot_for_log = np.where(H_plot > 0, H_plot, np.nan)
  mesh = ax.pcolormesh(
      xedges,
      yedges,
      H_plot_for_log.T,
      cmap=cmap,
      shading="auto",
      norm=LogNorm(vmin=Z_VMIN, vmax=Z_VMAX),
  )

  # Overlay per-Enu_true slice median / mean / 16-84% band so the energy
  # dependence of the bias is visible as curves rather than buried in the heatmap.
  xcenters = 0.5 * (xedges[:-1] + xedges[1:])
  med, mean, q16, q84 = [], [], [], []
  for ix in range(len(xcenters)):
      in_slice = (Enu_t_sel >= xedges[ix]) & (Enu_t_sel < xedges[ix + 1])
      if in_slice.sum() < 30:
          med.append(np.nan); mean.append(np.nan); q16.append(np.nan); q84.append(np.nan); continue
      s = diff_sel[in_slice]
      med.append(np.median(s))
      mean.append(np.mean(s))
      q16.append(np.percentile(s, 16))
      q84.append(np.percentile(s, 84))
  ax.fill_between(xcenters, q16, q84, color="#F5DEB3", alpha=0.30)
  ax.plot(xcenters, med,  color="#F5DEB3", lw=1.8)
  ax.plot(xcenters, mean, color="#E69F00", lw=1.7, ls="--")

  # y-axis label tracks mode; x-axis is Enu_true [MeV] in both.
  ax.set_ylabel(bias_xlabel("qe", mode))
  ax.set_xlabel(r"$E_\nu^{\rm true}$ [MeV]")
  ax.set_xlim(150, 2000)
  ax.set_ylim(*yspec["ylim"])

  plt.savefig(outpath("Fig6_plots", f"Fig6_HK_EnuRecoBias2D_{plot_name}_{mode}.pdf"))
  plt.close(fig)


_events = -1
_xbins = np.arange(150, 2000 + 60, 60)   # Enu_true bins, HK 150 MeV - 2 GeV, 60 MeV/bin

for mode in ("abs", "rel"):
    plot_Enu_bias(filename="../../Remade_April26/nuwro_25031_morestats/HK/HK_numu_FSI.flat.root",    isNub=True,  nEvents=_events, plot_name="FSI_numub", mode=mode, xbins=_xbins)
    plot_Enu_bias(filename="../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root", isNub=False, nEvents=_events, plot_name="FSI_numu",  mode=mode, xbins=_xbins)
