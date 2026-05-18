from FlatTreeMod import *
from matplotlib.colors import LogNorm
ROOT.gROOT.SetBatch(True)

# Shared z-scale across every Fig6 panel — the per-panel colorbar is dropped
# and the standalone colorbar PDF (make_Fig6_colorbar.py) becomes the
# common z-axis in the LaTeX subfigure environment.
Z_CMAP = "RdPu"
Z_VMAX = 0.3
Z_VMIN = Z_VMAX * 1e-4

# Per-mode y-axis bin specs. x-axis (Enu_true) stays in MeV in both modes.
YBIN_SPECS = {
    "abs": dict(bin_width=20.0,  lo=-1000.0,           hi=1000.0,            ylim=(-900.0, 300.0)),
    "rel": dict(bin_width=0.005, lo=REL_BIAS_XLIM[0],  hi=REL_BIAS_XLIM[1],  ylim=(-0.5, 0.3)),
}


def plot_Enu_bias_numu(filename, nEvents, plot_name, withPion, mode, xbins):
  fig, ax = make_fig('single')
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  observable = "had" if withPion else "avail"
  yvals = bias_arr(arr, observable, kind=mode, vertex=False)
  # Enu_true on the same CC selection.
  _, _, cc_mask = enu_had_arr(arr, vertex=False)
  Enu_t = np.asarray(arr['Enu_true'])[np.asarray(cc_mask, dtype=bool)] * 1000.0  # MeV

  yspec = YBIN_SPECS[mode]
  ybins = np.arange(yspec["lo"], yspec["hi"] + yspec["bin_width"], step=yspec["bin_width"])

  H, xedges, yedges = np.histogram2d(
      Enu_t,
      yvals,
      bins=[xbins, ybins],
  )

  # Normalise so each Enu_true slice sums to 1.
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

  # Overlay per-Enu_true slice median / mean / 16-84% band.
  xcenters = 0.5 * (xedges[:-1] + xedges[1:])
  med, mean, q16, q84 = [], [], [], []
  for ix in range(len(xcenters)):
      in_slice = (Enu_t >= xedges[ix]) & (Enu_t < xedges[ix + 1])
      if in_slice.sum() < 30:
          med.append(np.nan); mean.append(np.nan); q16.append(np.nan); q84.append(np.nan); continue
      s = yvals[in_slice]
      med.append(np.median(s))
      mean.append(np.mean(s))
      q16.append(np.percentile(s, 16))
      q84.append(np.percentile(s, 84))
  ax.fill_between(xcenters, q16, q84, color="#F5DEB3", alpha=0.30)
  ax.plot(xcenters, med,  color="#F5DEB3", lw=1.8)
  ax.plot(xcenters, mean, color="#E69F00", lw=1.7, ls="--")

  ax.set_ylabel(bias_xlabel(observable, mode))
  ax.set_xlabel(r"$E_\nu^{\rm true}$ [MeV]")
  ax.set_xlim(300, 6000)
  ax.set_ylim(*yspec["ylim"])
  plt.savefig(outpath("Fig6_plots", f"Fig6_DUNE_EnuRecoBias2D_{plot_name}_{mode}.pdf"))
  plt.close(fig)


_events = -1
_xbins = np.arange(300, 6000 + 120, 120)   # Enu_true bins, DUNE 300 MeV - 6 GeV, 120 MeV/bin

for mode in ("abs", "rel"):
    plot_Enu_bias_numu(filename="../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root",  nEvents=_events, plot_name="FSI_WithoutPion_numu",    withPion=False, mode=mode, xbins=_xbins)
    plot_Enu_bias_numu(filename="../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root", nEvents=_events, plot_name="FSI_WithoutPion_numubar", withPion=False, mode=mode, xbins=_xbins)
    plot_Enu_bias_numu(filename="../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root",  nEvents=_events, plot_name="FSI_WithPion_numu",       withPion=True,  mode=mode, xbins=_xbins)
    plot_Enu_bias_numu(filename="../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root", nEvents=_events, plot_name="FSI_WithPion_numubar",    withPion=True,  mode=mode, xbins=_xbins)
