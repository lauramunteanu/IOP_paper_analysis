from FlatTreeMod import *
from matplotlib.colors import LogNorm
ROOT.gROOT.SetBatch(True)

# Shared z-scale across every Fig6 panel — see Fig6_DUNE script header.
Z_CMAP = "RdPu"
Z_VMAX = 0.3
Z_VMIN = Z_VMAX * 1e-4

def plot_Enu_bias(filename, isNub, nEvents, plot_name, xbins, ybins):
  fig, ax = make_fig('single')
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  flag = is_cc0pi_arr(arr, vertex=False)
  Enu_t_sel  = np.asarray(arr['Enu_true'])[flag] * 1000.0
  Enu_QE_sel = np.asarray(arr['Enu_QE'])  [flag] * 1000.0
  diff_sel = Enu_QE_sel - Enu_t_sel

  H, xedges, yedges = np.histogram2d(
      Enu_t_sel,
      diff_sel,
      bins=[ybins, xbins]
  )

  norm_mode = "y"

  if norm_mode == "x":
      # each enu bias bin sums to 1
      denom = H.sum(axis=0, keepdims=True)
      cbar_label = "Fraction per bias bin"

  elif norm_mode == "y":
      # each enu true bin sums to 1
      denom = H.sum(axis=1, keepdims=True)
      cbar_label = r"Fraction per $E_{\nu}^{\text{true}}$ bin"

  elif norm_mode == "total":
      # whole histogram sums to 1
      denom = H.sum()
      cbar_label = "Fraction of all events"

  else:
      denom = 1
      cbar_label = "Counts"

  H_plot = np.divide(H, denom, out=np.zeros_like(H), where=denom != 0)

  # Log z-scale, fixed (Z_VMIN, Z_VMAX). See Fig6_DUNE for context.
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

  # Overlay per-Eν_true slice median, mean, and 16-84% band so the energy
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
  ax.plot(xcenters, med,  color="#F5DEB3", lw=1.8)        # wheat / golden sand
  ax.plot(xcenters, mean, color="#E69F00", lw=1.7, ls="--")  # Wong orange — CB-friendly, contrasts with RdPu purple end
  # Per-plot legend dropped — make_Fig6_legend.py generates a standalone
  # Fig6_legend.pdf shared across all 4 Fig6 panels in the LaTeX subfigure.

  ax.set_ylabel(r"$E_\nu^{reco} - E_\nu^{true}$ [MeV]")
  ax.set_xlabel(r"$E_\nu^{true}$ [MeV]")
  ax.set_xlim(150, 2000)
  ax.set_ylim(-900, 300)

  plt.savefig(f"Fig6_plots/Fig6_HK_EnuRecoBias2D_{plot_name}.pdf")
  plt.close(fig)


_events = -1
_xbins = np.arange(-1000, 1000, 16)      # bias bins (MeV)
_ybins = np.arange(150, 2000 + 60, 60)   # Enu_true bins, HK 150 MeV - 2 GeV, 60 MeV/bin (3x coarser than original 20 MeV)

plot_Enu_bias(filename="../../Remade_April26/nuwro_25031_morestats/HK/HK_numu_FSI.flat.root", isNub=True, nEvents=_events, plot_name="FSI_numub", xbins=_xbins, ybins=_ybins)
plot_Enu_bias(filename="../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root", isNub=False, nEvents=_events, plot_name="FSI_numu", xbins=_xbins, ybins=_ybins)