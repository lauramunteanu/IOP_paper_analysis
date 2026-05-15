from FlatTreeMod import *
from matplotlib.colors import LogNorm
ROOT.gROOT.SetBatch(True)

# Shared z-scale across every Fig6 panel — the per-panel colorbar is dropped
# and the standalone colorbar PDF (make_Fig6_colorbar.py) becomes the
# common z-axis in the LaTeX subfigure environment.
Z_CMAP = "RdPu"
Z_VMAX = 0.3
Z_VMIN = Z_VMAX * 1e-4

def plot_Enu_bias_numu(filename, nEvents, plot_name, withPion, xbins, ybins):
  fig, ax = make_fig('single')
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  bias_wo_GeV, bias_with_GeV, valid = enu_had_arr(arr, vertex=False)
  # convert to MeV for unified axes
  bias_wo_list   = bias_wo_GeV   * 1000.0
  bias_with_list = bias_with_GeV * 1000.0
  Enu_t = np.asarray(arr['Enu_true'])[valid] * 1000.0  # MeV

  if withPion:
    yvals = bias_with_list
  else:
    yvals = bias_wo_list

  H, xedges, yedges = np.histogram2d(
      Enu_t,
      yvals,
      bins=[ybins, xbins]
  )

  # Choose normalisation mode
  norm_mode = "y"

  if norm_mode == "x":
      # each Enu bin sums to 1
      denom = H.sum(axis=0, keepdims=True)
      cbar_label = "Fraction per bias bin"

  elif norm_mode == "y":
      # each bias bin sums to 1
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

  # Log z-scale, fixed (Z_VMIN, Z_VMAX) so every Fig6 panel uses the same
  # normalisation. Empty cells get the bottom-of-cmap colour. The colorbar
  # is NOT drawn here — make_Fig6_colorbar.py produces a standalone
  # Fig6_colorbar.pdf that serves as the shared z-axis in the LaTeX
  # subfigure environment.
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
      in_slice = (Enu_t >= xedges[ix]) & (Enu_t < xedges[ix + 1])
      if in_slice.sum() < 30:
          med.append(np.nan); mean.append(np.nan); q16.append(np.nan); q84.append(np.nan); continue
      s = yvals[in_slice]
      med.append(np.median(s))
      mean.append(np.mean(s))
      q16.append(np.percentile(s, 16))
      q84.append(np.percentile(s, 84))
  ax.fill_between(xcenters, q16, q84, color="#F5DEB3", alpha=0.30)
  ax.plot(xcenters, med,  color="#F5DEB3", lw=1.8)        # wheat / golden sand
  ax.plot(xcenters, mean, color="#E69F00", lw=1.7, ls="--")  # Wong orange — CB-friendly, contrasts with RdPu purple end
  # Per-plot legend dropped — make_Fig6_legend.py generates a standalone
  # Fig6_legend.pdf shared across all 4 Fig6 panels in the LaTeX subfigure.

  if(withPion==True):
    ax.set_ylabel(r"$E_\nu^{\rm had} - E_\nu^{\rm true}$ [MeV]")
  else:
    ax.set_ylabel(r"$E_\nu^{\rm avail} - E_\nu^{\rm true}$ [MeV]")

  ax.set_xlabel(r"$E_\nu^{\rm true}$ [MeV]")
  ax.set_xlim(300, 6000)
  ax.set_ylim(-900, 300)
  plt.savefig(f"Fig6_plots/Fig6_DUNE_EnuRecoBias2D_{plot_name}.pdf")
  plt.close(fig)


_events = -1
_xbins = np.arange(-1000, 1000 + 20, 20)   # bias bins (MeV)
_ybins = np.arange(300, 6000 + 120, 120)   # Enu_true bins, DUNE 300 MeV - 6 GeV, 120 MeV/bin

plot_Enu_bias_numu(filename="../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root", nEvents=_events, plot_name="FSI_WithoutPion_numu", withPion=False, xbins=_xbins, ybins=_ybins)
plot_Enu_bias_numu(filename="../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root", nEvents=_events, plot_name="FSI_WithoutPion_numubar", withPion=False, xbins=_xbins, ybins=_ybins)

plot_Enu_bias_numu(filename="../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root", nEvents=_events, plot_name="FSI_WithPion_numu", withPion=True, xbins=_xbins, ybins=_ybins)
plot_Enu_bias_numu(filename="../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root", nEvents=_events, plot_name="FSI_WithPion_numubar", withPion=True, xbins=_xbins, ybins=_ybins)