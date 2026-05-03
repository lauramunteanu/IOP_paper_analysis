from FlatTreeMod import *
from matplotlib.colors import LogNorm
ROOT.gROOT.SetBatch(True)

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

  # Log z-scale: avoids saturated peak swallowing the off-peak structure.
  # Empty cells get the bottom-of-cmap colour so the canvas looks uniform
  # instead of having white "weird space" patches in low-stat regions.
  import matplotlib as _mpl
  cmap = _mpl.colormaps["magma"].copy()    # softer, more pleasing than viridis; white-line overlays still read clearly
  cmap.set_bad("black")
  vmax = float(H_plot.max())
  vmin = max(vmax * 1e-4, 1e-6)
  H_plot_for_log = np.where(H_plot > 0, H_plot, np.nan)
  mesh = ax.pcolormesh(
      xedges,
      yedges,
      H_plot_for_log.T,
      cmap=cmap,
      shading="auto",
      norm=LogNorm(vmin=vmin, vmax=vmax),
  )

  plt.colorbar(mesh, ax=ax, label=cbar_label)

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
  ax.fill_between(xcenters, q16, q84, color="white", alpha=0.20, label=r"16–84%")
  ax.plot(xcenters, med,  color="white",  lw=1.8, label="median")
  ax.plot(xcenters, mean, color="#882255", lw=1.7, ls="--", label="mean")  # Tol burgundy, CB-friendly, distinguishable from white median
  ax.legend(loc="upper right", frameon=True, facecolor="black",
            edgecolor="white", labelcolor="white")

  ax.set_ylabel(r"$E_\nu^{reco} - E_\nu^{true}$ [MeV]")
  ax.set_xlabel(r"$E_\nu^{true}$ [MeV]")
  ax.set_xlim(150, 2000)
  ax.set_ylim(-1000, 1000)
  if(isNub == True):
    ax.set_title(r"$\bar{\nu}_{\mu}$")
  else:
    ax.set_title(r"$\nu_{\mu}$")

  plt.savefig(f"Fig6_plots/Fig6_HK_EnuRecoBias2D_{plot_name}.pdf")
  plt.close(fig)


_events = -1
_xbins = np.arange(-1000, 1000, 16)      # bias bins (MeV)
_ybins = np.arange(150, 2000 + 60, 60)   # Enu_true bins, HK 150 MeV - 2 GeV, 60 MeV/bin (3x coarser than original 20 MeV)

plot_Enu_bias(filename="../../Remade_April26/HK/HK_numu_FSI.flat.root", isNub=True, nEvents=_events, plot_name="FSI_numub", xbins=_xbins, ybins=_ybins)
plot_Enu_bias(filename="../../Remade_April26/HK/HK_numubar_FSI.flat.root", isNub=False, nEvents=_events, plot_name="FSI_numu", xbins=_xbins, ybins=_ybins)