from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

def plot_Enu_bias(filename, isNub, nEvents, plot_name, xbins, ybins):
  Print(f"Reading: {filename}")
  fig, ax = plt.subplots()
  # ---------------------------------
  # Open input file and tree
  # ---------------------------------
  fin = ROOT.TFile.Open(filename)
  tree = fin.Get("FlatTree_VARS")

  # ---------------------------------
  # Event loop
  # ---------------------------------
  nentries    = tree.GetEntries()
  diff_sel    = []
  Enu_t_sel   = []
  Enu_QE_sel   = []
  nevs = 0
  if(nEvents == -1):
      nevs = nentries
  else:
      nevs = nEvents

  for i in range(nevs):

    tree.GetEntry(i)

    Enu_true = tree.Enu_true*1000
    Enu_QE   = tree.Enu_QE*1000
    nfsp     = tree.nfsp
    pdg      = tree.pdg
    isCC0pi     = tree.flagCC0pi

    if(isCC0pi == True):
    # -------------------------
    # Fill only if passed
    # -------------------------
        diff = Enu_QE - Enu_true

        diff_sel.append(diff)
        Enu_t_sel.append(Enu_true)
        Enu_QE_sel.append(Enu_QE)


  diff_sel = np.array(diff_sel)
  Enu_t_sel = np.array(Enu_t_sel)
  Enu_QE_sel = np.array(Enu_QE_sel)

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

  mesh = ax.pcolormesh(
      xedges,
      yedges,
      H_plot.T,
      cmap="viridis",
      shading="auto"
  )

  plt.colorbar(mesh, ax=ax, label=cbar_label)

  ax.set_ylabel(r"$E_\nu^{reco} - E_\nu^{true}$ [MeV]")
  ax.set_xlabel(r"$E_\nu^{true}$ [MeV]")
  if(isNub == True):
    ax.set_title(r"$\bar{\nu}_{\mu}$")
  else:
    ax.set_title(r"$\nu_{\mu}$")

  plt.savefig(f"Fig6_plots/Fig6_HK_EnuRecoBias2D_{plot_name}.pdf")
  fin.Close()


_events = -1
_xbins = np.arange(-1000, 1000, 16)      # bias bins
_ybins = np.linspace(0, 3000, 10)     # Enu bins 

plot_Enu_bias(filename="../../Remade_April26/HK/HK_numu_FSI.flat.root", isNub=True, nEvents=_events, plot_name="FSI_numub", xbins=_xbins, ybins=_ybins)
plot_Enu_bias(filename="../../Remade_April26/HK/HK_numubar_FSI.flat.root", isNub=False, nEvents=_events, plot_name="FSI_numu", xbins=_xbins, ybins=_ybins)