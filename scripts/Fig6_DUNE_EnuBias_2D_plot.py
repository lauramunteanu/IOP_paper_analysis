from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

def plot_Enu_bias_numu(filename, nEvents, plot_name, withPion, xbins, ybins):
  Print(f"Reading: {filename}")
  fig, ax = plt.subplots()
  # ---------------------------------
  # Open input file and tree
  # ---------------------------------
  fin = ROOT.TFile.Open(filename)
  tree = fin.Get("FlatTree_VARS")

  bias_wo_list   = []
  bias_with_list = []
  Enu_t          = []
  # ---------------------------------
  # Event loop
  # ---------------------------------
  nentries = tree.GetEntries()
  nevs = 0
  if(nEvents == -1):
    nevs = nentries
  else:
    nevs = nEvents

  for i in range(nevs):
      tree.GetEntry(i)

      ELep     = tree.ELep
      Enu_true = tree.Enu_true
      nfsp     = tree.nfsp
      mode     = tree.Mode
      bad_event = False

      E  = tree.E
      px = tree.px
      py = tree.py
      pz = tree.pz
      pdg = tree.pdg

      # -------------------------
      # Lepton energy
      # -------------------------
      enuhad_wo   = ELep
      enuhad_with = ELep

      # Loop over final state particles
      for j in range(nfsp):

          apdg = abs(int(pdg[j]))
          Ej   = float(E[j])
          pxj  = float(px[j])
          pyj  = float(py[j])
          pzj  = float(pz[j])

          p2 = pxj*pxj + pyj*pyj + pzj*pzj

          # -------------------------
          # Heavy baryons (both defs)
          # -------------------------
          if apdg > 3000: # Remove contribution > 0
              bad_event = True
              continue
          if apdg > 2300 and apdg < 3000:
              enuhad_wo   += Ej
              enuhad_with += Ej
              continue

          # -------------------------
          # Definition 1 (no pion mass subtraction)
          # -------------------------
          if (apdg == 11 or (apdg > 17 and apdg < 2000)) and (apdg != 211):
              enuhad_wo += Ej

          elif apdg == 2212 or apdg == 211:
              mass2 = Ej*Ej - p2
              if mass2 > 0:
                  mass = np.sqrt(mass2)
                  enuhad_wo += (Ej - mass)

          # -------------------------
          # Definition 2 (with pion masses)
          # -------------------------
          if (apdg == 11 or (apdg > 17 and apdg < 2000)):
              enuhad_with += Ej

          elif apdg == 2212:
              mass2 = Ej*Ej - p2
              if mass2 > 0:
                  mass = np.sqrt(mass2)
                  enuhad_with += (Ej - mass)

      # -------------------------
      # Fill
      # -------------------------
      if(bad_event == False):
        bias_wo   = enuhad_wo   - Enu_true
        bias_with = enuhad_with - Enu_true
        bias_wo_list.append(bias_wo)
        bias_with_list.append(bias_with)
        Enu_t.append(Enu_true)
      else:
         continue
  # ---------------------------------
  # Write output
  # ---------------------------------
  bias_wo_list = np.array(bias_wo_list)
  bias_with_list = np.array(bias_with_list)
  Enu_t = np.array(Enu_t)

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

  mesh = ax.pcolormesh(
      xedges,
      yedges,
      H_plot.T,
      cmap="viridis",
      shading="auto"
  )

  plt.colorbar(mesh, ax=ax, label=cbar_label)

  if(withPion==True):
    ax.set_ylabel(r"$E_\nu^{\text{had}} - E_\nu^{\text{true}}$ [GeV]")
  else:
    ax.set_ylabel(r"$E_\nu^{\text{avail}} - E_\nu^{\text{true}}$ [GeV]")

  if("bar" in plot_name):
     ax.set_title(r"$\bar{\nu}_{\mu}$")
  else:
     ax.set_title(r"$\nu_{\mu}$")
     
     
  ax.set_xlabel(r"$E_\nu^{\text{true}}$ [GeV]")
  plt.savefig(f"Fig6_plots/Fig6_DUNE_EnuRecoBias2D_{plot_name}.pdf")
  fin.Close()
  # plt.show()


_events = -1
_xbins = np.arange(-0.4, 0.4, 0.05)      # bias bins
_ybins = np.linspace(0, 3, 100)     # Enu bins (adjust range!)

plot_Enu_bias_numu(filename="../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root", nEvents=_events, plot_name="FSI_WithoutPion_numu", withPion=False, xbins=_xbins, ybins=_ybins)
plot_Enu_bias_numu(filename="../../Remade_April26/DUNE/DUNE_numub_FSI.flat.root", nEvents=_events, plot_name="FSI_WithoutPion_numubar", withPion=False, xbins=_xbins, ybins=_ybins)

plot_Enu_bias_numu(filename="../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root", nEvents=_events, plot_name="FSI_WithPion_numu", withPion=True, xbins=_xbins, ybins=_ybins)
plot_Enu_bias_numu(filename="../../Remade_April26/DUNE/DUNE_numub_FSI.flat.root", nEvents=_events, plot_name="FSI_WithPion_numubar", withPion=True, xbins=_xbins, ybins=_ybins)