from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

def plot_Enu_bias_numu(ax, ax_ratio, filename, nEvents, withPion, nominal=False, counts_nom=None):
  # ---------------------------------
  # Open input file and tree
  # ---------------------------------
  fin = ROOT.TFile.Open(filename)
  tree = fin.Get("FlatTree_VARS")

  bias_wo_list   = []
  bias_with_list = []

  # ---------------------------------
  # Event loop
  # ---------------------------------
  nentries = tree.GetEntries()
  nevs = 0
  fScaleFactor = 0

  if(nEvents == -1):
    nevs = nentries
  else:
    nevs = nEvents

  for i in range(nevs):
      tree.GetEntry(i)

      ELep     = tree.ELep
      Enu_true = tree.Enu_true
      nfsp     = tree.nfsp
      bad_event = False

      _fscalefactor = tree.fScaleFactor
      if(_fscalefactor > fScaleFactor):
         fScaleFactor = _fscalefactor

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
          # Heavy baryons
          # -------------------------
          if apdg > 3000:
              bad_event = True
              continue

          if apdg > 2300 and apdg < 3000:
              enuhad_wo   += Ej
              enuhad_with += Ej
              continue

          # -------------------------
          # Definition 1: no pion mass subtraction
          # -------------------------
          if (apdg == 11 or (apdg > 17 and apdg < 2000)) and (apdg != 211):
              enuhad_wo += Ej

          elif apdg == 2212 or apdg == 211:
              mass2 = Ej*Ej - p2
              if mass2 > 0:
                  mass = np.sqrt(mass2)
                  enuhad_wo += (Ej - mass)

          # -------------------------
          # Definition 2: with pion masses
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
      else:
         continue

  # ---------------------------------
  # Histogram setup
  # ---------------------------------
  bias_wo_list = np.array(bias_wo_list)
  bias_with_list = np.array(bias_with_list)

  bin_width = 0.04
  bins = np.arange(-3, 1, step=bin_width)

  weights_with = fScaleFactor*np.ones_like(bias_with_list)/bin_width
  weights_wo   = fScaleFactor*np.ones_like(bias_wo_list)/bin_width

  # ---------------------------------
  # Choose which definition to plot
  # ---------------------------------
  if(withPion == True):
    bias = bias_with_list
    weights = weights_with
    pion_label = "w/ pion mass"
  else:
    bias = bias_wo_list
    weights = weights_wo
    pion_label = "w/o pion mass"

  # ---------------------------------
  # Choose FSI / noFSI 
  # ---------------------------------
  if("noFSI" in filename):
    color = dark_blue
    label = pion_label + " noFSI"
  else:
    color = dark_red
    label = pion_label + " FSI"

  # ---------------------------------
  # Main histogram
  # ---------------------------------
  ax.hist(
      bias,
      bins=bins,
      histtype='step',
      weights=weights,
      color=color,
      linewidth=1.5,
      label=label
  )

  custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
  labels.append(label)

  # ---------------------------------
  # Ratio histogram
  # ---------------------------------
  counts, edges = np.histogram(bias, weights=weights, bins=bins)

  if nominal == True:
      ax_ratio.hlines(1, bins[0], bins[-1], linestyle='--', color='black')
      fin.Close()
      Print(f"Done: {filename}")
      return counts

  else:
      ratio = counts / counts_nom
      ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

      ax_ratio.step(
        edges,
        np.r_[ratio, ratio[-1]],
        color=color,
        linestyle='-',
        where='post'
        )

  fin.Close()
  Print(f"Done: {filename}")
  return counts


_events = 10000

plot_configs = [
    {
        "withPion": True,
        "flavor": "numu",
        "nofsi": "../../Remade_April26/DUNE/DUNE_numu_noFSI.flat.root",
        "fsi": "../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root",
        "title": r"$\nu_{\mu}$, w/ pion mass",
        "outfile": "Fig4_plots/Fig4_DUNE_EnuRecoFSIBias_WithPion_numu.pdf"
    },
    {
        "withPion": True,
        "flavor": "numubar",
        "nofsi": "../../Remade_April26/DUNE/DUNE_numub_noFSI.flat.root",
        "fsi": "../../Remade_April26/DUNE/DUNE_numub_FSI.flat.root",
        "title": r"$\bar{\nu}_{\mu}$, w/ pion mass",
        "outfile": "Fig4_plots/Fig4_DUNE_EnuRecoFSIBias_WithPion_numubar.pdf"
    },
    {
        "withPion": False,
        "flavor": "numu",
        "nofsi": "../../Remade_April26/DUNE/DUNE_numu_noFSI.flat.root",
        "fsi": "../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root",
        "title": r"$\nu_{\mu}$, w/o pion mass",
        "outfile": "Fig4_plots/Fig4_DUNE_EnuRecoFSIBias_WithoutPion_numu.pdf"
    },
    {
        "withPion": False,
        "flavor": "numubar",
        "nofsi": "../../Remade_April26/DUNE/DUNE_numub_noFSI.flat.root",
        "fsi": "../../Remade_April26/DUNE/DUNE_numub_FSI.flat.root",
        "title": r"$\bar{\nu}_{\mu}$, w/o pion mass",
        "outfile": "Fig4_plots/Fig4_DUNE_EnuRecoFSIBias_WithoutPion_numubar.pdf"
    }
]

for cfg in plot_configs:

    custom_lines, labels = [], []

    fig, (ax, ax_ratio) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
    )

    counts_nom = plot_Enu_bias_numu(
        ax=ax,
        ax_ratio=ax_ratio,
        filename=cfg["nofsi"],
        nEvents=_events,
        withPion=cfg["withPion"],
        nominal=True
    )

    plot_Enu_bias_numu(
        ax=ax,
        ax_ratio=ax_ratio,
        filename=cfg["fsi"],
        nEvents=_events,
        withPion=cfg["withPion"],
        nominal=False,
        counts_nom=counts_nom
    )

    ax.legend(custom_lines, labels, loc='best', fontsize=15)
    ax.set_title(cfg["title"])

    ax.set_ylabel(
        r"$\text{d}\sigma/\text{d}E_{\nu}^{\text{bias}}$ "
        r"[cm$^{2}$/nucleon GeV]"
    )

    ax_ratio.set_xlabel(r"$E_{\nu}^{\text{bias}}$ [GeV]")
    ax_ratio.set_ylabel("FSI/noFSI")
    ax_ratio.set_ylim(0, 2)

    plt.savefig(cfg["outfile"])
    plt.close(fig)