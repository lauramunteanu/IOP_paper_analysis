from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

outfile = open("Stats_table.txt", "a")
outfile.write(
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\begin{tabular}{lcccccc}\n"
    "\\hline\n"
    "Name & median & mean & q10 & q90 & q16 & q84 \\\\\n"
    "\\hline\n"
)

def write_stats(outfile, stats):
    outfile.write(
        f"{stats[0]} & {stats[1]:.3f} & {stats[2]:.3f} & {stats[3]:.3f} & {stats[4]:.3f} & {stats[5]:.3f} & {stats[6]:.3f} \\\\\n"
    )


def plot_HK_Enu_bias(ax, filename, index, label, nEvents):
  global outfile
  Print(f"Reading: {filename}")
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
  nevs = 0
  if(nEvents == -1):
      nevs = nentries
  else:
      nevs = nEvents

  for i in range(nevs):
    tree.GetEntry(i)
    Enu_true = tree.Enu_true*1000
    Enu_QE   = tree.Enu_QE*1000
    isCC0pi     = tree.flagCC0pi

    if(isCC0pi == True):
      diff = Enu_QE - Enu_true
      if(abs(diff) < 1000):
        diff_sel.append(diff)


  diff_sel = np.array(diff_sel)

  # Central 68% interval
  q16, q50, q84 = np.percentile(diff_sel, [16, 50, 84])
  q10, q90      = np.percentile(diff_sel, [10,90])
  # vmin = diff_sel.min()
  vmin = q10
  # vmax = diff_sel.max()
  vmax = q90
  mean = diff_sel.mean()

  write_stats(outfile, [label, q50, mean, q10, q90, q16, q84])

  stats = [{
      "label": label,
      "med": q50,
      "q1": q16,
      "q3": q84,
      "whislo": vmin,
      "whishi": vmax,
      "fliers": [],
      "mean": mean,
  }]

  artists = ax.bxp(
      stats,
      vert=False,
      positions=[index],
      showfliers=False,
      patch_artist=True,
      widths=0.6
  )

  # Optional styling
  for box in artists["boxes"]:
      box.set(facecolor='indigo', edgecolor="purple", alpha=0.7)

  for median in artists["medians"]:
      median.set(color="black", linewidth=2)

  for whisker in artists["whiskers"]:
      whisker.set(color="purple", linewidth=1.2)

  for cap in artists["caps"]:
      cap.set(color="purple", linewidth=1.2)

  ax.invert_yaxis()
  fin.Close()


def plot_DUNE_Enu_bias(ax, filename, index, label, nEvents, withPiCorr):
    global outfile
    Print(f"Reading: {filename}")
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
    if(nEvents == -1):
        nevs = nentries
    else:
        nevs = nEvents

    for i in range(nevs):
        tree.GetEntry(i)
        bad_event = False
        ELep     = tree.ELep
        Enu_true = tree.Enu_true
        nfsp     = tree.nfsp

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
            # Remove heavy stuff
            # -------------------------
            if apdg > 3000:
                bad_event = True
                continue
            # -------------------------
            # Heavy baryons
            # -------------------------
            if 2300 < apdg < 3000:
                enuhad_wo   += Ej
                enuhad_with += Ej
                continue
            # -------------------------
            # Definition 1
            # -------------------------
            if (apdg == 11 or (17 < apdg < 2000)) and apdg != 211:
                enuhad_wo += Ej
            elif apdg in (2212, 211):
                mass2 = Ej*Ej - p2
                if mass2 > 0:
                    enuhad_wo += Ej - np.sqrt(mass2)
            # -------------------------
            # Definition 2
            # -------------------------
            if (apdg == 11 or (17 < apdg < 2000)):
                enuhad_with += Ej
            elif apdg == 2212:
                mass2 = Ej*Ej - p2
                if mass2 > 0:
                    enuhad_with += Ej - np.sqrt(mass2)

        # -------------------------
        # Fill
        # -------------------------
        bias_wo   = enuhad_wo   - Enu_true
        bias_with = enuhad_with - Enu_true

        if(bad_event == False):
            # Total
            bias_wo_list.append(1000*bias_wo)
            bias_with_list.append(1000*bias_with)
        else:
            continue

    bias_wo_list = np.array(bias_wo_list)
    bias_with_list = np.array(bias_with_list)
    fin.Close()

    diff_sel = []
    if(withPiCorr == True):
        diff_sel = bias_with_list
    else:
        diff_sel = bias_wo_list
    
    # Central 68% interval
    q16, q50, q84 = np.percentile(diff_sel, [16, 50, 84])
    q10, q90      = np.percentile(diff_sel, [10,90])
    # vmin = diff_sel.min()
    vmin = q10
    # vmax = diff_sel.max()
    vmax = q90
    mean = diff_sel.mean()
  
    write_stats(outfile, [label, q50, mean, q10, q90, q16, q84])


    stats = [{
        "label": label,
        "med": q50,
        "q1": q16,
        "q3": q84,
        "whislo": vmin,
        "whishi": vmax,
        "fliers": [],
        "mean": mean,
    }]

    # index is used to set which row it is on
    artists = ax.bxp(
        stats,
        vert=False,
        positions=[index],
        showfliers=False,
        patch_artist=True,
        widths=0.6
    )

    # Set colours
    for box in artists["boxes"]:
        box.set(facecolor='indigo', edgecolor="purple", alpha=0.7)

    for median in artists["medians"]:
        median.set(color="black", linewidth=2)

    for whisker in artists["whiskers"]:
        whisker.set(color="purple", linewidth=1.2)

    for cap in artists["caps"]:
        cap.set(color="purple", linewidth=1.2)

    ax.invert_yaxis()



_events = 100000

fig, ax = plt.subplots()
plot_HK_Enu_bias(ax, filename="../../Remade_April26/HK/HK_numu_noFSI.flat.root", index=0, label = r"HK $\nu_{\mu}$ no FSI", nEvents=_events)
plot_HK_Enu_bias(ax, filename="../../Remade_April26/HK/HK_numu_FSI.flat.root", index=1, label = r"HK $\nu_{\mu}$ FSI", nEvents=_events)
plot_HK_Enu_bias(ax, filename="../../Remade_April26/HK/HK_numubar_noFSI.flat.root", index=2, label = r"HK $\bar{\nu}_{\mu}$ no FSI", nEvents=_events)
plot_HK_Enu_bias(ax, filename="../../Remade_April26/HK/HK_numubar_FSI.flat.root", index=3, label = r"HK $\bar{\nu}_{\mu}$ FSI", nEvents=_events)

plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/DUNE_numu_noFSI.flat.root",  index=4, label = r"DUNE $\nu_{\mu}$ no FSI $T_{\pi}$", nEvents=_events, withPiCorr=True)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/DUNE_numub_noFSI.flat.root", index=5, label = r"DUNE $\bar{\nu}_{\mu}$ no FSI $T_{\pi}$", nEvents=_events, withPiCorr=True)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root",  index=6, label = r"DUNE $\nu_{\mu}$ FSI $T_{\pi}$", nEvents=_events, withPiCorr=True)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/DUNE_numub_FSI.flat.root", index=7, label = r"DUNE $\bar{\nu}_{\mu}$ FSI $T_{\pi}$", nEvents=_events, withPiCorr=True)

plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/DUNE_numu_noFSI.flat.root",  index=8, label = r"DUNE $\nu_{\mu}$ no FSI $E_{\pi}$", nEvents=_events, withPiCorr=False)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/DUNE_numub_noFSI.flat.root", index=9, label = r"DUNE $\bar{\nu}_{\mu}$ no FSI $E_{\pi}$", nEvents=_events, withPiCorr=False)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root",  index=10, label = r"DUNE $\nu_{\mu}$ FSI $E_{\pi}$", nEvents=_events, withPiCorr=False)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/DUNE_numub_FSI.flat.root", index=11, label = r"DUNE $\bar{\nu}_{\mu}$ FSI $E_{\pi}$", nEvents=_events, withPiCorr=False)

## Increase FSI
plot_HK_Enu_bias(ax, filename="../../Remade_April26/HK/plus30_FSI/HK_numu_0p7MFP_FSI.flat.root", index=12, label = r"HK $\nu_{\mu}$ 0.7x MFP", nEvents=_events)
plot_HK_Enu_bias(ax, filename="../../Remade_April26/HK/plus30_FSI/HK_numu_1p3MFP_FSI.flat.root", index=13, label = r"HK $\nu_{\mu}$ 1.3x MFP", nEvents=_events)
plot_HK_Enu_bias(ax, filename="../../Remade_April26/HK/plus30_FSI/HK_numubar_0p7MFP_FSI.flat.root", index=14, label = r"HK $\bar{\nu}_{\mu}$ 0.7x MFP", nEvents=_events)
plot_HK_Enu_bias(ax, filename="../../Remade_April26/HK/plus30_FSI/HK_numubar_1p3MFP_FSI.flat.root", index=15, label = r"HK $\bar{\nu}_{\mu}$ 1.3x MFP", nEvents=_events)

plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/plus30_FSI/DUNE_numu_0p7MFP_FSI.flat.root",  index=16, label = r"DUNE $\nu_{\mu}$ 0.7x MFP $T_{\pi}$", nEvents=_events, withPiCorr=True)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/plus30_FSI/DUNE_numub_0p7MFP_FSI.flat.root", index=17, label = r"DUNE $\bar{\nu}_{\mu}$ 0.7x MFP $T_{\pi}$", nEvents=_events, withPiCorr=True)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/plus30_FSI/DUNE_numu_1p3MFP_FSI.flat.root",  index=18, label = r"DUNE $\nu_{\mu}$ 1.3x MFP $T_{\pi}$", nEvents=_events, withPiCorr=True)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/plus30_FSI/DUNE_numub_1p3MFP_FSI.flat.root", index=19, label = r"DUNE $\bar{\nu}_{\mu}$ 1.3x MFP $T_{\pi}$", nEvents=_events, withPiCorr=True)

plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/plus30_FSI/DUNE_numu_0p7MFP_FSI.flat.root",  index=20, label = r"DUNE $\nu_{\mu}$ 0.7x MFP $E_{\pi}$", nEvents=_events, withPiCorr=False)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/plus30_FSI/DUNE_numub_0p7MFP_FSI.flat.root", index=21, label = r"DUNE $\bar{\nu}_{\mu}$ 0.7x MFP $E_{\pi}$", nEvents=_events, withPiCorr=False)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/plus30_FSI/DUNE_numu_1p3MFP_FSI.flat.root",  index=22, label = r"DUNE $\nu_{\mu}$ 1.3x MFP $E_{\pi}$", nEvents=_events, withPiCorr=False)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/DUNE/plus30_FSI/DUNE_numub_1p3MFP_FSI.flat.root", index=23, label = r"DUNE $\bar{\nu}_{\mu}$ 1.3x MFP $E_{\pi}$", nEvents=_events, withPiCorr=False)

ax.set_xlabel(r"Absolute $E_{\nu}^{\text{reco}}$ bias [MeV]")
plt.savefig("BW_plots/BW_test_capped_bias.pdf")
outfile.close()
plt.show()