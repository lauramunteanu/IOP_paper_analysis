from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

outfile = open("Stats_table.txt", "w")
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


def plot_HK_Enu_bias(ax, filename, index, label, nEvents, vertex=False):
  global outfile
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  diff_all = diff_enu_qe_arr(arr, vertex=vertex)
  diff_sel = diff_all[np.abs(diff_all) < 1000]

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


def plot_DUNE_Enu_bias(ax, filename, index, label, nEvents, withPiCorr, vertex=False):
    global outfile
    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
    bias_wo_list, bias_with_list, _ = enu_had_arr(arr, vertex=vertex)
    bias_wo_list = bias_wo_list * 1000.0     # GeV -> MeV (keep parity with Jake's plot)
    bias_with_list = bias_with_list * 1000.0
    diff_sel = bias_with_list if withPiCorr else bias_wo_list
    
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

# All FSI/noFSI pairs come from the SAME FSI files; vertex=True gives the
# 'no FSI' rows (correlated stat errors with the FSI rows).
HK_NUMU    = "../../Remade_April26/nuwro_25031_morestats/HK/HK_numu_FSI.flat.root"
HK_NUMUBAR = "../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root"
DUNE_NUMU  = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root"
DUNE_NUMUB = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root"

fig, ax = make_fig('box')
plot_HK_Enu_bias(ax, filename=HK_NUMU,    index=0, label=r"HK $\nu_{\mu}$ no FSI",       nEvents=_events, vertex=True)
plot_HK_Enu_bias(ax, filename=HK_NUMU,    index=1, label=r"HK $\nu_{\mu}$ FSI",          nEvents=_events, vertex=False)
plot_HK_Enu_bias(ax, filename=HK_NUMUBAR, index=2, label=r"HK $\bar{\nu}_{\mu}$ no FSI", nEvents=_events, vertex=True)
plot_HK_Enu_bias(ax, filename=HK_NUMUBAR, index=3, label=r"HK $\bar{\nu}_{\mu}$ FSI",    nEvents=_events, vertex=False)

plot_DUNE_Enu_bias(ax, filename=DUNE_NUMU,  index=4, label=r"DUNE $\nu_{\mu}$ no FSI $T_{\pi}$",       nEvents=_events, withPiCorr=True,  vertex=True)
plot_DUNE_Enu_bias(ax, filename=DUNE_NUMUB, index=5, label=r"DUNE $\bar{\nu}_{\mu}$ no FSI $T_{\pi}$", nEvents=_events, withPiCorr=True,  vertex=True)
plot_DUNE_Enu_bias(ax, filename=DUNE_NUMU,  index=6, label=r"DUNE $\nu_{\mu}$ FSI $T_{\pi}$",          nEvents=_events, withPiCorr=True,  vertex=False)
plot_DUNE_Enu_bias(ax, filename=DUNE_NUMUB, index=7, label=r"DUNE $\bar{\nu}_{\mu}$ FSI $T_{\pi}$",    nEvents=_events, withPiCorr=True,  vertex=False)

plot_DUNE_Enu_bias(ax, filename=DUNE_NUMU,  index=8,  label=r"DUNE $\nu_{\mu}$ no FSI $E_{\pi}$",       nEvents=_events, withPiCorr=False, vertex=True)
plot_DUNE_Enu_bias(ax, filename=DUNE_NUMUB, index=9,  label=r"DUNE $\bar{\nu}_{\mu}$ no FSI $E_{\pi}$", nEvents=_events, withPiCorr=False, vertex=True)
plot_DUNE_Enu_bias(ax, filename=DUNE_NUMU,  index=10, label=r"DUNE $\nu_{\mu}$ FSI $E_{\pi}$",          nEvents=_events, withPiCorr=False, vertex=False)
plot_DUNE_Enu_bias(ax, filename=DUNE_NUMUB, index=11, label=r"DUNE $\bar{\nu}_{\mu}$ FSI $E_{\pi}$",    nEvents=_events, withPiCorr=False, vertex=False)

## Increase FSI
plot_HK_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/HK/ChangeMFP/HK_numu_0p7MFP_FSI.flat.root", index=12, label = r"HK $\nu_{\mu}$ 0.7x MFP", nEvents=_events)
plot_HK_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/HK/ChangeMFP/HK_numu_1p3MFP_FSI.flat.root", index=13, label = r"HK $\nu_{\mu}$ 1.3x MFP", nEvents=_events)
plot_HK_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/HK/ChangeMFP/HK_numubar_0p7MFP_FSI.flat.root", index=14, label = r"HK $\bar{\nu}_{\mu}$ 0.7x MFP", nEvents=_events)
plot_HK_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/HK/ChangeMFP/HK_numubar_1p3MFP_FSI.flat.root", index=15, label = r"HK $\bar{\nu}_{\mu}$ 1.3x MFP", nEvents=_events)

plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/DUNE/plus30_FSI/DUNE_numu_0p7MFP_FSI.flat.root",  index=16, label = r"DUNE $\nu_{\mu}$ 0.7x MFP $T_{\pi}$", nEvents=_events, withPiCorr=True)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/DUNE/plus30_FSI/DUNE_numub_0p7MFP_FSI.flat.root", index=17, label = r"DUNE $\bar{\nu}_{\mu}$ 0.7x MFP $T_{\pi}$", nEvents=_events, withPiCorr=True)
# plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/DUNE/plus30_FSI/DUNE_numu_1p3MFP_FSI.flat.root",  index=18, label = r"DUNE $\nu_{\mu}$ 1.3x MFP $T_{\pi}$", nEvents=_events, withPiCorr=True)  # 1p3MFP not in project share
# plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/DUNE/plus30_FSI/DUNE_numub_1p3MFP_FSI.flat.root", index=19, label = r"DUNE $\bar{\nu}_{\mu}$ 1.3x MFP $T_{\pi}$", nEvents=_events, withPiCorr=True)  # 1p3MFP not in project share

plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/DUNE/plus30_FSI/DUNE_numu_0p7MFP_FSI.flat.root",  index=20, label = r"DUNE $\nu_{\mu}$ 0.7x MFP $E_{\pi}$", nEvents=_events, withPiCorr=False)
plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/DUNE/plus30_FSI/DUNE_numub_0p7MFP_FSI.flat.root", index=21, label = r"DUNE $\bar{\nu}_{\mu}$ 0.7x MFP $E_{\pi}$", nEvents=_events, withPiCorr=False)
# plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/DUNE/plus30_FSI/DUNE_numu_1p3MFP_FSI.flat.root",  index=22, label = r"DUNE $\nu_{\mu}$ 1.3x MFP $E_{\pi}$", nEvents=_events, withPiCorr=False)  # 1p3MFP not in project share
# plot_DUNE_Enu_bias(ax, filename="../../Remade_April26/nuwro_25031_morestats/DUNE/plus30_FSI/DUNE_numub_1p3MFP_FSI.flat.root", index=23, label = r"DUNE $\bar{\nu}_{\mu}$ 1.3x MFP $E_{\pi}$", nEvents=_events, withPiCorr=False)  # 1p3MFP not in project share

ax.set_xlabel(r"Absolute $E_{\nu}^{\text{reco}}$ bias [MeV]")
plt.savefig("BW_plots/BW_test_capped_bias.pdf")
plt.close(fig)
outfile.close()