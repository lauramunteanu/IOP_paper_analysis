from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)


def _open_stats_file(mode):
    outfile = open(outpath(".", f"Stats_table_{mode}.txt"), "w")
    outfile.write(
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\hline\n"
        "Name & median & mean & q10 & q90 & q16 & q84 \\\\\n"
        "\\hline\n"
    )
    return outfile


def write_stats(outfile, stats):
    outfile.write(
        f"{stats[0]} & {stats[1]:.4f} & {stats[2]:.4f} & {stats[3]:.4f} & {stats[4]:.4f} & {stats[5]:.4f} & {stats[6]:.4f} \\\\\n"
    )


def plot_HK_Enu_bias(ax, filename, index, label, nEvents, mode, vertex=False):
  global outfile
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  diff_all = bias_arr(arr, "qe", kind=mode, vertex=vertex)
  # Cap on |diff| to drop tails — threshold scales with mode units.
  cap = 1000.0 if mode == "abs" else 1.0
  diff_sel = diff_all[np.abs(diff_all) < cap]

  q16, q50, q84 = np.percentile(diff_sel, [16, 50, 84])
  q10, q90      = np.percentile(diff_sel, [10, 90])
  vmin = q10
  vmax = q90
  mean = diff_sel.mean()

  write_stats(outfile, [label, q50, mean, q10, q90, q16, q84])

  stats = [{
      "label": label,
      "med": q50, "q1": q16, "q3": q84,
      "whislo": vmin, "whishi": vmax,
      "fliers": [], "mean": mean,
  }]

  artists = ax.bxp(
      stats, vert=False, positions=[index],
      showfliers=False, patch_artist=True, widths=0.6,
  )

  for box in artists["boxes"]:
      box.set(facecolor='indigo', edgecolor="purple", alpha=0.7)
  for median in artists["medians"]:
      median.set(color="black", linewidth=2)
  for whisker in artists["whiskers"]:
      whisker.set(color="purple", linewidth=1.2)
  for cap_artist in artists["caps"]:
      cap_artist.set(color="purple", linewidth=1.2)

  ax.invert_yaxis()


def plot_DUNE_Enu_bias(ax, filename, index, label, nEvents, withPiCorr, mode, vertex=False):
    global outfile
    arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
    observable = "had" if withPiCorr else "avail"
    diff_sel = bias_arr(arr, observable, kind=mode, vertex=vertex)

    q16, q50, q84 = np.percentile(diff_sel, [16, 50, 84])
    q10, q90      = np.percentile(diff_sel, [10, 90])
    vmin = q10
    vmax = q90
    mean = diff_sel.mean()

    write_stats(outfile, [label, q50, mean, q10, q90, q16, q84])

    stats = [{
        "label": label,
        "med": q50, "q1": q16, "q3": q84,
        "whislo": vmin, "whishi": vmax,
        "fliers": [], "mean": mean,
    }]

    artists = ax.bxp(
        stats, vert=False, positions=[index],
        showfliers=False, patch_artist=True, widths=0.6,
    )

    for box in artists["boxes"]:
        box.set(facecolor='indigo', edgecolor="purple", alpha=0.7)
    for median in artists["medians"]:
        median.set(color="black", linewidth=2)
    for whisker in artists["whiskers"]:
        whisker.set(color="purple", linewidth=1.2)
    for cap_artist in artists["caps"]:
        cap_artist.set(color="purple", linewidth=1.2)

    ax.invert_yaxis()


_events = 100000

# All FSI/noFSI pairs come from the SAME FSI files; vertex=True gives the
# 'no FSI' rows (correlated stat errors with the FSI rows).
HK_NUMU    = "../../Remade_April26/nuwro_25031_morestats/HK/HK_numu_FSI.flat.root"
HK_NUMUBAR = "../../Remade_April26/nuwro_25031_morestats/HK/HK_numubar_FSI.flat.root"
DUNE_NUMU  = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root"
DUNE_NUMUB = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numub_FSI.flat.root"

# MFP samples for the 0.7x rows (NB: DUNE 1.3x is not in the project share — see
# the followups memory; rows are commented out below).
HK_NUMU_07MFP    = "../../Remade_April26/nuwro_25031_morestats/HK/ChangeMFP/HK_numu_0p7MFP_FSI.flat.root"
HK_NUMU_13MFP    = "../../Remade_April26/nuwro_25031_morestats/HK/ChangeMFP/HK_numu_1p3MFP_FSI.flat.root"
HK_NUMUBAR_07MFP = "../../Remade_April26/nuwro_25031_morestats/HK/ChangeMFP/HK_numubar_0p7MFP_FSI.flat.root"
HK_NUMUBAR_13MFP = "../../Remade_April26/nuwro_25031_morestats/HK/ChangeMFP/HK_numubar_1p3MFP_FSI.flat.root"
DUNE_NUMU_07MFP  = "../../Remade_April26/nuwro_25031_morestats/DUNE/ChangeMFP/DUNE_numu_0p7MFP_FSI.flat.root"
DUNE_NUMUB_07MFP = "../../Remade_April26/nuwro_25031_morestats/DUNE/ChangeMFP/DUNE_numub_0p7MFP_FSI.flat.root"

for mode in ("abs", "rel"):
    outfile = _open_stats_file(mode)
    fig, ax = make_fig('box')
    plot_HK_Enu_bias(ax, filename=HK_NUMU,    index=0, label=r"HK $\nu_{\mu}$ no FSI",       nEvents=_events, mode=mode, vertex=True)
    plot_HK_Enu_bias(ax, filename=HK_NUMU,    index=1, label=r"HK $\nu_{\mu}$ FSI",          nEvents=_events, mode=mode, vertex=False)
    plot_HK_Enu_bias(ax, filename=HK_NUMUBAR, index=2, label=r"HK $\bar{\nu}_{\mu}$ no FSI", nEvents=_events, mode=mode, vertex=True)
    plot_HK_Enu_bias(ax, filename=HK_NUMUBAR, index=3, label=r"HK $\bar{\nu}_{\mu}$ FSI",    nEvents=_events, mode=mode, vertex=False)

    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMU,  index=4, label=r"DUNE $\nu_{\mu}$ no FSI $T_{\pi}$",       nEvents=_events, withPiCorr=True,  mode=mode, vertex=True)
    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMUB, index=5, label=r"DUNE $\bar{\nu}_{\mu}$ no FSI $T_{\pi}$", nEvents=_events, withPiCorr=True,  mode=mode, vertex=True)
    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMU,  index=6, label=r"DUNE $\nu_{\mu}$ FSI $T_{\pi}$",          nEvents=_events, withPiCorr=True,  mode=mode, vertex=False)
    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMUB, index=7, label=r"DUNE $\bar{\nu}_{\mu}$ FSI $T_{\pi}$",    nEvents=_events, withPiCorr=True,  mode=mode, vertex=False)

    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMU,  index=8,  label=r"DUNE $\nu_{\mu}$ no FSI $E_{\pi}$",       nEvents=_events, withPiCorr=False, mode=mode, vertex=True)
    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMUB, index=9,  label=r"DUNE $\bar{\nu}_{\mu}$ no FSI $E_{\pi}$", nEvents=_events, withPiCorr=False, mode=mode, vertex=True)
    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMU,  index=10, label=r"DUNE $\nu_{\mu}$ FSI $E_{\pi}$",          nEvents=_events, withPiCorr=False, mode=mode, vertex=False)
    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMUB, index=11, label=r"DUNE $\bar{\nu}_{\mu}$ FSI $E_{\pi}$",    nEvents=_events, withPiCorr=False, mode=mode, vertex=False)

    ## Increase FSI
    plot_HK_Enu_bias(ax, filename=HK_NUMU_07MFP,    index=12, label=r"HK $\nu_{\mu}$ 0.7x MFP",       nEvents=_events, mode=mode)
    plot_HK_Enu_bias(ax, filename=HK_NUMU_13MFP,    index=13, label=r"HK $\nu_{\mu}$ 1.3x MFP",       nEvents=_events, mode=mode)
    plot_HK_Enu_bias(ax, filename=HK_NUMUBAR_07MFP, index=14, label=r"HK $\bar{\nu}_{\mu}$ 0.7x MFP", nEvents=_events, mode=mode)
    plot_HK_Enu_bias(ax, filename=HK_NUMUBAR_13MFP, index=15, label=r"HK $\bar{\nu}_{\mu}$ 1.3x MFP", nEvents=_events, mode=mode)

    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMU_07MFP,  index=16, label=r"DUNE $\nu_{\mu}$ 0.7x MFP $T_{\pi}$",       nEvents=_events, withPiCorr=True,  mode=mode)
    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMUB_07MFP, index=17, label=r"DUNE $\bar{\nu}_{\mu}$ 0.7x MFP $T_{\pi}$", nEvents=_events, withPiCorr=True,  mode=mode)
    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMU_07MFP,  index=20, label=r"DUNE $\nu_{\mu}$ 0.7x MFP $E_{\pi}$",       nEvents=_events, withPiCorr=False, mode=mode)
    plot_DUNE_Enu_bias(ax, filename=DUNE_NUMUB_07MFP, index=21, label=r"DUNE $\bar{\nu}_{\mu}$ 0.7x MFP $E_{\pi}$", nEvents=_events, withPiCorr=False, mode=mode)

    if mode == "abs":
        ax.set_xlabel(r"Absolute $E_{\nu}^{\rm reco}$ bias [MeV]")
    else:
        ax.set_xlabel(r"Relative $E_{\nu}^{\rm reco}$ bias $(E^{\rm reco} - E^{\rm true}) / E^{\rm true}$")

    plt.savefig(outpath("BW_plots", f"BW_test_capped_bias_{mode}.pdf"))
    plt.close(fig)
    outfile.close()
