from FlatTreeMod import *
from collections import defaultdict
ROOT.gROOT.SetBatch(True)

def plot_Enu_bias(filename, label, isNuBar, nEvents, plot_name):
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
  diff_by_mode = defaultdict(list)   # per Mode
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
    mode     = tree.Mode
    isCC0pi     = tree.flagCC0pi

    if(isCC0pi == True):
      # -------------------------
      # Fill only if passed
      # -------------------------
      diff = Enu_QE - Enu_true

      diff_sel.append(diff)
      diff_by_mode[mode].append(diff)   # per Mode
      Enu_t_sel.append(Enu_true)
      Enu_QE_sel.append(Enu_QE)


  diff_sel = np.array(diff_sel)
  Enu_t_sel = np.array(Enu_t_sel)
  Enu_QE_sel = np.array(Enu_QE_sel)
  for m in diff_by_mode:
    diff_by_mode[m] = np.array(diff_by_mode[m])

  ax.hist(diff_sel, bins=np.arange(-1000, 1000, step=10), histtype='step', weights=np.ones_like(diff_sel), color=dark_blue,linewidth=1.5, label = label)
  custom_lines.append(Line2D([0], [0], color=dark_blue, lw=2, linestyle='-'))
  labels.append(label)

  # Per mode
  mode_colors = {}
  if(isNuBar == False):
    mode_colors = {
        1: "red",
        2: "blue",
        # 21: "orange",
        # 26: "purple",
    }
  else:
    mode_colors = {
        -1: "red",
        -2: "blue",
        # -21: "orange",
        # -26: "purple",
    }

  for m, diffs in diff_by_mode.items():
    if(abs(m) == 1 or abs(m) ==2):

      if len(diffs) == 0:
          continue
      if m == 0: # no 0 mode
        continue

      ax.hist(
          diffs,
          bins=np.arange(-1000, 1000, 10),
          # histtype='step',
          stacked=True,
          linewidth=1.5,
          linestyle="--",
          color=mode_colors.get(m, "gray"),
          label=f"Mode {m}"
      )

  ax.legend()
  ax.set_xlabel(r"$E_{\nu}^{\text{QE}} - E_{\nu}^{\text{true}}$ [MeV]")
  ax.set_ylabel("Counts")
  plt.gca()
  plt.savefig(f"Fig2_plots/Fig2_HK_EnuRecoBias_{plot_name}.pdf")
  # plt.show()

  fin.Close()

_events = -1
plot_Enu_bias(filename="../../Remade_April26/HK/HK_numu_noFSI.flat.root", label = r"no FSI $\nu_{\mu}$", isNuBar = False, nEvents=_events, plot_name="noFSI_numu")
plot_Enu_bias(filename="../../Remade_April26/HK/HK_numubar_noFSI.flat.root", label = r"no FSI $\bar{\nu}_{\mu}$", isNuBar = True, nEvents=_events, plot_name="noFSI_numubar")