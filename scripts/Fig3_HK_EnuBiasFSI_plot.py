from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

def plot_Enu_bias_numu(ax, ax_ratio, filename, isNub, nEvents, nominal=False, counts_nom=None):
  # ---------------------------------
  # Open input file and tree
  # ---------------------------------
  Print(f"Reading: {filename}")
  fin = ROOT.TFile.Open(filename)
  tree = fin.Get("FlatTree_VARS")

  # ---------------------------------
  # Event loop
  # ---------------------------------
  nentries    = tree.GetEntries()
  diff_sel    = []
  Enu_t_sel   = []
  Enu_QE_sel  = []
  nevs = 0
  fScaleFactor = 0

  if(nEvents == -1):
      nevs = nentries
  else:
      nevs = nEvents

  pass_flag = 0
  fail_flag = 0

  for i in range(nevs):

    tree.GetEntry(i)
    isCC0pi = tree.flagCC0pi

    _fscalefactor = tree.fScaleFactor
    if(_fscalefactor > fScaleFactor):
       fScaleFactor = _fscalefactor

    if(i==1):
      Log(f"Scale factor: {fScaleFactor}")

    if(isCC0pi == True):
      pass_flag += 1
      Enu_true = tree.Enu_true*1000
      Enu_QE   = tree.Enu_QE*1000

      diff = Enu_QE - Enu_true
      diff_sel.append(diff)
      Enu_t_sel.append(Enu_true)
      Enu_QE_sel.append(Enu_QE)
    else:
       fail_flag += 1

  Log(f"Flag stats: pass {pass_flag}, fail {fail_flag}")
  Log(f"Flag*fScaleFactor stats: pass {pass_flag*fScaleFactor}, fail {fail_flag*fScaleFactor}")

  diff_sel = np.array(diff_sel)
  Enu_t_sel = np.array(Enu_t_sel)
  Enu_QE_sel = np.array(Enu_QE_sel)

  bin_width = 20
  bins = np.arange(-1000, 1000, step=bin_width)
  weights = fScaleFactor*np.ones_like(diff_sel)/bin_width

  if("noFSI" in filename):  
    color = dark_blue
    label = "noFSI"
  else:
    color = dark_red
    label = "FSI"

  ax.hist(
      diff_sel,
      bins=bins,
      histtype='step',
      weights=weights,
      color=color,
      linewidth=1.8,
      label=label
  )

  custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
  labels.append(label)

  # -------------------------
  # Ratio panel
  # -------------------------
  counts, edges = np.histogram(diff_sel, weights=weights, bins=bins)

  if nominal:
      ax_ratio.hlines(1, bins[0], bins[-1], linestyle='--', color='black')
      fin.Close()
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

  if(isNub == False):
    ax.set_title(r"$\nu_{\mu}$")
  else:
    ax.set_title(r"$\bar{\nu}_{\mu}$")

  fin.Close()
  return counts


fig, (ax, ax_ratio) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
)

_events = -1

counts_nom = plot_Enu_bias_numu(
    ax,
    ax_ratio,
    filename="../../Remade_April26/HK/HK_numubar_noFSI.flat.root",
    isNub=True,
    nEvents=_events,
    nominal=True
)

plot_Enu_bias_numu(
    ax,
    ax_ratio,
    filename="../../Remade_April26/HK/HK_numubar_FSI.flat.root",
    isNub=True,
    nEvents=_events,
    nominal=False,
    counts_nom=counts_nom
)

ax.vlines(x=0, ymin=0, ymax=ax.get_ylim()[1], color='black', linestyles='--')
ax.legend(custom_lines, labels, loc='upper right')

ax.set_ylabel(r"$\text{d}\sigma/\text{d}E_{\nu}^{\text{bias}}$ [cm$^{2}$/nucleon MeV]")

ax_ratio.set_xlabel(r"$E_{\nu}^{\text{QE}} - E_{\nu}^{\text{true}}$ [MeV]")
ax_ratio.set_ylabel("FSI/noFSI")
ax_ratio.set_ylim(0, 2)

plt.savefig("Fig3_plots/Enu_bias_FSIvsNoFSI_numubar.pdf")


# fig, ax = plt.subplots()
# _events = 1000
# plot_Enu_bias_numu(ax, filename="../../Remade_April26/HK/HK_numu_noFSI.flat.root", isNub=False, nEvents=_events)
# ax = plot_Enu_bias_numu(ax, filename="../../Remade_April26/HK/HK_numu_FSI.flat.root", isNub=False, nEvents=_events)
# ax.vlines(x=0, ymin=0, ymax = ax.get_ylim()[1], color='black', linestyles='--')
# ax.legend(custom_lines, labels, loc = 'upper right')
# ax.set_xlabel(r"$E_{\nu}^{\text{QE}} - E_{\nu}^{\text{true}}$ [MeV]")
# ax.set_ylabel(r"$\text{d}\sigma/\text{d}E_{\nu}^{\text{bias}}$ [cm$^{2}$/nucleon MeV]")
# plt.savefig("Fig3_plots/Enu_bias_FSIvsNoFSI_numu.pdf")

# ax.clear()
# custom_lines, labels = [], []
# plot_Enu_bias_numu(ax, filename="../../Remade_April26/HK/HK_numubar_noFSI.flat.root", isNub=True, nEvents=_events)
# ax = plot_Enu_bias_numu(ax, filename="../../Remade_April26/HK/HK_numubar_FSI.flat.root", isNub=True, nEvents=_events)
# ax.vlines(x=0, ymin=0, ymax = ax.get_ylim()[1], color='black', linestyles='--')
# ax.legend(custom_lines, labels, loc = 'upper right')
# ax.set_xlabel(r"$E_{\nu}^{\text{QE}} - E_{\nu}^{\text{true}}$ [MeV]")
# ax.set_ylabel(r"$\text{d}\sigma/\text{d}E_{\nu}^{\text{bias}}$ [cm$^{2}$/nucleon MeV]")
# plt.savefig("Fig3_plots/Enu_bias_FSIvsNoFSI_numubar.pdf")

# plt.show()
