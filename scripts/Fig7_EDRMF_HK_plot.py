from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

def plot_Enu_bias_numu(ax, ax_ratio, filename, label, nEvents, nominal=False, counts_nom=None):
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
  Enu_t_sel   = []
  Enu_QE_sel  = []
  nevs = 0
  fScaleFactor = 0

  if(nEvents == -1):
      nevs = nentries
  else:
      nevs = nEvents

  for i in range(nevs):

    tree.GetEntry(i)

    Enu_true = tree.Enu_true*1000
    Enu_QE   = tree.Enu_QE*1000
    isCC0pi  = tree.flagCC0pi

    _fscalefactor = tree.fScaleFactor
    if(_fscalefactor > fScaleFactor):
       fScaleFactor = _fscalefactor

    if(isCC0pi == True):
      diff = Enu_QE - Enu_true

      diff_sel.append(diff)
      Enu_t_sel.append(Enu_true)
      Enu_QE_sel.append(Enu_QE)

  diff_sel  = np.array(diff_sel)
  Enu_t_sel = np.array(Enu_t_sel)
  Enu_QE_sel = np.array(Enu_QE_sel)

  bin_width = 20
  bins = np.arange(-1000, 1000, step=bin_width)
  weights = fScaleFactor*np.ones_like(diff_sel)/bin_width

  # ---------------------------------
  # Style
  # ---------------------------------
  if(label == "ED-RMF"):
    color = dark_red
  elif(label == "RPWIA"):
    color = dark_blue
  else:
    color = "black"

  # ---------------------------------
  # Main histogram
  # ---------------------------------
  ax.hist(
      diff_sel,
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
  counts, edges = np.histogram(diff_sel, weights=weights, bins=bins)

  if(nominal == True):
      ax_ratio.hlines(1, bins[0], bins[-1], linestyle='--', color=dark_blue)
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
      

  ax.set_title(r"HK $\nu_{\mu}$")

  fin.Close()
  return counts



_events = -1
custom_lines, labels = [], []

fig, (ax, ax_ratio) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
)

counts_rpwia = plot_Enu_bias_numu(
    ax=ax,
    ax_ratio=ax_ratio,
    filename="../../FSI/NEUT_HK_RPWIA_numu.flat.root",
    label="RPWIA",
    nEvents=_events,
    nominal=True
)

plot_Enu_bias_numu(
    ax=ax,
    ax_ratio=ax_ratio,
    filename="../../FSI/NEUT_HK_EDRMF_numu.flat.root",
    label="ED-RMF",
    nEvents=_events,
    nominal=False,
    counts_nom=counts_rpwia
)

ax.vlines(x=0, ymin=0, ymax=ax.get_ylim()[1], color='black', linestyles='--')
ax.legend(custom_lines, labels, loc='best', fontsize=15)

ax.set_ylabel(r"$\text{d}\sigma/\text{d}E_{\nu}^{\text{bias}}$ [cm$^{2}$/nucleon MeV]")

ax_ratio.set_xlabel(r"$E_{\nu}^{\text{QE}} - E_{\nu}^{\text{true}}$ [MeV]")
ax_ratio.set_ylabel("ED-RMF/RPWIA")
ax_ratio.set_ylim(0, 2)

plt.savefig("Fig7_plots/Fig7_HK_EnuRecoFSIBias_EDRMF_numu.pdf")