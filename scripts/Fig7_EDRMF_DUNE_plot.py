from FlatTreeMod import *

def plot_Enu_bias_numu(ax, ax_ratio, filename, label, nEvents, withPion,
                       nominal=False, counts_nom=None):

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

      _fscalefactor = tree.fScaleFactor
      if(_fscalefactor > fScaleFactor):
         fScaleFactor = _fscalefactor

      E  = tree.E
      px = tree.px
      py = tree.py
      pz = tree.pz
      pdg = tree.pdg

      enuhad_wo   = ELep
      enuhad_with = ELep

      for j in range(nfsp):

          apdg = abs(int(pdg[j]))
          Ej   = float(E[j])
          pxj  = float(px[j])
          pyj  = float(py[j])
          pzj  = float(pz[j])

          p2 = pxj*pxj + pyj*pyj + pzj*pzj

          if apdg > 3000:
              continue

          if apdg > 2300 and apdg < 3000:
              enuhad_wo   += Ej
              enuhad_with += Ej
              continue

          # Definition 1: no pion mass subtraction
          if (apdg == 11 or (apdg > 17 and apdg < 2000)) and (apdg != 211):
              enuhad_wo += Ej

          elif apdg == 2212 or apdg == 211:
              mass2 = Ej*Ej - p2
              if mass2 > 0:
                  mass = np.sqrt(mass2)
                  enuhad_wo += (Ej - mass)

          # Definition 2: with pion masses
          if (apdg == 11 or (apdg > 17 and apdg < 2000)):
              enuhad_with += Ej

          elif apdg == 2212:
              mass2 = Ej*Ej - p2
              if mass2 > 0:
                  mass = np.sqrt(mass2)
                  enuhad_with += (Ej - mass)

      bias_wo   = enuhad_wo   - Enu_true
      bias_with = enuhad_with - Enu_true

      bias_wo_list.append(bias_wo)
      bias_with_list.append(bias_with)

  bias_wo_list = np.array(bias_wo_list)
  bias_with_list = np.array(bias_with_list)

  bin_width = 0.05
  bins = np.arange(-0.7, 0 + bin_width, step=bin_width)

  weights_with = fScaleFactor*np.ones_like(bias_with_list)/bin_width
  weights_wo   = fScaleFactor*np.ones_like(bias_wo_list)/bin_width

  # ---------------------------------
  # Choose with/without pion correction
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
  # Style
  # ---------------------------------
  if(label == "ED-RMF"):
      color = dark_red
  elif(label == "RPWIA"):
      color = dark_blue
  else:
      color = "black"

  plot_label = f"{label} {pion_label}"

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
      label=plot_label
  )

  custom_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='-'))
  labels.append(plot_label)

  # ---------------------------------
  # Ratio histogram
  # ---------------------------------
  counts, edges = np.histogram(bias, weights=weights, bins=bins)

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

  fin.Close()
  return counts

_events = 100000

# ============================================================
# Plot 1: without pion mass correction
# ============================================================
custom_lines, labels = [], []

fig, (ax, ax_ratio) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
)

_withPion = False

counts_rpwia_wo = plot_Enu_bias_numu(
    ax=ax,
    ax_ratio=ax_ratio,
    filename="../../FSI/RPWIA_1M_Cas_numu_Ar40.flat.root",
    label="RPWIA",
    nEvents=_events,
    withPion=_withPion,
    nominal=True
)

plot_Enu_bias_numu(
    ax=ax,
    ax_ratio=ax_ratio,
    filename="../../FSI/NEUT_Ar40_EDRMF_numu.flat.root",
    label="ED-RMF",
    nEvents=_events,
    withPion=_withPion,
    nominal=False,
    counts_nom=counts_rpwia_wo
)

ax.legend(custom_lines, labels, loc='best', fontsize=15)
ax.set_title(r"$\nu_{\mu}$, w/o pion mass")
ax.set_ylabel(
    r"$\text{d}\sigma/\text{d}E_{\nu}^{\text{bias}}$ "
    r"[cm$^{2}$/nucleon GeV]"
)

ax_ratio.set_xlabel(r"$E_{\nu}^{\text{avail}} - E_{\nu}^{\text{true}}$ [GeV]")
ax_ratio.set_ylabel("ED-RMF/RPWIA")
ax_ratio.set_ylim(0, 2)

plt.savefig("Fig7_plots/Fig7_Ar40_EnuRecoBias_EDRMF_RPWIA_WithoutPion_ratio.pdf")
plt.close(fig)


# ============================================================
# Plot 2: with pion mass correction
# ============================================================
custom_lines, labels = [], []

fig, (ax, ax_ratio) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
)

_withPion = True

counts_rpwia_with = plot_Enu_bias_numu(
    ax=ax,
    ax_ratio=ax_ratio,
    filename="../../FSI/RPWIA_1M_Cas_numu_Ar40.flat.root",
    label="RPWIA",
    nEvents=_events,
    withPion=_withPion,
    nominal=True
)

plot_Enu_bias_numu(
    ax=ax,
    ax_ratio=ax_ratio,
    filename="../../FSI/NEUT_Ar40_EDRMF_numu.flat.root",
    label="ED-RMF",
    nEvents=_events,
    withPion=_withPion,
    nominal=False,
    counts_nom=counts_rpwia_with
)

ax.legend(custom_lines, labels, loc='best', fontsize=15)
ax.set_title(r"$\nu_{\mu}$, w/ pion mass")
ax.set_ylabel(
    r"$\text{d}\sigma/\text{d}E_{\nu}^{\text{bias}}$ "
    r"[cm$^{2}$/nucleon GeV]"
)

ax_ratio.set_xlabel(r"$E_{\nu}^{\text{had}} - E_{\nu}^{\text{true}}$ [GeV]")
ax_ratio.set_ylabel("ED-RMF/RPWIA")
ax_ratio.set_ylim(0, 2)

plt.savefig("Fig7_plots/Fig7_Ar40_EnuRecoBias_EDRMF_RPWIA_WithPion_ratio.pdf")
plt.close(fig)



