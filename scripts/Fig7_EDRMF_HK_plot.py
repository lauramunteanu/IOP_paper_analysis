from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

def plot_Enu_bias_numu(ax, ax_ratio, filename, label, nEvents, nominal=False, counts_nom=None):
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  diff_sel = diff_enu_qe_arr(arr, vertex=False)
  fScaleFactor = float(np.max(arr['fScaleFactor']))

  bin_width = 20
  bins = np.arange(-1000, 1000, step=bin_width)
  # Use the canonical dσ/dE weight scaling so the y-axis autoscales sensibly
  # (raw fScaleFactor is ~1e-45; without DSIGMA_DE_SCALE the bars vanish at y~0).
  weights = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(diff_sel)

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

  return counts



_events = -1
custom_lines, labels = [], []

fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))

counts_rpwia = plot_Enu_bias_numu(
    ax=ax,
    ax_ratio=ax_ratio,
    filename="/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/HK/NEUT_HK_RPWIA_numu.flat.root",
    label="RPWIA",
    nEvents=_events,
    nominal=True
)

plot_Enu_bias_numu(
    ax=ax,
    ax_ratio=ax_ratio,
    filename="/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/HK/NEUT_HK_EDRMF_numu.flat.root",
    label="ED-RMF",
    nEvents=_events,
    nominal=False,
    counts_nom=counts_rpwia
)

ax.vlines(x=0, ymin=0, ymax=ax.get_ylim()[1], color='black', linestyles='--')
ax.legend(custom_lines, labels, loc='best')

ax.set_ylabel(DSIGMA_DE_LABEL)

ax_ratio.set_xlabel(r"$E_{\nu}^{\text{QE}} - E_{\nu}^{\text{true}}$ [MeV]")
ax_ratio.set_ylabel("ED-RMF/RPWIA")
ax_ratio.set_ylim(0.5, 1.5)

plt.savefig("Fig7_plots/Fig7_HK_EnuRecoFSIBias_EDRMF_numu.pdf")
plt.close(fig)