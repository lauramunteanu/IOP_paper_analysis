from FlatTreeMod import *

def plot_Enu_bias_numu(ax, ax_ratio, filename, label, nEvents, withPion,
                       nominal=False, counts_nom=None):
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  bias_wo_list, bias_with_list, _ = enu_had_arr(arr, vertex=False)
  fScaleFactor = float(np.max(arr['fScaleFactor']))

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

  return counts

_events = 100000

# ============================================================
# Plot 1: without pion mass correction
# ============================================================
custom_lines, labels = [], []

fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))

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

ax.legend(custom_lines, labels, loc='best')
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

fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))

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

ax.legend(custom_lines, labels, loc='best')
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



