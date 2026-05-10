from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

def plot_Enu_bias_numu(ax, ax_ratio, filename, isNub, nEvents, nominal=False, counts_nom=None):
  # noFSI vs FSI no longer toggled via vertex flag — caller passes a noFSI
  # file when they want the noFSI line (nominal=True) and an FSI file for the
  # FSI overlay. Vertex stack of an FSI file would be biased by NuWro binding-
  # energy bookkeeping at cascade exit, so we read the post-FSI stack of the
  # appropriate file instead.
  arr = load_arrays(filename, max_events=(None if nEvents == -1 else nEvents))
  diff_sel = diff_enu_qe_arr(arr, vertex=False)
  fScaleFactor = float(np.max(arr['fScaleFactor']))
  Log(f"  pass {len(diff_sel)} / total {len(arr['Enu_true'])}  (file={filename})")

  bin_width = 20
  bins = np.arange(-1000, 1000, step=bin_width)
  weights = make_weights_dxsec(arr, bin_width, fScaleFactor) * np.ones_like(diff_sel)

  if nominal:
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

  return counts


fig, (ax, ax_ratio) = make_fig_ratio('single_ratio', height_ratios=(3, 1))

_events = -1
fname_FSI   = "../../Remade_April26/nuwro_25031/HK/HK_numubar_FSI.flat.root"
fname_noFSI = noFSI_path(fname_FSI)

counts_nom = plot_Enu_bias_numu(
    ax, ax_ratio, filename=fname_noFSI, isNub=True,
    nEvents=_events, nominal=True,
)
counts_fsi = plot_Enu_bias_numu(
    ax, ax_ratio, filename=fname_FSI, isNub=True,
    nEvents=_events, nominal=False, counts_nom=counts_nom,
)

ax.axvline(0, color='black', linestyle='--', lw=0.7)
ax.legend(custom_lines, labels, loc='upper right')
ax.set_xlim(-1000, 1000)
# Force ylim from the actual histogram peak — matplotlib autoscale can fail
# at the dσ/dE ~ 1e-42 cm²/nucleon/MeV range.
peak = max(float(counts_nom.max()), float(counts_fsi.max()))
ax.set_ylim(0, peak * 1.15)
ax.set_ylabel(DSIGMA_DE_LABEL)

ax_ratio.set_xlabel(r"$E_{\nu}^{\rm QE} - E_{\nu}^{\rm true}$ [MeV]")
ax_ratio.set_ylabel("FSI/noFSI")
ax_ratio.set_xlim(-1000, 1000)
# y-range left to matplotlib auto-scale

plt.savefig("Fig3_plots/Enu_bias_FSIvsNoFSI_numubar.pdf")
plt.close(fig)


# fig, ax = plt.subplots()
# _events = 1000
# plot_Enu_bias_numu(ax, filename="../../Remade_April26/nuwro_25031/HK/HK_numu_noFSI.flat.root", isNub=False, nEvents=_events)
# ax = plot_Enu_bias_numu(ax, filename="../../Remade_April26/nuwro_25031/HK/HK_numu_FSI.flat.root", isNub=False, nEvents=_events)
# ax.vlines(x=0, ymin=0, ymax = ax.get_ylim()[1], color='black', linestyles='--')
# ax.legend(custom_lines, labels, loc = 'upper right')
# ax.set_xlabel(r"$E_{\nu}^{\text{QE}} - E_{\nu}^{\text{true}}$ [MeV]")
# ax.set_ylabel(r"$\text{d}\sigma/\text{d}E_{\nu}^{\text{bias}}$ [cm$^{2}$/nucleon MeV]")
# plt.savefig("Fig3_plots/Enu_bias_FSIvsNoFSI_numu.pdf")

# ax.clear()
# custom_lines, labels = [], []
# plot_Enu_bias_numu(ax, filename="../../Remade_April26/nuwro_25031/HK/HK_numubar_noFSI.flat.root", isNub=True, nEvents=_events)
# ax = plot_Enu_bias_numu(ax, filename="../../Remade_April26/nuwro_25031/HK/HK_numubar_FSI.flat.root", isNub=True, nEvents=_events)
# ax.vlines(x=0, ymin=0, ymax = ax.get_ylim()[1], color='black', linestyles='--')
# ax.legend(custom_lines, labels, loc = 'upper right')
# ax.set_xlabel(r"$E_{\nu}^{\text{QE}} - E_{\nu}^{\text{true}}$ [MeV]")
# ax.set_ylabel(r"$\text{d}\sigma/\text{d}E_{\nu}^{\text{bias}}$ [cm$^{2}$/nucleon MeV]")
# plt.savefig("Fig3_plots/Enu_bias_FSIvsNoFSI_numubar.pdf")

# plt.show()
