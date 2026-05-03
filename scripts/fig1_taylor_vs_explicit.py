"""Compare explicit per-event ±5 MeV shift vs analytical Taylor approximation
for the Fig1_dm32-style HK Enu_QE oscillated histogram. The output is two
panels:
  - top: nominal (PMNS-default) histogram in MeV
  - bottom: ratio H(E ± 5 MeV) / H(E) computed both ways:
            * thin coloured = explicit per-event shift (noisy)
            * thick black/grey = analytical exp(-Δ d ln H / dE) (smooth)
"""
from FlatTreeMod import *
ROOT.gROOT.SetBatch(True)

filename = "../../Remade_April26/HK/HK_numu_FSI.flat.root"
arr = load_arrays(filename, max_events=1_000_000)
flag = is_cc0pi_arr(arr, vertex=False)
Enu_t  = np.asarray(arr['Enu_true'])[flag] * 1000.0   # MeV
Enu_QE = np.asarray(arr['Enu_QE'])  [flag] * 1000.0

# Default-PMNS oscillation weight per event from Enu_true.
prob_default = np.array([pmns.Prob(1, 1, E/1000.0, L) for E in Enu_t])  # νμ → νμ

bin_width = 20.0
bins = np.arange(0, 2000, step=bin_width)
centers = 0.5 * (bins[:-1] + bins[1:])

# Unshifted weighted histogram (the spectrum).
counts_nom, _ = np.histogram(Enu_QE, bins=bins, weights=prob_default)
# Explicit per-event shifted histograms.
counts_p5, _  = np.histogram(Enu_QE + 5, bins=bins, weights=prob_default)
counts_m5, _  = np.histogram(Enu_QE - 5, bins=bins, weights=prob_default)

# Explicit ratios (noisy).
ratio_p5_explicit = counts_p5 / np.maximum(counts_nom, 1e-30)
ratio_m5_explicit = counts_m5 / np.maximum(counts_nom, 1e-30)

# Analytical Taylor ratios (smooth).
ratio_p5_taylor = smooth_shift_ratio(counts_nom, bins,  5.0)
ratio_m5_taylor = smooth_shift_ratio(counts_nom, bins, -5.0)

fig = plt.figure(figsize=(8, 6.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1.6], hspace=0.07)
ax = fig.add_subplot(gs[0])
ax_r = fig.add_subplot(gs[1], sharex=ax)
plt.setp(ax.get_xticklabels(), visible=False)

ax.step(centers, counts_nom, where='mid', color='k', lw=1.6, label='nominal')
ax.set_ylabel("Oscillated events / 20 MeV")
ax.legend(loc='upper right')
ax.set_xlim(0, 1200)
ax.grid(alpha=0.25)
ax.set_title("HK $\\nu_\\mu$, ±5 MeV shift: explicit vs Taylor")

# Explicit (thin, coloured)
ax_r.step(centers, ratio_p5_explicit, where='mid', color=dark_blue, lw=1.0,
          alpha=0.7, label='+5 MeV explicit')
ax_r.step(centers, ratio_m5_explicit, where='mid', color=dark_red, lw=1.0,
          alpha=0.7, label='-5 MeV explicit')
# Taylor (thick, dashed)
ax_r.step(centers, ratio_p5_taylor, where='mid', color=dark_blue, lw=2.4,
          ls='--', label='+5 MeV Taylor')
ax_r.step(centers, ratio_m5_taylor, where='mid', color=dark_red, lw=2.4,
          ls='--', label='-5 MeV Taylor')
ax_r.axhline(1.0, color='0.4', ls=':', lw=0.8)

ax_r.set_xlabel(r"$E_\nu^{\rm QE}$ [MeV]")
ax_r.set_ylabel("ratio shifted / nominal")
ax_r.set_xlim(0, 1200)
ax_r.set_ylim(0.85, 1.15)
ax_r.legend(loc='lower left', ncol=2, fontsize=9)
ax_r.grid(alpha=0.25)

plt.savefig("Fig1_taylor_vs_explicit.png", dpi=180)
print("saved Fig1_taylor_vs_explicit.png")
