"""Apply OscProb disappearance reweight to the NUISFLAT bias distributions
and compare against the nominal (unoscillated) shape.

Per (sample, observable):
  1. Read Enu_true and the same selection cuts used in fsi_iop_plots.py.
  2. Compute P(nu_alpha -> nu_alpha) with OscProb at the right baseline and a
     constant-density Earth (rho = 2.8 g/cm^3). T2K = 295 km; DUNE = 1300 km.
     T2K-2024-NH best-fit PMNS parameters (override at the top if needed).
  3. Multiply event weight by P_osc.
  4. Compute weighted percentile / mean stats and draw a histogram silhouette
     overlay (nominal solid vs oscillated dashed) per row.

Outputs (next to this script):
  osc_compare_{numu,numubar}.{png,pdf}
  + a textual summary table per sample.
"""
import os
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import ROOT

plt.style.use(["science", "notebook"])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_EVENTS = None        # None = full stats

# Mixing parameters as defined in scripts/FlatTreeMod.py ("PDG-ish")
PMNS = dict(theta12=0.583, theta23=0.857, theta13=0.149, deltaCP=-np.pi / 2,
            dm21=7.41e-5, dm32=2.437e-3)
EARTH_DENSITY_GCC = 2.8

BASELINE_KM = {"hk": 295.0, "dune": 1300.0}

XMIN_FIXED, XMAX_FIXED = -1000.0, 1000.0
HK_NBINS, HK_HIST_RANGE   = 200, (-1000.0, 1000.0)
DUNE_NBINS, DUNE_HIST_RANGE = 400, (-3000.0, 1000.0)
HIST_FILL_ALPHA = 0.22
WHISKER_ALPHA = 0.50
BOX_LW = 1.4
COL_NOM = "#444444"
COL_OSC = "#0072B2"

# ---------------------------------------------------------------------------
# OscProb setup
# ---------------------------------------------------------------------------
ROOT.gSystem.Load("libOscProb.so")
pmns = ROOT.OscProb.PMNS_Fast()
pmns.SetMix(PMNS["theta12"], PMNS["theta23"], PMNS["theta13"], PMNS["deltaCP"])
pmns.SetDeltaMsqrs(PMNS["dm21"], PMNS["dm32"])


def disappearance_grid(baseline_km, is_nubar):
    pmns.SetPath(float(baseline_km), EARTH_DENSITY_GCC)
    pmns.SetIsNuBar(bool(is_nubar))
    grid_E = np.linspace(0.05, 30.0, 6000)
    grid_P = np.array([float(pmns.Prob(1, 1, float(E))) for E in grid_E])
    return grid_E, grid_P


_OSC_CACHE = {}

def osc_factor(enu_GeV, baseline_km, is_nubar):
    key = (round(baseline_km, 4), bool(is_nubar))
    if key not in _OSC_CACHE:
        _OSC_CACHE[key] = disappearance_grid(*key)
    grid_E, grid_P = _OSC_CACHE[key]
    return np.interp(enu_GeV, grid_E, grid_P,
                     left=grid_P[0], right=grid_P[-1])


# ---------------------------------------------------------------------------
# File readers (return enu_GeV alongside x, w, mode)
# ---------------------------------------------------------------------------
def read_hk(path):
    a = uproot.open(path)["FlatTree_VARS"].arrays(
        ["Enu_true", "Enu_QE", "cc", "pdg", "fScaleFactor", "Mode"],
        library="ak", entry_stop=MAX_EVENTS)
    apdg = np.abs(a["pdg"])
    n_chpi = ak.sum(apdg == 211, axis=1)
    n_pi0 = ak.sum(apdg == 111, axis=1)
    sel = (a["cc"] == 1) & (n_chpi == 0) & (n_pi0 == 0)
    bias = (a["Enu_QE"] - a["Enu_true"]) * 1000.0
    return dict(
        x=np.asarray(bias[sel]),
        w=np.asarray(a["fScaleFactor"][sel]),
        mode=np.asarray(a["Mode"][sel]),
        enu=np.asarray(a["Enu_true"][sel]),
    )


def read_dune(path):
    a = uproot.open(path)["FlatTree_VARS"].arrays(
        ["Enu_true", "ELep", "cc", "pdg", "E", "px", "py", "pz",
         "fScaleFactor", "Mode"],
        library="ak", entry_stop=MAX_EVENTS)
    apdg = np.abs(a["pdg"])
    p2 = a["px"] ** 2 + a["py"] ** 2 + a["pz"] ** 2
    m2 = a["E"] ** 2 - p2
    KE = a["E"] - ak.where(m2 > 0, np.sqrt(np.maximum(m2, 0.0)), 0.0)
    is_cc = a["cc"] == 1
    heavy_bar = (apdg > 2300) & (apdg < 3000)
    is_e = apdg == 11
    is_mid = (apdg > 17) & (apdg < 2000)
    is_chpi = apdg == 211
    is_proton = apdg == 2212
    contrib_tpi = ak.where(heavy_bar, a["E"],
                  ak.where(is_e | is_mid, a["E"],
                  ak.where(is_proton, KE, 0.0)))
    contrib_epi = ak.where(heavy_bar, a["E"],
                  ak.where((is_e | is_mid) & ~is_chpi, a["E"],
                  ak.where(is_proton | is_chpi, KE, 0.0)))
    ehad_tpi = ak.sum(contrib_tpi, axis=1)
    ehad_epi = ak.sum(contrib_epi, axis=1)
    bias_tpi = ((a["ELep"] + ehad_tpi) - a["Enu_true"]) * 1000.0
    bias_epi = ((a["ELep"] + ehad_epi) - a["Enu_true"]) * 1000.0
    return dict(
        x_tpi=np.asarray(bias_tpi[is_cc]),
        x_epi=np.asarray(bias_epi[is_cc]),
        w=np.asarray(a["fScaleFactor"][is_cc]),
        mode=np.asarray(a["Mode"][is_cc]),
        enu=np.asarray(a["Enu_true"][is_cc]),
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def weighted_percentile(x, w, q):
    keep = (w > 0) & np.isfinite(x)
    x, w = x[keep], w[keep]
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    p = (cw - 0.5 * ws) / cw[-1]
    return np.interp(np.asarray(q) / 100.0, p, xs)


def stats(x, w):
    return dict(
        median=float(weighted_percentile(x, w, 50)),
        p16=float(weighted_percentile(x, w, 16)),
        p84=float(weighted_percentile(x, w, 84)),
        p5=float(weighted_percentile(x, w, 5)),
        p95=float(weighted_percentile(x, w, 95)),
        p0p5=float(weighted_percentile(x, w, 0.5)),
        p99p5=float(weighted_percentile(x, w, 99.5)),
        mean=float(np.average(x, weights=w)),
    )


# ---------------------------------------------------------------------------
# Drawing (mirrors fsi_iop_plots.py / draw_hybrid_hist)
# ---------------------------------------------------------------------------
def draw_hybrid_hist(ax, y, half, x, w, color,
                     nbins=HK_NBINS, hist_range=HK_HIST_RANGE):
    s = stats(x, w)
    counts, edges = np.histogram(x, bins=nbins, range=hist_range, weights=w)
    if counts.max() > 0:
        counts = counts / counts.max()
    xs = np.empty(2 * len(counts))
    xs[0::2] = edges[:-1]
    xs[1::2] = edges[1:]
    ys = np.repeat(counts, 2)
    poly_x = np.concatenate([xs, xs[::-1]])
    poly_y = np.concatenate([y - half * ys, y + half * ys[::-1]])
    ax.fill(poly_x, poly_y, color=color, alpha=HIST_FILL_ALPHA, linewidth=0,
            zorder=2)
    bh = 0.65 * half
    ax.add_patch(Rectangle((s["p16"], y - bh), s["p84"] - s["p16"], 2 * bh,
                           fill=False, edgecolor=color, linewidth=BOX_LW, zorder=4))
    wlo, whi = s["p5"], s["p95"]
    ax.plot([wlo, s["p16"]], [y, y], color=color, lw=1.0, alpha=WHISKER_ALPHA, zorder=3)
    ax.plot([s["p84"], whi], [y, y], color=color, lw=1.0, alpha=WHISKER_ALPHA, zorder=3)
    ax.plot([wlo] * 2, [y - bh / 3, y + bh / 3], color=color, lw=0.9,
            alpha=WHISKER_ALPHA, zorder=3)
    ax.plot([whi] * 2, [y - bh / 3, y + bh / 3], color=color, lw=0.9,
            alpha=WHISKER_ALPHA, zorder=3)
    ax.plot(s["median"], y, "o", color="white", markersize=5,
            markeredgecolor=color, markeredgewidth=1.4, zorder=6)
    ax.plot(s["mean"], y, "D", color="white", markersize=4.5,
            markeredgecolor=color, markeredgewidth=1.4, zorder=6)
    return s


# ---------------------------------------------------------------------------
# Sample list (uses NuWro SF FSI files; baseline + flavour for OscProb)
# ---------------------------------------------------------------------------
SAMPLES = [
    # (label, kind, baseline_kind, is_nubar, file_path, observable_label, obs_extractor)
    ("HK $E_\\nu^{\\rm QE}$",
     "hk",   "hk",   False, f"{BASE}/HK/HK_numu_FSI.flat.root",
     "hk_qe_numu", lambda d: d["x"]),
    ("DUNE $E_\\nu^{\\rm avail}$ (charged-$\\pi$ KE)",
     "dune", "dune", False, f"{BASE}/DUNE/DUNE_numu_FSI.flat.root",
     "dune_avail_numu", lambda d: d["x_epi"]),
    ("DUNE $E_\\nu^{\\rm had}$ (charged-$\\pi$ full E)",
     "dune", "dune", False, f"{BASE}/DUNE/DUNE_numu_FSI.flat.root",
     "dune_had_numu", lambda d: d["x_tpi"]),
    ("HK $E_\\nu^{\\rm QE}$",
     "hk",   "hk",   True,  f"{BASE}/HK/HK_numubar_FSI.flat.root",
     "hk_qe_numubar", lambda d: d["x"]),
    ("DUNE $E_\\nu^{\\rm avail}$",
     "dune", "dune", True,  f"{BASE}/DUNE/DUNE_numub_FSI.flat.root",
     "dune_avail_numubar", lambda d: d["x_epi"]),
    ("DUNE $E_\\nu^{\\rm had}$",
     "dune", "dune", True,  f"{BASE}/DUNE/DUNE_numub_FSI.flat.root",
     "dune_had_numubar", lambda d: d["x_tpi"]),
]


def make_panel(rows, fname, title):
    n = len(rows)
    sub_pitch = 0.45
    group_pitch = 2 * sub_pitch + 0.6
    half = sub_pitch / 2 * 0.85
    fig, ax = plt.subplots(figsize=(11.0, group_pitch * n + 1.6))
    centres = []
    table_lines = []
    for gi, r in enumerate(rows):
        y0 = gi * group_pitch
        nbins, hrange = (HK_NBINS, HK_HIST_RANGE) if r["kind"] == "hk" \
                        else (DUNE_NBINS, DUNE_HIST_RANGE)
        s_nom = draw_hybrid_hist(ax, y0,                 half, r["x"], r["w_nom"],
                                 COL_NOM, nbins=nbins, hist_range=hrange)
        s_osc = draw_hybrid_hist(ax, y0 + sub_pitch,     half, r["x"], r["w_osc"],
                                 COL_OSC, nbins=nbins, hist_range=hrange)
        centres.append(y0 + 0.5 * sub_pitch)
        table_lines.append((r, s_nom, s_osc))
    for gi in range(1, n):
        ax.axhline(gi * group_pitch - sub_pitch / 2,
                   color="k", lw=0.4, alpha=0.3)
    ax.set_yticks(centres)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_ylim(group_pitch * (n - 1) + sub_pitch + 0.4, -0.4)
    ax.axvline(0, color="gray", ls="--", lw=0.7)
    ax.set_xlim(XMIN_FIXED, XMAX_FIXED)
    ax.set_xlabel(r"$E_{\nu}^{\rm reco} - E_{\nu}^{\rm true}$ [MeV]")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{fname}.png", dpi=180, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    return table_lines


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("=== reading + reweighting ===")
print(f"PMNS  th12={PMNS['theta12']:.3f} th23={PMNS['theta23']:.3f} "
      f"th13={PMNS['theta13']:.3f} dCP={PMNS['deltaCP']:+.3f}")
print(f"      dm21={PMNS['dm21']:.3e} dm32={PMNS['dm32']:.3e}")
print(f"baseline: HK = {BASELINE_KM['hk']} km;  DUNE = {BASELINE_KM['dune']} km;  "
      f"rho = {EARTH_DENSITY_GCC} g/cm^3")
print()

cache = {}
rows_numu, rows_numubar = [], []
for label, kind, base_kind, is_nubar, path, tag, extract in SAMPLES:
    if path not in cache:
        cache[path] = read_hk(path) if kind == "hk" else read_dune(path)
    d = cache[path]
    x = extract(d)
    w_nom = d["w"]
    enu = d["enu"]
    p_osc = osc_factor(enu, BASELINE_KM[base_kind], is_nubar)
    w_osc = w_nom * p_osc
    row = dict(label=label, kind=kind, baseline=BASELINE_KM[base_kind],
               is_nubar=is_nubar, x=x, w_nom=w_nom, w_osc=w_osc,
               p_osc_min=float(p_osc.min()), p_osc_max=float(p_osc.max()),
               sumw_nom=float(np.sum(w_nom)), sumw_osc=float(np.sum(w_osc)))
    print(f"  {tag:<22}  N={len(x):>7}  baseline={row['baseline']:.0f} km  "
          f"is_nubar={is_nubar}  P_osc range=[{row['p_osc_min']:.3f}, "
          f"{row['p_osc_max']:.3f}]  Sum w osc/Sum w nom = "
          f"{row['sumw_osc']/row['sumw_nom']:.3f}")
    if is_nubar:
        rows_numubar.append(row)
    else:
        rows_numu.append(row)

print()
print("=== drawing comparison panels ===")
t_numu = make_panel(rows_numu, "osc_compare_numu",
    rf"NuWro SF — $\nu_\mu$ FHC — nominal vs disappearance reweight  "
    rf"($\delta_{{\rm CP}} = -\pi/2$, NH)")
t_numubar = make_panel(rows_numubar, "osc_compare_numubar",
    rf"NuWro SF — $\bar{{\nu}}_\mu$ RHC — nominal vs disappearance reweight  "
    rf"($\delta_{{\rm CP}} = -\pi/2$, NH)")
print(f"saved {OUT_DIR}/osc_compare_numu.png")
print(f"saved {OUT_DIR}/osc_compare_numubar.png")

# ---------------------------------------------------------------------------
# Numeric table
# ---------------------------------------------------------------------------
print()
print(f"{'sample':<46}  {'med nom':>9} {'med osc':>9} {'Δmed':>7}  "
      f"{'mean nom':>10} {'mean osc':>10} {'Δmean':>7}  "
      f"{'1σ nom':>7} {'1σ osc':>7}  {'90% nom':>8} {'90% osc':>8}")
print("-" * 140)
for tlines, name in ((t_numu, "numu"), (t_numubar, "numubar")):
    for r, s_nom, s_osc in tlines:
        sigma_nom = s_nom["p84"] - s_nom["p16"]
        sigma_osc = s_osc["p84"] - s_osc["p16"]
        w90_nom = s_nom["p95"] - s_nom["p5"]
        w90_osc = s_osc["p95"] - s_osc["p5"]
        sample_str = f"{name}  {r['label']}"
        print(f"{sample_str:<46}  "
              f"{s_nom['median']:>+9.1f} {s_osc['median']:>+9.1f} "
              f"{s_osc['median']-s_nom['median']:>+7.1f}  "
              f"{s_nom['mean']:>+10.1f} {s_osc['mean']:>+10.1f} "
              f"{s_osc['mean']-s_nom['mean']:>+7.1f}  "
              f"{sigma_nom:>7.1f} {sigma_osc:>7.1f}  "
              f"{w90_nom:>8.1f} {w90_osc:>8.1f}")
