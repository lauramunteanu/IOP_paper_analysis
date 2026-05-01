"""FSI / pi-abs / cascade box-and-whisker comparison plots for the IOP paper.

Six output figures (PNG + PDF) are written next to this script:
  pi_abs_{numu,numubar}.{png,pdf}
      NuWro SF FSI files only.
      Per row (HK QE, DUNE Tpi, DUNE Epi), three overlaid variants:
        - nominal (dark grey, solid)
        - pi_abs +40% (Wong vermilion, dashed)
        - pi_abs -40% (Wong blue, dotted)
      Implementation: weight events with |Mode|>2 by 1.4 / 0.6 respectively.

  fsi_compare_{numu,numubar}.{png,pdf}
      NuWro SF, FSI files vs noFSI files.
      Per row (HK QE, DUNE Tpi, DUNE Epi), two overlaid variants:
        - with FSI (dark grey, solid)
        - no FSI  (Wong vermilion, dashed)

  cascade_compare_{numu,numubar}.{png,pdf}
      GENIE NUISFLAT, all four tunes (10a/b/c/d).
      Three row-groups (HK QE, DUNE Tpi, DUNE Epi), 4 cascades per group.
      Cascade colours: 10a #0072B2 / 10b #D55E00 / 10c #009E73 / 10d #CC79A7.

Conventions for box and whiskers (all figures):
  box                       = 1 sigma (16-84%)
  inner whisker (solid)     = 2 sigma (2.5-97.5%)
  outer whisker (faint)     = 99% (0.5-99.5%)
  median tick (solid)       = central percentile
  mean tick (dotted)        = weighted arithmetic mean
  weights                   = fScaleFactor (* pi-abs reweight where applicable)

Set MAX_EVENTS=None for full statistics; small value gives fast layout iteration.
"""
import os
import pickle
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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
OUT_DIR = "/eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw"
MAX_EVENTS = None  # full stats (cache hit after first run)

PI_UP_FACTOR = 1.4
PI_DOWN_FACTOR = 0.6
OUTER_ALPHA = 0.35

XMIN_FIXED, XMAX_FIXED = -1000.0, 1000.0  # visible x-axis [MeV]
# Per-experiment histogram binning matching Jake's Fig3 / Fig4 conventions.
HK_NBINS, HK_HIST_RANGE   = 200, (-1000.0, 1000.0)   # 10 MeV bins, [-1, +1] GeV
DUNE_NBINS, DUNE_HIST_RANGE = 100, (-3000.0, 1000.0) # 40 MeV bins, [-3, +1] GeV
HIST_FILL_ALPHA = 0.22    # translucent histogram silhouette
WHISKER_ALPHA = 0.50      # faint whiskers
BOX_LW = 1.4              # 1-sigma outline width


def hist_params_for(obs_key):
    """Pick (nbins, range) by experiment to match Jake's Fig3 / Fig4."""
    if obs_key == "hk_qe":
        return HK_NBINS, HK_HIST_RANGE
    return DUNE_NBINS, DUNE_HIST_RANGE

# Wong CB-friendly palette
COL_GREY = "#444444"
COL_VERMILION = "#D55E00"
COL_BLUE = "#0072B2"
COL_GREEN = "#009E73"
COL_PURPLE = "#CC79A7"
COL_SKYBLUE = "#56B4E9"

CASCADE_COLORS = {"10a": COL_BLUE, "10b": COL_VERMILION,
                  "10c": COL_GREEN, "10d": COL_PURPLE}

NUWRO_HK = {
    "numu": {"FSI":   f"{BASE}/HK/HK_numu_FSI.flat.root",
             "noFSI": f"{BASE}/HK/HK_numu_noFSI.flat.root"},
    "numubar": {"FSI":   f"{BASE}/HK/HK_numubar_FSI.flat.root",
                "noFSI": f"{BASE}/HK/HK_numubar_noFSI.flat.root"},
}
NUWRO_DUNE = {
    "numu": {"FSI":   f"{BASE}/DUNE/DUNE_numu_FSI.flat.root",
             "noFSI": f"{BASE}/DUNE/DUNE_numu_noFSI.flat.root"},
    "numubar": {"FSI":   f"{BASE}/DUNE/DUNE_numub_FSI.flat.root",
                "noFSI": f"{BASE}/DUNE/DUNE_numub_noFSI.flat.root"},
}
GENIE_HK = {
    ("numu", t):    f"{BASE}/HK/T2KSK_unosc_FHC_numu_H2O_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
}
GENIE_HK.update({
    ("numubar", t): f"{BASE}/HK/T2KSK_unosc_RHC_numubar_H2O_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
})
GENIE_DUNE = {
    ("numu", t):    f"{BASE}/DUNE/DUNEFD_unosc_FHC_numu_Ar40_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
}
GENIE_DUNE.update({
    ("numubar", t): f"{BASE}/DUNE/DUNEFD_unosc_RHC_numubar_Ar40_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
})

OBS_LABELS = {
    "hk_qe":   r"HK  $E_{\nu}^{\mathrm{QE}}$",
    # dune_epi = with_pi_corr=False = charged pi KE only  ->  E_avail
    "dune_epi": r"DUNE  $E_{\nu}^{\mathrm{avail}}$",
    # dune_tpi = with_pi_corr=True = charged pi full E  ->  E_had
    "dune_tpi": r"DUNE  $E_{\nu}^{\mathrm{had}}$",
}

NU_LABELS = {"numu": r"$\nu_{\mu}$ FHC", "numubar": r"$\bar{\nu}_{\mu}$ RHC"}


# ---------------------------------------------------------------------------
# Stats helpers
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
    if len(x) == 0:
        nan = float("nan")
        return dict(median=nan, p16=nan, p84=nan, p2p5=nan, p97p5=nan,
                    p5=nan, p95=nan, p0p5=nan, p99p5=nan, mean=nan)
    return dict(
        median=float(weighted_percentile(x, w, 50)),
        p16=float(weighted_percentile(x, w, 16)),
        p84=float(weighted_percentile(x, w, 84)),
        p2p5=float(weighted_percentile(x, w, 2.5)),
        p97p5=float(weighted_percentile(x, w, 97.5)),
        p5=float(weighted_percentile(x, w, 5)),
        p95=float(weighted_percentile(x, w, 95)),
        p0p5=float(weighted_percentile(x, w, 0.5)),
        p99p5=float(weighted_percentile(x, w, 99.5)),
        mean=float(np.average(x, weights=w)),
    )


# ---------------------------------------------------------------------------
# File readers (one read per (file, observable). Returns (x [MeV], w_nom, mode))
# ---------------------------------------------------------------------------
def read_hk(path):
    """HK QE bias [MeV] on derived CC0pi (cc=1, no charged-pi, no pi0)."""
    a = uproot.open(path)["FlatTree_VARS"].arrays(
        ["Enu_true", "Enu_QE", "cc", "pdg", "fScaleFactor", "Mode"],
        library="ak", entry_stop=MAX_EVENTS)
    apdg = np.abs(a["pdg"])
    n_chpi = ak.sum(apdg == 211, axis=1)
    n_pi0 = ak.sum(apdg == 111, axis=1)
    sel = (a["cc"] == 1) & (n_chpi == 0) & (n_pi0 == 0)
    bias = (a["Enu_QE"] - a["Enu_true"]) * 1000.0
    x = np.asarray(bias[sel])
    w = np.asarray(a["fScaleFactor"][sel])
    mode = np.asarray(a["Mode"][sel])
    return x, w, mode


def read_dune(path):
    """DUNE calorimetric bias [MeV], two definitions, on CC inclusive.
    Returns ((x_tpi, x_epi), w_nom, mode).
    """
    a = uproot.open(path)["FlatTree_VARS"].arrays(
        ["Enu_true", "ELep", "cc", "pdg", "E", "px", "py", "pz",
         "fScaleFactor", "Mode"],
        library="ak", entry_stop=MAX_EVENTS)
    apdg = np.abs(a["pdg"])
    p2 = a["px"] * a["px"] + a["py"] * a["py"] + a["pz"] * a["pz"]
    m2 = a["E"] * a["E"] - p2
    KE = a["E"] - ak.where(m2 > 0, np.sqrt(np.maximum(m2, 0.0)), 0.0)

    is_cc = a["cc"] == 1
    heavy_bar = (apdg > 2300) & (apdg < 3000)
    is_e = apdg == 11
    is_mid = (apdg > 17) & (apdg < 2000)
    is_chpi = apdg == 211
    is_proton = apdg == 2212

    # Tpi  = with pi corr (charged pi full E)
    contrib_tpi = ak.where(heavy_bar, a["E"],
                  ak.where(is_e | is_mid, a["E"],
                  ak.where(is_proton, KE, 0.0)))
    # Epi  = without pi corr (charged pi KE only)
    contrib_epi = ak.where(heavy_bar, a["E"],
                  ak.where((is_e | is_mid) & ~is_chpi, a["E"],
                  ak.where(is_proton | is_chpi, KE, 0.0)))

    ehad_tpi = ak.sum(contrib_tpi, axis=1)
    ehad_epi = ak.sum(contrib_epi, axis=1)
    bias_tpi = ((a["ELep"] + ehad_tpi) - a["Enu_true"]) * 1000.0
    bias_epi = ((a["ELep"] + ehad_epi) - a["Enu_true"]) * 1000.0
    sel = is_cc
    x_tpi = np.asarray(bias_tpi[sel])
    x_epi = np.asarray(bias_epi[sel])
    w = np.asarray(a["fScaleFactor"][sel])
    mode = np.asarray(a["Mode"][sel])
    return x_tpi, x_epi, w, mode


def pi_reweight(w, mode, factor):
    boost = np.where(np.abs(mode) > 2, factor, 1.0)
    return w * boost


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def draw_hybrid_hist(ax, y, half, x, w, color,
                     nbins=HK_NBINS, hist_range=HK_HIST_RANGE):
    """Histogram silhouette (no smoothing) + 1-sigma box outline +
    faint 90% whiskers + median/mean markers, all in `color`.

    `hist_range` is the *binning* range (passed to np.histogram); the visual
    x-axis is set separately by setup_axes().
    """
    s = stats(x, w)
    counts, edges = np.histogram(x, bins=nbins, range=hist_range, weights=w)
    if counts.max() > 0:
        counts = counts / counts.max()
    # step polygon mirrored above and below the row centre
    xs_step = np.empty(2 * len(counts))
    xs_step[0::2] = edges[:-1]
    xs_step[1::2] = edges[1:]
    ys_step = np.repeat(counts, 2)
    poly_x = np.concatenate([xs_step, xs_step[::-1]])
    poly_y = np.concatenate([y - half * ys_step, y + half * ys_step[::-1]])
    ax.fill(poly_x, poly_y, color=color, alpha=HIST_FILL_ALPHA, linewidth=0,
            zorder=2)

    # 1-sigma box outline (slightly shorter than the silhouette)
    bh = 0.65 * half
    ax.add_patch(Rectangle((s["p16"], y - bh), s["p84"] - s["p16"], 2 * bh,
                           fill=False, edgecolor=color, linewidth=BOX_LW, zorder=4))

    # 90% whiskers (faint)
    wlo, whi = s["p5"], s["p95"]
    ax.plot([wlo, s["p16"]], [y, y], color=color, lw=1.0, alpha=WHISKER_ALPHA, zorder=3)
    ax.plot([s["p84"], whi], [y, y], color=color, lw=1.0, alpha=WHISKER_ALPHA, zorder=3)
    ax.plot([wlo] * 2, [y - bh / 3, y + bh / 3], color=color, lw=0.9,
            alpha=WHISKER_ALPHA, zorder=3)
    ax.plot([whi] * 2, [y - bh / 3, y + bh / 3], color=color, lw=0.9,
            alpha=WHISKER_ALPHA, zorder=3)

    # markers
    ax.plot(s["median"], y, "o", color="white", markersize=5,
            markeredgecolor=color, markeredgewidth=1.4, zorder=6)
    ax.plot(s["mean"], y, "D", color="white", markersize=4.5,
            markeredgecolor=color, markeredgewidth=1.4, zorder=6)


def quantity_legend_handles():
    return [
        Line2D([0], [0], color="k", ls="-", lw=2.0, label="median"),
        Line2D([0], [0], color="k", ls=":", lw=2.0, label="mean"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="k",
                  label=r"$1\sigma$ (16-84)"),
        Line2D([0], [0], color="k", ls="-", lw=1.4, label=r"$2\sigma$ (2.5-97.5)"),
        Line2D([0], [0], color="k", ls="-", lw=1.0, alpha=OUTER_ALPHA,
               label=r"99\% (0.5-99.5)"),
    ]


def save_standalone_legend(handles, ncol, fname, fig_w=10.0, fig_h=0.6,
                           fontsize=10):
    """Render a legend by itself into its own image file."""
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.legend(handles=handles, loc="center", ncol=ncol, fontsize=fontsize,
               frameon=True, framealpha=0.9)
    plt.savefig(f"{OUT_DIR}/{fname}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{fname}.png")


def setup_axes(ax, n_groups, xlabel, xmin, xmax, group_labels, group_centers):
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(group_centers)
    ax.set_yticklabels(group_labels)
    ax.axvline(0, color="gray", ls="--", lw=0.7)
    ax.set_xlabel(xlabel)
    # no background grid (gradient bands carry the visual)


# ---------------------------------------------------------------------------
# Cache: in-process dict + on-disk pickle cache
# ---------------------------------------------------------------------------
_CACHE = {}
CACHE_DIR = f"{OUT_DIR}/_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _disk_cache(reader, path, kind):
    base = os.path.basename(path)
    tag = "all" if MAX_EVENTS is None else f"n{MAX_EVENTS}"
    cache_path = f"{CACHE_DIR}/{kind}_{base}_{tag}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"  reading {kind}: {base}")
    result = reader(path)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def get_hk(path):
    if ("hk", path) not in _CACHE:
        _CACHE[("hk", path)] = _disk_cache(read_hk, path, "hk")
    return _CACHE[("hk", path)]

def get_dune(path):
    if ("dune", path) not in _CACHE:
        _CACHE[("dune", path)] = _disk_cache(read_dune, path, "dune")
    return _CACHE[("dune", path)]


# ---------------------------------------------------------------------------
# Figure 1 group: pi-abs +/- 40%, NuWro SF FSI files only
# ---------------------------------------------------------------------------
def make_pi_abs_figure(flavour):
    rows_meta = [
        ("hk_qe",    NUWRO_HK[flavour]["FSI"],   "hk"),
        ("dune_epi", NUWRO_DUNE[flavour]["FSI"], "dune_epi"),  # E_avail
        ("dune_tpi", NUWRO_DUNE[flavour]["FSI"], "dune_tpi"),  # E_had
    ]
    triples = []
    for obs_key, path, kind in rows_meta:
        if kind == "hk":
            x, w, mode = get_hk(path)
        else:
            x_tpi, x_epi, w, mode = get_dune(path)
            x = x_tpi if kind == "dune_tpi" else x_epi
        triples.append((obs_key, x, w, mode))

    n_groups = len(triples)
    rows_per_group = 3
    sub_pitch = 0.45
    group_pitch = rows_per_group * sub_pitch + 0.6
    half = sub_pitch / 2 * 0.85
    fig, ax = plt.subplots(figsize=(11.0, group_pitch * n_groups + 1.6))

    group_centers = []
    for gi, (obs_key, x, w, mode) in enumerate(triples):
        y0 = gi * group_pitch
        nbins, hrange = hist_params_for(obs_key)
        # variant order top -> bottom: +40, nom, -40
        draw_hybrid_hist(ax, y0 + 0 * sub_pitch, half, x,
                         pi_reweight(w, mode, PI_UP_FACTOR), COL_VERMILION,
                         nbins=nbins, hist_range=hrange)
        draw_hybrid_hist(ax, y0 + 1 * sub_pitch, half, x, w, COL_GREY,
                         nbins=nbins, hist_range=hrange)
        draw_hybrid_hist(ax, y0 + 2 * sub_pitch, half, x,
                         pi_reweight(w, mode, PI_DOWN_FACTOR), COL_BLUE,
                         nbins=nbins, hist_range=hrange)
        group_centers.append(y0 + sub_pitch)

    for gi in range(1, n_groups):
        ax.axhline(gi * group_pitch - sub_pitch / 2, color="k", lw=0.4, alpha=0.3)

    ax.set_ylim(group_pitch * (n_groups - 1) + (rows_per_group - 1) * sub_pitch + 0.4,
                -0.4)

    ylabels = [OBS_LABELS[k] for k, *_ in triples]
    setup_axes(ax, n_groups, r"$E_{\nu}^{\rm reco} - E_{\nu}^{\rm true}$ [MeV]",
               XMIN_FIXED, XMAX_FIXED, ylabels, group_centers)

    plt.tight_layout()
    fname = f"pi_abs_{flavour}"
    plt.savefig(f"{OUT_DIR}/{fname}.png", dpi=180, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{fname}.png")
    return triples


# ---------------------------------------------------------------------------
# Figure 2 group: FSI vs noFSI, NuWro SF
# ---------------------------------------------------------------------------
def make_fsi_compare_figure(flavour):
    pairs = [
        ("hk_qe",    NUWRO_HK[flavour]["FSI"],   NUWRO_HK[flavour]["noFSI"],   "hk"),
        ("dune_epi", NUWRO_DUNE[flavour]["FSI"], NUWRO_DUNE[flavour]["noFSI"], "dune_epi"),  # E_avail
        ("dune_tpi", NUWRO_DUNE[flavour]["FSI"], NUWRO_DUNE[flavour]["noFSI"], "dune_tpi"),  # E_had
    ]
    rows = []
    for obs_key, fsi_path, nofsi_path, kind in pairs:
        if kind == "hk":
            x_f, w_f, _ = get_hk(fsi_path)
            x_n, w_n, _ = get_hk(nofsi_path)
        else:
            x_tpi_f, x_epi_f, w_f, _ = get_dune(fsi_path)
            x_tpi_n, x_epi_n, w_n, _ = get_dune(nofsi_path)
            x_f = x_tpi_f if kind == "dune_tpi" else x_epi_f
            x_n = x_tpi_n if kind == "dune_tpi" else x_epi_n
        rows.append((obs_key, x_f, w_f, x_n, w_n))

    n_groups = len(rows)
    rows_per_group = 2
    sub_pitch = 0.45
    group_pitch = rows_per_group * sub_pitch + 0.6
    half = sub_pitch / 2 * 0.85
    fig, ax = plt.subplots(figsize=(11.0, group_pitch * n_groups + 1.6))

    group_centers = []
    for gi, (obs_key, xf, wf, xn, wn) in enumerate(rows):
        y0 = gi * group_pitch
        nbins, hrange = hist_params_for(obs_key)
        draw_hybrid_hist(ax, y0 + 0 * sub_pitch, half, xf, wf, COL_GREY,
                         nbins=nbins, hist_range=hrange)
        draw_hybrid_hist(ax, y0 + 1 * sub_pitch, half, xn, wn, COL_VERMILION,
                         nbins=nbins, hist_range=hrange)
        group_centers.append(y0 + 0.5 * sub_pitch)

    for gi in range(1, n_groups):
        ax.axhline(gi * group_pitch - sub_pitch / 2, color="k", lw=0.4, alpha=0.3)

    ax.set_ylim(group_pitch * (n_groups - 1) + (rows_per_group - 1) * sub_pitch + 0.4,
                -0.4)

    ylabels = [OBS_LABELS[k] for k, *_ in rows]
    setup_axes(ax, n_groups, r"$E_{\nu}^{\rm reco} - E_{\nu}^{\rm true}$ [MeV]",
               XMIN_FIXED, XMAX_FIXED, ylabels, group_centers)

    plt.tight_layout()
    fname = f"fsi_compare_{flavour}"
    plt.savefig(f"{OUT_DIR}/{fname}.png", dpi=180, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{fname}.png")
    return rows


# ---------------------------------------------------------------------------
# Figure 3 group: Cascade variation, GENIE 10a/b/c/d
# ---------------------------------------------------------------------------
def make_cascade_compare_figure(flavour):
    obs_groups = [("hk_qe", "hk"),
                  ("dune_epi", "dune_epi"),  # E_avail
                  ("dune_tpi", "dune_tpi")]  # E_had
    tunes = ["10a", "10b", "10c", "10d"]
    rows = []
    for obs_key, kind in obs_groups:
        for tune in tunes:
            if kind == "hk":
                x, w, _ = get_hk(GENIE_HK[(flavour, tune)])
            else:
                x_tpi, x_epi, w, _ = get_dune(GENIE_DUNE[(flavour, tune)])
                x = x_tpi if kind == "dune_tpi" else x_epi
            rows.append({"obs_key": obs_key, "tune": tune,
                         "color": CASCADE_COLORS[tune], "x": x, "w": w})

    obs_keys = [k for k, _ in obs_groups]
    n_groups = len(obs_keys)
    rows_per_group = len(tunes)
    sub_pitch = 0.45
    group_pitch = rows_per_group * sub_pitch + 0.6
    half = sub_pitch / 2 * 0.85
    fig, ax = plt.subplots(figsize=(11.0, group_pitch * n_groups + 1.6))

    by_obs = {ok: [r for r in rows if r["obs_key"] == ok] for ok in obs_keys}

    group_centers = []
    for gi, ok in enumerate(obs_keys):
        y0 = gi * group_pitch
        nbins, hrange = hist_params_for(ok)
        for ti, t in enumerate(tunes):
            r = next(rr for rr in by_obs[ok] if rr["tune"] == t)
            draw_hybrid_hist(ax, y0 + ti * sub_pitch, half,
                             r["x"], r["w"], CASCADE_COLORS[t],
                             nbins=nbins, hist_range=hrange)
        group_centers.append(y0 + (rows_per_group - 1) * sub_pitch / 2)

    for gi in range(1, n_groups):
        ax.axhline(gi * group_pitch - sub_pitch / 2, color="k", lw=0.4, alpha=0.3)

    ax.set_ylim(group_pitch * (n_groups - 1) + (rows_per_group - 1) * sub_pitch + 0.4,
                -0.4)

    ylabels = [OBS_LABELS[ok] for ok in obs_keys]
    setup_axes(ax, n_groups, r"$E_{\nu}^{\rm reco} - E_{\nu}^{\rm true}$ [MeV]",
               XMIN_FIXED, XMAX_FIXED, ylabels, group_centers)

    plt.tight_layout()
    fname = f"cascade_compare_{flavour}"
    plt.savefig(f"{OUT_DIR}/{fname}.png", dpi=180, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{fname}.png")
    return rows


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
print(f"MAX_EVENTS = {MAX_EVENTS}\n")
for flavour in ("numu", "numubar"):
    print(f"=== {flavour} pi-abs ===")
    make_pi_abs_figure(flavour)
    print(f"=== {flavour} FSI vs noFSI ===")
    make_fsi_compare_figure(flavour)
    print(f"=== {flavour} cascade ===")
    make_cascade_compare_figure(flavour)

def colour_swatch(color, label, alpha=HIST_FILL_ALPHA + 0.10):
    return Line2D([0], [0], marker="s", linestyle="", markersize=14,
                  markerfacecolor=color, markeredgecolor=color,
                  alpha=alpha, label=label)

quantity_handles_hybrid = [
    colour_swatch("#777777", "histogram (per-row colour)"),
    Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=BOX_LW,
              label=r"$1\sigma$ box (16-84)"),
    Line2D([0], [0], color="black", lw=1.0, alpha=WHISKER_ALPHA,
           label=r"90\% whiskers (5-95)"),
    Line2D([0], [0], marker="o", linestyle="", markersize=7,
           markerfacecolor="white", markeredgecolor="black",
           markeredgewidth=1.4, label="median"),
    Line2D([0], [0], marker="D", linestyle="", markersize=6,
           markerfacecolor="white", markeredgecolor="black",
           markeredgewidth=1.4, label="mean"),
]

print("\n=== standalone legends ===")
save_standalone_legend(quantity_handles_hybrid, 5, "legend_quantity",
                       fig_w=14.0, fig_h=0.6)
save_standalone_legend([
    colour_swatch(COL_GREY,      r"$\pi_{\rm abs}$ nominal"),
    colour_swatch(COL_VERMILION, r"$\pi_{\rm abs}$ $+40\%$"),
    colour_swatch(COL_BLUE,      r"$\pi_{\rm abs}$ $-40\%$"),
], 3, "legend_pi_abs", fig_w=9.0, fig_h=0.55)
save_standalone_legend([
    colour_swatch(COL_GREY,      "with FSI"),
    colour_swatch(COL_VERMILION, "no FSI"),
], 2, "legend_fsi", fig_w=6.0, fig_h=0.55)
save_standalone_legend(
    [colour_swatch(c, f"GENIE G18\\_{t}") for t, c in CASCADE_COLORS.items()],
    4, "legend_cascade", fig_w=10.0, fig_h=0.55)

print("\nDONE")
