"""Box-and-whisker summary plots for the FSI IOP paper.

10 output figures (PNG + PDF) written to OUT_DIR, named in the same FigN_*
style as the canonical scripts/ folder:

  Fig8_FSIvsNoFSI_{numu,numubar}    NuWro SF, FSI vs noFSI samples (2 variants)
  Fig9_PiAbs_{numu,numubar}         NuWro SF FSI, kaskada_piN_abs_scale grid
                                      (3 variants: 0.69 / nominal / 1.31)
  Fig10_MFP_{numu,numubar}          NuWro SF FSI, kaskada_NN_mfp_scale grid
                                      (3 variants: 0.7 / nominal / 1.3)
  Fig11_GENIE_{numu,numubar}        GENIE NUISFLAT, four cascade tunes
                                      (10a/b/c/d)
  Fig12_EDRMF_{numu,numubar}        NEUT EDRMF vs RPWIA  (RPWIA paths are
                                      placeholders; row is skipped if files
                                      are missing)

Each figure has 3 row-groups (HK Eν_QE, DUNE Eν_avail, DUNE Eν_had). Each row
in a group is one variant rendered as a translucent histogram silhouette plus
1σ box (16-84%), faint 90% whiskers (5-95%), median (white circle) and
weighted mean (white diamond) markers.

Path-fallback: if a file is missing, the loader transparently falls back to
its `.bak.preOPpatch` or `.bak.preNNmfp` sibling. This lets the script run
against the saved-aside backups while a fresh kaskada_piN_abs_scale /
kaskada_NN_mfp_scale regen is still in flight on condor.

Set MAX_EVENTS=None for full statistics; small value gives fast layout iteration.
"""
import os
import sys
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Share Jake's mtime-keyed cache via FlatTreeMod.load_arrays. FlatTreeMod also
# sets the project rcParams (Computer Modern serif, IOP-calibrated font sizes).
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from FlatTreeMod import load_arrays  # noqa: E402  -- after sys.path tweak

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs"
NEUT_BASE = "/eos/home-l/lamuntea/FSI_IOP_paper/neut_runs"
OUT_DIR = "/eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw"
MAX_EVENTS = None  # full stats (cache hit after first run)

OUTER_ALPHA = 0.35

XMIN_FIXED, XMAX_FIXED = -1000.0, 1000.0  # visible x-axis [MeV]
HK_NBINS, HK_HIST_RANGE   = 200, (-1000.0, 1000.0)   # 10 MeV bins, [-1, +1] GeV
DUNE_NBINS, DUNE_HIST_RANGE = 400, (-3000.0, 1000.0) # 10 MeV bins, [-3, +1] GeV
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
COL_ORANGE = "#E69F00"

CASCADE_COLORS = {"10a": COL_BLUE, "10b": COL_VERMILION,
                  "10c": COL_GREEN, "10d": COL_PURPLE}

# ---------------------------------------------------------------------------
# Path resolver: live files only. Returns None if the file is missing
# (used for EDRMF/RPWIA placeholders so the corresponding rows skip cleanly).
# Backup-fallback to .bak.preOPpatch was removed once the OP-patched regen
# landed on 2026-05-03; if you need to read backups again, restore the
# fallback or rename the .bak files back into place.
# ---------------------------------------------------------------------------
def resolve_path(path):
    if path and os.path.exists(path):
        return path
    if path:
        print(f"  [resolve_path] WARNING: file missing: {path}")
    return None


# ---------------------------------------------------------------------------
# File catalogues
# ---------------------------------------------------------------------------
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

# kaskada_piN_abs_scale grid (069 = -31%, 131 = +31%). Nominal is the
# top-level _FSI file (kaskada_piN_abs_scale defaults to 1.0).
NUWRO_HK_PIABS = {
    ("numu",    "069"): f"{BASE}/HK/piabs/HK_numu_piabs069_FSI.flat.root",
    ("numu",    "131"): f"{BASE}/HK/piabs/HK_numu_piabs131_FSI.flat.root",
    ("numubar", "069"): f"{BASE}/HK/piabs/HK_numubar_piabs069_FSI.flat.root",
    ("numubar", "131"): f"{BASE}/HK/piabs/HK_numubar_piabs131_FSI.flat.root",
}
NUWRO_DUNE_PIABS = {
    ("numu",    "069"): f"{BASE}/DUNE/piabs/DUNE_numu_piabs069_FSI.flat.root",
    ("numu",    "131"): f"{BASE}/DUNE/piabs/DUNE_numu_piabs131_FSI.flat.root",
    ("numubar", "069"): f"{BASE}/DUNE/piabs/DUNE_numub_piabs069_FSI.flat.root",
    ("numubar", "131"): f"{BASE}/DUNE/piabs/DUNE_numub_piabs131_FSI.flat.root",
}

# kaskada_NN_mfp_scale grid (070 = 0.7, 130 = 1.3). Nominal is top-level _FSI.
NUWRO_HK_MFP = {
    ("numu",    "0p7"): f"{BASE}/HK/ChangeMFP/HK_numu_0p7MFP_FSI.flat.root",
    ("numu",    "1p3"): f"{BASE}/HK/ChangeMFP/HK_numu_1p3MFP_FSI.flat.root",
    ("numubar", "0p7"): f"{BASE}/HK/ChangeMFP/HK_numubar_0p7MFP_FSI.flat.root",
    ("numubar", "1p3"): f"{BASE}/HK/ChangeMFP/HK_numubar_1p3MFP_FSI.flat.root",
}
NUWRO_DUNE_MFP = {
    ("numu",    "0p7"): f"{BASE}/DUNE/ChangeMFP/DUNE_numu_0p7MFP_FSI.flat.root",
    ("numu",    "1p3"): f"{BASE}/DUNE/ChangeMFP/DUNE_numu_1p3MFP_FSI.flat.root",
    ("numubar", "0p7"): f"{BASE}/DUNE/ChangeMFP/DUNE_numub_0p7MFP_FSI.flat.root",
    ("numubar", "1p3"): f"{BASE}/DUNE/ChangeMFP/DUNE_numub_1p3MFP_FSI.flat.root",
}

# GENIE 4-tune cascade comparison. Files moved into a GENIE/ subfolder when
# the project share was reorganised (see scripts/move_genie.sh equivalent).
GENIE_HK = {
    ("numu", t):    f"{BASE}/HK/GENIE/T2KSK_unosc_FHC_numu_H2O_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
}
GENIE_HK.update({
    ("numubar", t): f"{BASE}/HK/GENIE/T2KSK_unosc_RHC_numubar_H2O_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
})
GENIE_DUNE = {
    ("numu", t):    f"{BASE}/DUNE/GENIE/DUNEFD_unosc_FHC_numu_Ar40_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
}
GENIE_DUNE.update({
    ("numubar", t): f"{BASE}/DUNE/GENIE/DUNEFD_unosc_RHC_numubar_Ar40_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
})

# ---------------------------------------------------------------------------
# NEUT EDRMF vs RPWIA paths -- PLACEHOLDERS.
# RPWIA flat files are not yet on disk for any sample. EDRMF flat files
# exist as test trees only. Replace with production paths once available.
# Each row is rendered only if its NEUT files resolve via resolve_path().
# ---------------------------------------------------------------------------
NEUT_EDRMF = {
    "hk_numu":     f"{NEUT_BASE}/HK_numu_H2O_EDRMF/EDRMF.flat.root",          # PLACEHOLDER
    "hk_numubar":  f"{NEUT_BASE}/HK_numubar_H2O_EDRMF/EDRMF.flat.root",       # PLACEHOLDER
    "dune_numu":   f"{NEUT_BASE}/DUNE_numu_Ar40_EDRMF/EDRMF.flat.root",       # PLACEHOLDER
    "dune_numubar":f"{NEUT_BASE}/DUNE_numubar_Ar40_EDRMF/EDRMF.flat.root",    # PLACEHOLDER
}
NEUT_RPWIA = {
    "hk_numu":     f"{NEUT_BASE}/HK_numu_H2O_RPWIA/RPWIA.flat.root",          # PLACEHOLDER
    "hk_numubar":  f"{NEUT_BASE}/HK_numubar_H2O_RPWIA/RPWIA.flat.root",       # PLACEHOLDER
    "dune_numu":   f"{NEUT_BASE}/DUNE_numu_Ar40_RPWIA/RPWIA.flat.root",       # PLACEHOLDER
    "dune_numubar":f"{NEUT_BASE}/DUNE_numubar_Ar40_RPWIA/RPWIA.flat.root",    # PLACEHOLDER
}

OBS_LABELS = {
    "hk_qe":   r"HK  $E_{\nu}^{\mathrm{QE}}$",
    "dune_epi": r"DUNE  $E_{\nu}^{\mathrm{avail}}$",
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
# File readers — go through FlatTreeMod.load_arrays so the on-disk cache is
# shared with all the canonical scripts/ figures. load_arrays is keyed on
# (file path, size, mtime, branch list, max_events) → auto-busts when the
# underlying ROOT file changes. The bias derivation below stays here in
# laura_extras (it's specific to this BW summary; not in FlatTreeMod).
# ---------------------------------------------------------------------------
def read_hk(path):
    """HK QE bias [MeV] on derived CC0pi (cc=1, no charged-pi, no pi0)."""
    a = load_arrays(path, max_events=MAX_EVENTS)
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
    Returns (x_tpi, x_epi, w, mode).
    """
    a = load_arrays(path, max_events=MAX_EVENTS)
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
    sel = is_cc
    x_tpi = np.asarray(bias_tpi[sel])
    x_epi = np.asarray(bias_epi[sel])
    w = np.asarray(a["fScaleFactor"][sel])
    mode = np.asarray(a["Mode"][sel])
    return x_tpi, x_epi, w, mode


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def draw_hybrid_hist(ax, y, half, x, w, color,
                     nbins=HK_NBINS, hist_range=HK_HIST_RANGE):
    """Histogram silhouette + 1-sigma box outline + faint 90% whiskers +
    median/mean markers, all in `color`. Skips drawing if x is empty.

    Weights are divided by bin width so the silhouette shape is dσ/dE in
    arbitrary fScaleFactor / MeV units. The silhouette is then normalised
    to peak=1 to fit a row, but the underlying shape is now consistent with
    the validation distributions that show absolute dσ/dE.
    """
    if x is None or len(x) == 0:
        return
    s = stats(x, w)
    bin_width = (hist_range[1] - hist_range[0]) / nbins
    counts, edges = np.histogram(x, bins=nbins, range=hist_range,
                                 weights=w / bin_width)
    if counts.max() > 0:
        counts = counts / counts.max()
    xs_step = np.empty(2 * len(counts))
    xs_step[0::2] = edges[:-1]
    xs_step[1::2] = edges[1:]
    ys_step = np.repeat(counts, 2)
    poly_x = np.concatenate([xs_step, xs_step[::-1]])
    poly_y = np.concatenate([y - half * ys_step, y + half * ys_step[::-1]])
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


def setup_axes(ax, n_groups, xlabel, xmin, xmax, group_labels, group_centers):
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(group_centers)
    ax.set_yticklabels(group_labels)
    ax.axvline(0, color="gray", ls="--", lw=0.7)
    ax.set_xlabel(xlabel)


def save_standalone_legend(handles, ncol, fname, fig_w=10.0, fig_h=0.6,
                           fontsize=None):
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.legend(handles=handles, loc="center", ncol=ncol, fontsize=fontsize,
               frameon=True, framealpha=0.9)
    plt.savefig(f"{OUT_DIR}/{fname}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{fname}.png")


def save_combined_legend(metric_handles, color_handles, color_title, fname,
                         fig_w=10.0, fig_h=2.0):
    """Per-BW-set legend combining (left) the metric markers shared by every
    figure and (right) the variant colours specific to this set."""
    fig = plt.figure(figsize=(fig_w, fig_h))
    leg1 = fig.legend(handles=metric_handles, title="Markers",
                      loc="center", bbox_to_anchor=(0.30, 0.5),
                      ncol=1, frameon=True, framealpha=0.9,
                      title_fontproperties={"weight": "bold"})
    leg2 = fig.legend(handles=color_handles, title=color_title,
                      loc="center", bbox_to_anchor=(0.72, 0.5),
                      ncol=1, frameon=True, framealpha=0.9,
                      title_fontproperties={"weight": "bold"})
    fig.add_artist(leg1)  # ensure both legends render (second overrides first by default)
    plt.savefig(f"{OUT_DIR}/{fname}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{fname}.png")


# ---------------------------------------------------------------------------
# In-process memoisation for the (already-cached) reads.
# Disk cache is provided by FlatTreeMod.load_arrays at ~/.cache/iop_paper/ —
# shared with every other script. We just memoise the post-derivation result
# (x, w, mode) so repeated calls in one Python process don't re-do the
# CC0π / Tpi / Epi computation on the awkward arrays.
# ---------------------------------------------------------------------------
_CACHE = {}


def get_hk(path):
    p = resolve_path(path)
    if p is None:
        return None
    if ("hk", p) not in _CACHE:
        print(f"  deriving hk from: {os.path.basename(p)}")
        _CACHE[("hk", p)] = read_hk(p)
    return _CACHE[("hk", p)]


def get_dune(path):
    p = resolve_path(path)
    if p is None:
        return None
    if ("dune", p) not in _CACHE:
        print(f"  deriving dune from: {os.path.basename(p)}")
        _CACHE[("dune", p)] = read_dune(p)
    return _CACHE[("dune", p)]


def get_obs(path, obs_kind):
    """Generic per-observable getter. obs_kind in {hk_qe, dune_epi, dune_tpi}.
    Returns (x, w) or None if file unresolvable."""
    if obs_kind == "hk_qe":
        out = get_hk(path)
        if out is None: return None
        x, w, _ = out
        return x, w
    out = get_dune(path)
    if out is None: return None
    x_tpi, x_epi, w, _ = out
    return (x_tpi, w) if obs_kind == "dune_tpi" else (x_epi, w)


def get_obs_with_mode(path, obs_kind):
    """Like get_obs but also returns the per-event Mode array.
    Returns (x, w, mode) or None if file unresolvable. Used by the
    validation plot to highlight |Mode|==16 (coherent π production)."""
    if obs_kind == "hk_qe":
        out = get_hk(path)
        if out is None: return None
        return out  # already (x, w, mode)
    out = get_dune(path)
    if out is None: return None
    x_tpi, x_epi, w, mode = out
    return ((x_tpi, w, mode) if obs_kind == "dune_tpi"
            else (x_epi, w, mode))


def hk_or_dune_path(obs_kind, paths_hk, paths_dune):
    """Pick HK path for hk_qe rows, DUNE path otherwise."""
    return paths_hk if obs_kind == "hk_qe" else paths_dune


# ---------------------------------------------------------------------------
# Generic figure builder. variants: list of (label, color, paths_per_obs)
# where paths_per_obs is a dict {obs_key: file_path}.
# ---------------------------------------------------------------------------
OBS_GROUPS = [("hk_qe", "hk"),
              ("dune_epi", "dune_epi"),
              ("dune_tpi", "dune_tpi")]


def _make_figure(fname_stem, flavour, variants, group_label_suffix=""):
    """variants: list of dicts with keys
        label (str),  color (hex),  paths (dict obs_key -> file_path)
    Each row in a group is one variant. Skips a variant in a row if the
    corresponding file unresolvable."""
    obs_keys = [k for k, _ in OBS_GROUPS]
    n_groups = len(obs_keys)
    rows_per_group = len(variants)
    sub_pitch = 0.45
    group_pitch = rows_per_group * sub_pitch + 0.6
    half = sub_pitch / 2 * 0.85
    fig, ax = plt.subplots(figsize=(11.0, group_pitch * n_groups + 1.6))

    group_centers = []
    for gi, ok in enumerate(obs_keys):
        y0 = gi * group_pitch
        nbins, hrange = hist_params_for(ok)
        for vi, v in enumerate(variants):
            path = v["paths"].get(ok)
            res = get_obs(path, ok) if path else None
            if res is None:
                continue
            x, w = res
            draw_hybrid_hist(ax, y0 + vi * sub_pitch, half, x, w,
                             v["color"], nbins=nbins, hist_range=hrange)
        group_centers.append(y0 + (rows_per_group - 1) * sub_pitch / 2)

    for gi in range(1, n_groups):
        ax.axhline(gi * group_pitch - sub_pitch / 2, color="k", lw=0.4, alpha=0.3)

    ax.set_ylim(group_pitch * (n_groups - 1) + (rows_per_group - 1) * sub_pitch + 0.4,
                -0.4)

    ylabels = [OBS_LABELS[ok] + group_label_suffix for ok in obs_keys]
    setup_axes(ax, n_groups, r"$E_{\nu}^{\rm reco} - E_{\nu}^{\rm true}$ [MeV]",
               XMIN_FIXED, XMAX_FIXED, ylabels, group_centers)

    plt.tight_layout()
    out = f"{fname_stem}_{flavour}"
    plt.savefig(f"{OUT_DIR}/{out}.png", dpi=180, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{out}.png")


# ---------------------------------------------------------------------------
# Fig 8 -- FSI vs noFSI, NuWro SF
# ---------------------------------------------------------------------------
def make_fsi_compare_figure(flavour):
    variants = [
        {"label": "with FSI", "color": COL_GREY, "paths": {
            "hk_qe":    NUWRO_HK[flavour]["FSI"],
            "dune_epi": NUWRO_DUNE[flavour]["FSI"],
            "dune_tpi": NUWRO_DUNE[flavour]["FSI"],
        }},
        {"label": "no FSI", "color": COL_VERMILION, "paths": {
            "hk_qe":    NUWRO_HK[flavour]["noFSI"],
            "dune_epi": NUWRO_DUNE[flavour]["noFSI"],
            "dune_tpi": NUWRO_DUNE[flavour]["noFSI"],
        }},
    ]
    _make_figure("Fig8_FSIvsNoFSI", flavour, variants)


# ---------------------------------------------------------------------------
# Fig 9 -- piabs grid (kaskada_piN_abs_scale = 0.69 / 1.0 / 1.31)
# ---------------------------------------------------------------------------
def make_pi_abs_figure(flavour):
    """Replaces the legacy event-reweight version. Now reads the actual
    cascade samples (kaskada_piN_abs_scale dial) so the variation is real
    FSI physics rather than a Mode-dependent reweight."""
    variants = [
        {"label": r"$\pi_{\rm abs}$ +31\%", "color": COL_VERMILION, "paths": {
            "hk_qe":    NUWRO_HK_PIABS[(flavour, "131")],
            "dune_epi": NUWRO_DUNE_PIABS[(flavour, "131")],
            "dune_tpi": NUWRO_DUNE_PIABS[(flavour, "131")],
        }},
        {"label": "nominal", "color": COL_GREY, "paths": {
            "hk_qe":    NUWRO_HK[flavour]["FSI"],
            "dune_epi": NUWRO_DUNE[flavour]["FSI"],
            "dune_tpi": NUWRO_DUNE[flavour]["FSI"],
        }},
        {"label": r"$\pi_{\rm abs}$ -31\%", "color": COL_BLUE, "paths": {
            "hk_qe":    NUWRO_HK_PIABS[(flavour, "069")],
            "dune_epi": NUWRO_DUNE_PIABS[(flavour, "069")],
            "dune_tpi": NUWRO_DUNE_PIABS[(flavour, "069")],
        }},
    ]
    _make_figure("Fig9_PiAbs", flavour, variants)


# ---------------------------------------------------------------------------
# Fig 10 -- NN_mfp grid (kaskada_NN_mfp_scale = 0.7 / 1.0 / 1.3)
# ---------------------------------------------------------------------------
def make_mfp_compare_figure(flavour):
    variants = [
        {"label": r"$0.7 \times \rm MFP$", "color": COL_VERMILION, "paths": {
            "hk_qe":    NUWRO_HK_MFP[(flavour, "0p7")],
            "dune_epi": NUWRO_DUNE_MFP[(flavour, "0p7")],
            "dune_tpi": NUWRO_DUNE_MFP[(flavour, "0p7")],
        }},
        {"label": "nominal", "color": COL_GREY, "paths": {
            "hk_qe":    NUWRO_HK[flavour]["FSI"],
            "dune_epi": NUWRO_DUNE[flavour]["FSI"],
            "dune_tpi": NUWRO_DUNE[flavour]["FSI"],
        }},
        {"label": r"$1.3 \times \rm MFP$", "color": COL_BLUE, "paths": {
            "hk_qe":    NUWRO_HK_MFP[(flavour, "1p3")],
            "dune_epi": NUWRO_DUNE_MFP[(flavour, "1p3")],
            "dune_tpi": NUWRO_DUNE_MFP[(flavour, "1p3")],
        }},
    ]
    _make_figure("Fig10_MFP", flavour, variants)


# ---------------------------------------------------------------------------
# Fig 11 -- GENIE cascade tunes (G18_10a/b/c/d)
# ---------------------------------------------------------------------------
def make_cascade_compare_figure(flavour):
    tunes = ["10a", "10b", "10c", "10d"]
    variants = [
        {"label": f"G18\\_{t}", "color": CASCADE_COLORS[t], "paths": {
            "hk_qe":    GENIE_HK[(flavour, t)],
            "dune_epi": GENIE_DUNE[(flavour, t)],
            "dune_tpi": GENIE_DUNE[(flavour, t)],
        }}
        for t in tunes
    ]
    _make_figure("Fig11_GENIE", flavour, variants)


# ---------------------------------------------------------------------------
# Fig 12 -- NEUT EDRMF vs RPWIA (RPWIA paths are PLACEHOLDERS for now)
# ---------------------------------------------------------------------------
def make_edrmf_compare_figure(flavour):
    hk_key = f"hk_{flavour}"
    dune_key = f"dune_{flavour}"
    variants = [
        {"label": "EDRMF", "color": COL_GREY, "paths": {
            "hk_qe":    NEUT_EDRMF[hk_key],
            "dune_epi": NEUT_EDRMF[dune_key],
            "dune_tpi": NEUT_EDRMF[dune_key],
        }},
        {"label": "RPWIA", "color": COL_ORANGE, "paths": {
            "hk_qe":    NEUT_RPWIA[hk_key],
            "dune_epi": NEUT_RPWIA[dune_key],
            "dune_tpi": NEUT_RPWIA[dune_key],
        }},
    ]
    _make_figure("Fig12_EDRMF", flavour, variants)


# ---------------------------------------------------------------------------
# Standalone legends (helpers used by main and importable)
# ---------------------------------------------------------------------------
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

def _save_all_legends():
    """One combined legend per BW set: metrics (median, mean, 1σ box,
    whiskers, histogram silhouette) on the left, the variant colour key
    for that set on the right. Filenames mirror the figure stems
    (Fig8_FSIvsNoFSI -> legend_Fig8_FSIvsNoFSI)."""
    save_combined_legend(
        quantity_handles_hybrid,
        [colour_swatch(COL_GREY,      "with FSI"),
         colour_swatch(COL_VERMILION, "no FSI")],
        "FSI vs no FSI", "legend_Fig8_FSIvsNoFSI",
    )
    save_combined_legend(
        quantity_handles_hybrid,
        [colour_swatch(COL_VERMILION, r"$\pi_{\rm abs}$ $+31\%$"),
         colour_swatch(COL_GREY,      "nominal"),
         colour_swatch(COL_BLUE,      r"$\pi_{\rm abs}$ $-31\%$")],
        r"$\pi$-absorption $\pm 31\%$", "legend_Fig9_PiAbs",
    )
    save_combined_legend(
        quantity_handles_hybrid,
        [colour_swatch(COL_VERMILION, r"$0.7\times$ NN MFP"),
         colour_swatch(COL_GREY,      "nominal"),
         colour_swatch(COL_BLUE,      r"$1.3\times$ NN MFP")],
        r"NN mean free path $\pm 30\%$", "legend_Fig10_MFP",
    )
    save_combined_legend(
        quantity_handles_hybrid,
        [colour_swatch(c, f"GENIE G18\\_{t}") for t, c in CASCADE_COLORS.items()],
        "GENIE cascade tunes", "legend_Fig11_GENIE",
    )
    save_combined_legend(
        quantity_handles_hybrid,
        [colour_swatch(COL_GREY,   "EDRMF"),
         colour_swatch(COL_ORANGE, "RPWIA")],
        "NEUT nuclear model", "legend_Fig12_EDRMF",
    )


# ---------------------------------------------------------------------------
# Variant catalogue — exposed so other scripts (validation, table) can walk
# the exact same set of (category, variant, paths) tuples that drive the BW
# plots above.
# ---------------------------------------------------------------------------
def _hk_or_dune(obs_key, hk_path, dune_path):
    return hk_path if obs_key == "hk_qe" else dune_path


def variants_fsi(flav):
    return [
        ("with FSI", COL_GREY, lambda obs: _hk_or_dune(
            obs, NUWRO_HK[flav]["FSI"], NUWRO_DUNE[flav]["FSI"])),
        ("no FSI",   COL_VERMILION, lambda obs: _hk_or_dune(
            obs, NUWRO_HK[flav]["noFSI"], NUWRO_DUNE[flav]["noFSI"])),
    ]


def variants_piabs(flav):
    # NB: '%' is escaped as '\%' so labels render correctly under
    # matplotlib's usetex=True (otherwise LaTeX treats % as a comment).
    return [
        (r"$\pi_{\rm abs}$ -31\%", COL_BLUE, lambda obs: _hk_or_dune(
            obs, NUWRO_HK_PIABS[(flav, "069")], NUWRO_DUNE_PIABS[(flav, "069")])),
        ("nominal", COL_GREY, lambda obs: _hk_or_dune(
            obs, NUWRO_HK[flav]["FSI"], NUWRO_DUNE[flav]["FSI"])),
        (r"$\pi_{\rm abs}$ +31\%", COL_VERMILION, lambda obs: _hk_or_dune(
            obs, NUWRO_HK_PIABS[(flav, "131")], NUWRO_DUNE_PIABS[(flav, "131")])),
    ]


def variants_mfp(flav):
    return [
        (r"0.7$\times$ MFP", COL_VERMILION, lambda obs: _hk_or_dune(
            obs, NUWRO_HK_MFP[(flav, "0p7")], NUWRO_DUNE_MFP[(flav, "0p7")])),
        ("nominal", COL_GREY, lambda obs: _hk_or_dune(
            obs, NUWRO_HK[flav]["FSI"], NUWRO_DUNE[flav]["FSI"])),
        (r"1.3$\times$ MFP", COL_BLUE, lambda obs: _hk_or_dune(
            obs, NUWRO_HK_MFP[(flav, "1p3")], NUWRO_DUNE_MFP[(flav, "1p3")])),
    ]


def variants_genie(flav):
    return [
        (f"G18_{t}", CASCADE_COLORS[t], lambda obs, _t=t: _hk_or_dune(
            obs, GENIE_HK[(flav, _t)], GENIE_DUNE[(flav, _t)]))
        for t in ("10a", "10b", "10c", "10d")
    ]


def variants_edrmf(flav):
    hk_key = f"hk_{flav}" if flav in NEUT_EDRMF else f"hk_{flav}"
    dune_key = f"dune_{flav}"
    return [
        ("EDRMF", COL_GREY, lambda obs: NEUT_EDRMF.get(hk_key) if obs == "hk_qe" else NEUT_EDRMF.get(dune_key)),
        ("RPWIA", COL_ORANGE, lambda obs: NEUT_RPWIA.get(hk_key) if obs == "hk_qe" else NEUT_RPWIA.get(dune_key)),
    ]


CATEGORY_VARIANTS = [
    ("FSI vs no FSI",            variants_fsi),
    (r"$\pi_{\rm abs}$ $\pm 31\%$", variants_piabs),
    (r"NN MFP $\pm 30\%$",       variants_mfp),
    ("GENIE cascade 10a-10d",    variants_genie),
    ("EDRMF vs RPWIA",           variants_edrmf),
]


# ---------------------------------------------------------------------------
# Main entrypoint — only runs when invoked directly. Other scripts can
# `from fsi_iop_plots import ...` without triggering the whole regen.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"MAX_EVENTS = {MAX_EVENTS}\n")
    for flavour in ("numu", "numubar"):
        print(f"=== {flavour} FSI vs noFSI ===")
        make_fsi_compare_figure(flavour)
        print(f"=== {flavour} pi-abs (kaskada_piN_abs_scale grid) ===")
        make_pi_abs_figure(flavour)
        print(f"=== {flavour} NN_mfp (kaskada_NN_mfp_scale grid) ===")
        make_mfp_compare_figure(flavour)
        print(f"=== {flavour} GENIE cascade tunes ===")
        make_cascade_compare_figure(flavour)
        print(f"=== {flavour} NEUT EDRMF vs RPWIA (placeholder) ===")
        make_edrmf_compare_figure(flavour)
    print("\n=== standalone legends ===")
    _save_all_legends()
    print("\nDONE")
