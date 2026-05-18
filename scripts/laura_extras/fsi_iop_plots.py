"""Box-and-whisker summary plots for the FSI IOP paper.

5 figures × 2 flavours × 2 modes (abs, rel) = 20 output PDFs (PNG + PDF),
written to OUT_DIR. Names follow the FigN_*_<flavour>_<mode>.pdf scheme:

  Fig8_FSIvsNoFSI_{numu,numubar}_{abs,rel}
  Fig9_PiAbs_{numu,numubar}_{abs,rel}
  Fig10_MFP_{numu,numubar}_{abs,rel}
  Fig11_GENIE_{numu,numubar}_{abs,rel}
  Fig12_EDRMF_{numu,numubar}_{abs,rel}

Each figure has 3 row-groups (HK Eν_QE, DUNE Eν_avail, DUNE Eν_had). Each row
in a group is one variant rendered as a translucent histogram silhouette plus
1σ box (16-84%), faint 90% whiskers (5-95%), median (white circle) and
weighted mean (white diamond) markers.

mode='abs' shows (E_reco − E_true) in MeV; mode='rel' shows the same divided
by E_true (dimensionless). Both share the same canonical CC0π / CC selections
via FlatTreeMod.bias_arr.

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
from FlatTreeMod import (load_arrays, bias_arr,  # noqa: E402
                         is_cc0pi_arr)

# BW-specific font bump. The BW grid figures sit at 3.5" wide (one half of
# \textwidth in LaTeX) with a lot of in-figure box content per row, so the
# global 10/9/9 pt scheme reads as small relative to the data ink. Bump to
# 13/11/11 here so labels and ticks remain legible at print size without
# affecting the canonical Fig1-7 plots.
plt.rcParams.update({
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/nuwro_25031_morestats"
NEUT_BASE = "/eos/home-l/lamuntea/FSI_IOP_paper/neut_runs"
NEUT_FILES_BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs"  # NEUT EDRMF/RPWIA samples — separate from NuWro BASE
GENIE_FILES_BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs"  # existing GENIE NUISFLAT samples — separate from NuWro BASE
# OUT_DIR is overridable via $OUTPUT_PLOTS_DIR for bulk regen into iop_plots/.
# When set, BW outputs land in $OUTPUT_PLOTS_DIR/BW_summaries/ (a subdir, to
# keep them grouped alongside the Fig{N}_plots/ subdirs the bias scripts use).
_root = os.environ.get("OUTPUT_PLOTS_DIR")
OUT_DIR = (os.path.join(_root, "BW_summaries") if _root
           else "/eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw")
MAX_EVENTS = None  # full stats (cache hit after first run)

OUTER_ALPHA = 0.35

# Per-mode plot limits. abs is MeV; rel is dimensionless (-E_true normalised).
# Note: laura_extras BW plots use a tighter range than the parent Fig*_plot.py
# scripts (which use REL_BIAS_XLIM = (-0.9, 0.3)) -- the BW box+whisker tails
# rarely extend below -50% so the wider range wastes horizontal space.
XLIM = {
    "abs": (-500.0, 300.0),
    "rel": (-0.5, 0.3),
}
HIST_RANGE_HK = {
    "abs": (-1000.0, 1000.0),
    "rel": (-1.0, 1.0),
}
HIST_RANGE_DUNE = {
    "abs": (-3000.0, 1000.0),
    "rel": (-3.0, 1.0),
}
HK_NBINS   = 200
DUNE_NBINS = 400
HIST_FILL_ALPHA = 0.22
WHISKER_ALPHA = 0.50
BOX_LW = 1.4


def hist_params_for(obs_key, mode):
    """Pick (nbins, range) by experiment + mode."""
    if obs_key == "hk_qe":
        return HK_NBINS, HIST_RANGE_HK[mode]
    return DUNE_NBINS, HIST_RANGE_DUNE[mode]


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
GENIE_LABELS = {"10a": "hA2018", "10b": "hN2018",
                "10c": "INCL++", "10d": "G4BC"}


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
GENIE_HK = {
    ("numu", t):    f"{GENIE_FILES_BASE}/HK/GENIE/T2KSK_unosc_FHC_numu_H2O_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
}
GENIE_HK.update({
    ("numubar", t): f"{GENIE_FILES_BASE}/HK/GENIE/T2KSK_unosc_RHC_numubar_H2O_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
})
GENIE_DUNE = {
    ("numu", t):    f"{GENIE_FILES_BASE}/DUNE/GENIE/DUNEFD_unosc_FHC_numu_Ar40_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
}
GENIE_DUNE.update({
    ("numubar", t): f"{GENIE_FILES_BASE}/DUNE/GENIE/DUNEFD_unosc_RHC_numubar_Ar40_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
})

NEUT_EDRMF = {
    "hk_numu":     f"{NEUT_FILES_BASE}/HK/NEUT_HK_EDRMF_numu.flat.root",
    "hk_numubar":  None,
    "dune_numu":   f"{NEUT_FILES_BASE}/DUNE/DUNE_EDRMF_numu.root",
    "dune_numubar":None,
}
NEUT_RPWIA = {
    "hk_numu":     f"{NEUT_FILES_BASE}/HK/NEUT_HK_RPWIA_numu.flat.root",
    "hk_numubar":  None,
    "dune_numu":   f"{NEUT_FILES_BASE}/DUNE/DUNE_RPWIA_numu.root",
    "dune_numubar":None,
}

OBS_LABELS = {
    "hk_qe":   r"HK  $E_{\nu}^{\mathrm{QE}}$",
    "dune_epi": r"DUNE  $E_{\nu}^{\mathrm{avail}}$",
    "dune_tpi": r"DUNE  $E_{\nu}^{\mathrm{had}}$",
}

NU_LABELS = {"numu": r"$\nu_{\mu}$ FHC", "numubar": r"$\bar{\nu}_{\mu}$ RHC"}


def _xlabel(mode):
    if mode == "abs":
        return r"$E_{\nu}^{\rm reco} - E_{\nu}^{\rm true}$ [MeV]"
    return r"$(E_{\nu}^{\rm reco} - E_{\nu}^{\rm true}) / E_{\nu}^{\rm true}$"


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
# File readers — go through FlatTreeMod.load_arrays for the disk cache; the
# observable extraction goes through FlatTreeMod.bias_arr so abs and rel use
# the same canonical CC0π / CC selections as the parent Fig*_plot.py scripts.
# ---------------------------------------------------------------------------
def read_hk(path, mode):
    """HK QE bias on derived CC0π. Returns (x, w, mode_arr)."""
    a = load_arrays(path, max_events=MAX_EVENTS)
    x = bias_arr(a, "qe", kind=mode, vertex=False)
    sel = is_cc0pi_arr(a, vertex=False)
    w = np.asarray(a["fScaleFactor"][sel])
    mode_arr = np.asarray(a["Mode"][sel])
    return x, w, mode_arr


def read_dune(path, mode):
    """DUNE calorimetric bias, two definitions, on CC inclusive.
    Returns (x_tpi, x_epi, w, mode_arr).
    """
    a = load_arrays(path, max_events=MAX_EVENTS)
    x_tpi = bias_arr(a, "had",   kind=mode, vertex=False)
    x_epi = bias_arr(a, "avail", kind=mode, vertex=False)
    is_cc = np.asarray(a["cc"], dtype=bool)
    w = np.asarray(a["fScaleFactor"][is_cc])
    mode_arr = np.asarray(a["Mode"][is_cc])
    return x_tpi, x_epi, w, mode_arr


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def draw_hybrid_hist(ax, y, half, x, w, color,
                     nbins=HK_NBINS, hist_range=HIST_RANGE_HK["abs"]):
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

    ax.plot(s["median"], y, "o", color="white", markersize=7,
            markeredgecolor=color, markeredgewidth=1.4, zorder=6)
    ax.plot(s["mean"], y, "D", color="white", markersize=4.0,
            markeredgecolor=color, markeredgewidth=1.4, zorder=6)


def setup_axes(ax, n_groups, xlabel, xmin, xmax, group_labels, group_centers,
               show_ylabels=True):
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(group_centers)
    if show_ylabels:
        ax.set_yticklabels(group_labels)
    else:
        ax.set_yticklabels([])
    # Faint vertical guide lines at the major x-ticks to help read box
    # positions across the figure. set_axisbelow keeps them behind the data.
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="gray", linestyle="-", linewidth=0.4, alpha=0.25)
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


def save_strip_legend(handles, fname, fig_w=12.0, fig_h=0.5):
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.legend(handles=handles, loc="center", ncol=len(handles),
               frameon=True, framealpha=1.0,
               handletextpad=0.5, columnspacing=2.0,
               borderpad=0.4, borderaxespad=0.0)
    plt.savefig(f"{OUT_DIR}/{fname}.png", dpi=200)
    plt.savefig(f"{OUT_DIR}/{fname}.pdf")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{fname}.png")


# ---------------------------------------------------------------------------
# In-process memoisation. Cache key includes mode so abs and rel reads are
# distinct entries. Disk cache lives in load_arrays at ~/.cache/iop_paper/.
# ---------------------------------------------------------------------------
_CACHE = {}


def get_hk(path, mode):
    p = resolve_path(path)
    if p is None:
        return None
    if ("hk", p, mode) not in _CACHE:
        print(f"  deriving hk[{mode}] from: {os.path.basename(p)}")
        _CACHE[("hk", p, mode)] = read_hk(p, mode)
    return _CACHE[("hk", p, mode)]


def get_dune(path, mode):
    p = resolve_path(path)
    if p is None:
        return None
    if ("dune", p, mode) not in _CACHE:
        print(f"  deriving dune[{mode}] from: {os.path.basename(p)}")
        _CACHE[("dune", p, mode)] = read_dune(p, mode)
    return _CACHE[("dune", p, mode)]


def get_obs(path, obs_kind, mode):
    """Generic per-observable getter. obs_kind in {hk_qe, dune_epi, dune_tpi}.
    Returns (x, w) or None if file unresolvable."""
    if obs_kind == "hk_qe":
        out = get_hk(path, mode)
        if out is None: return None
        x, w, _ = out
        return x, w
    out = get_dune(path, mode)
    if out is None: return None
    x_tpi, x_epi, w, _ = out
    return (x_tpi, w) if obs_kind == "dune_tpi" else (x_epi, w)


def get_obs_with_mode(path, obs_kind, mode):
    """Like get_obs but also returns the per-event Mode array.
    Used by the validation plot to highlight |Mode|==16."""
    if obs_kind == "hk_qe":
        out = get_hk(path, mode)
        if out is None: return None
        return out  # already (x, w, mode_arr)
    out = get_dune(path, mode)
    if out is None: return None
    x_tpi, x_epi, w, mode_arr = out
    return ((x_tpi, w, mode_arr) if obs_kind == "dune_tpi"
            else (x_epi, w, mode_arr))


def hk_or_dune_path(obs_kind, paths_hk, paths_dune):
    return paths_hk if obs_kind == "hk_qe" else paths_dune


# ---------------------------------------------------------------------------
# Generic figure builder. variants: list of (label, color, paths_per_obs)
# ---------------------------------------------------------------------------
OBS_GROUPS = [("hk_qe", "hk"),
              ("dune_epi", "dune_epi"),
              ("dune_tpi", "dune_tpi")]


def _make_figure(fname_stem, flavour, mode, variants, group_label_suffix=""):
    obs_keys = [k for k, _ in OBS_GROUPS]
    n_groups = len(obs_keys)
    rows_per_group = len(variants)
    # Row pitch (within a group), inter-group gap (between last row of group N
    # and first row of group N+1), and top/bottom axis margins. Designed so
    # within-group whitespace above the first row and below the last row of
    # each group is equal (= inter_group_gap / 2), eliminating the
    # asymmetric "lots of space below, none above" look.
    sub_pitch = 0.30
    inter_group_gap = 0.30
    top_margin = 0.20
    bottom_margin = 0.20
    group_pitch = (rows_per_group - 1) * sub_pitch + inter_group_gap
    half = sub_pitch / 2 * 0.85
    show_ylabels = (flavour == "numu")
    # Figure height scales with total data-y span plus a fixed allowance for
    # the x-axis label, ticks and any padding.
    data_span = ((n_groups - 1) * group_pitch
                 + (rows_per_group - 1) * sub_pitch
                 + top_margin + bottom_margin)
    fig, ax = plt.subplots(figsize=(4.0, data_span + 1.2))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.16)

    group_centers = []
    for gi, ok in enumerate(obs_keys):
        y0 = gi * group_pitch
        nbins, hrange = hist_params_for(ok, mode)
        for vi, v in enumerate(variants):
            path = v["paths"].get(ok)
            res = get_obs(path, ok, mode) if path else None
            if res is None:
                continue
            x, w = res
            draw_hybrid_hist(ax, y0 + vi * sub_pitch, half, x, w,
                             v["color"], nbins=nbins, hist_range=hrange)
        group_centers.append(y0 + (rows_per_group - 1) * sub_pitch / 2)

    # Dividers placed in the middle of the inter-group gap so the whitespace
    # above the next group's first row equals the whitespace below the
    # previous group's last row.
    for gi in range(1, n_groups):
        ax.axhline(gi * group_pitch - inter_group_gap / 2,
                   color="k", lw=0.4, alpha=0.3)

    ax.set_ylim((n_groups - 1) * group_pitch
                + (rows_per_group - 1) * sub_pitch + bottom_margin,
                -top_margin)

    xmin, xmax = XLIM[mode]
    ylabels = [OBS_LABELS[ok] + group_label_suffix for ok in obs_keys]
    setup_axes(ax, n_groups, _xlabel(mode), xmin, xmax, ylabels, group_centers,
               show_ylabels=show_ylabels)

    out = f"{fname_stem}_{flavour}_{mode}"
    plt.savefig(f"{OUT_DIR}/{out}.png", dpi=180)
    plt.savefig(f"{OUT_DIR}/{out}.pdf")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{out}.png")


# ---------------------------------------------------------------------------
# Fig 8 -- FSI vs noFSI, NuWro SF
# ---------------------------------------------------------------------------
def make_fsi_compare_figure(flavour, mode):
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
    _make_figure("Fig8_FSIvsNoFSI", flavour, mode, variants)


# ---------------------------------------------------------------------------
# Fig 9 -- piabs grid (kaskada_piN_abs_scale = 0.69 / 1.0 / 1.31)
# ---------------------------------------------------------------------------
def make_pi_abs_figure(flavour, mode):
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
    _make_figure("Fig9_PiAbs", flavour, mode, variants)


# ---------------------------------------------------------------------------
# Fig 10 -- NN_mfp grid (kaskada_NN_mfp_scale = 0.7 / 1.0 / 1.3)
# ---------------------------------------------------------------------------
def make_mfp_compare_figure(flavour, mode):
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
    _make_figure("Fig10_MFP", flavour, mode, variants)


# ---------------------------------------------------------------------------
# Fig 11 -- GENIE cascade tunes (G18_10a/b/c/d)
# ---------------------------------------------------------------------------
def make_cascade_compare_figure(flavour, mode):
    tunes = ["10a", "10b", "10c", "10d"]
    variants = [
        {"label": GENIE_LABELS[t], "color": CASCADE_COLORS[t], "paths": {
            "hk_qe":    GENIE_HK[(flavour, t)],
            "dune_epi": GENIE_DUNE[(flavour, t)],
            "dune_tpi": GENIE_DUNE[(flavour, t)],
        }}
        for t in tunes
    ]
    _make_figure("Fig11_GENIE", flavour, mode, variants)


# ---------------------------------------------------------------------------
# Fig 12 -- NEUT EDRMF vs RPWIA
# ---------------------------------------------------------------------------
def make_edrmf_compare_figure(flavour, mode):
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
    _make_figure("Fig12_EDRMF", flavour, mode, variants)


# ---------------------------------------------------------------------------
# Standalone legends (helpers used by main and importable)
# ---------------------------------------------------------------------------
def colour_swatch(color, label, alpha=HIST_FILL_ALPHA + 0.10):
    return Line2D([0], [0], marker="s", linestyle="", markersize=14,
                  markerfacecolor=color, markeredgecolor=color,
                  alpha=alpha, label=label)


quantity_handles_hybrid = [
    Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=BOX_LW,
              label=r"$1\sigma$ range (16-84\%)"),
    Line2D([0], [0], color="black", lw=1.0, alpha=WHISKER_ALPHA,
           label=r"90\% range (5-95\%)"),
    Line2D([0], [0], marker="o", linestyle="", markersize=10,
           markerfacecolor="white", markeredgecolor="black",
           markeredgewidth=1.4, label="median"),
    Line2D([0], [0], marker="D", linestyle="", markersize=6,
           markerfacecolor="white", markeredgecolor="black",
           markeredgewidth=1.4, label="mean"),
]


def _save_all_legends():
    """One combined legend per BW set — row 1 = metric markers, row 2 =
    variant colour key. Mode-agnostic (same legends for abs and rel)."""
    from FlatTreeMod import save_bw_legend
    BW_PAIR_WIDTH = 8.0
    save_bw_legend(
        quantity_handles_hybrid,
        [colour_swatch(COL_GREY,      "with FSI"),
         colour_swatch(COL_VERMILION, "no FSI")],
        OUT_DIR, "legend_Fig8_FSIvsNoFSI", fig_w=BW_PAIR_WIDTH,
    )
    save_bw_legend(
        quantity_handles_hybrid,
        [colour_swatch(COL_VERMILION, r"$\pi_{\rm abs}$ $+31\%$"),
         colour_swatch(COL_GREY,      "nominal"),
         colour_swatch(COL_BLUE,      r"$\pi_{\rm abs}$ $-31\%$")],
        OUT_DIR, "legend_Fig9_PiAbs", fig_w=BW_PAIR_WIDTH,
    )
    save_bw_legend(
        quantity_handles_hybrid,
        [colour_swatch(COL_VERMILION, r"$0.7\times$ NN MFP"),
         colour_swatch(COL_GREY,      "nominal"),
         colour_swatch(COL_BLUE,      r"$1.3\times$ NN MFP")],
        OUT_DIR, "legend_Fig10_MFP", fig_w=BW_PAIR_WIDTH,
    )
    save_bw_legend(
        quantity_handles_hybrid,
        [colour_swatch(c, GENIE_LABELS[t]) for t, c in CASCADE_COLORS.items()],
        OUT_DIR, "legend_Fig11_GENIE", fig_w=BW_PAIR_WIDTH,
    )
    save_bw_legend(
        quantity_handles_hybrid,
        [colour_swatch(COL_GREY,   "EDRMF"),
         colour_swatch(COL_ORANGE, "RPWIA")],
        OUT_DIR, "legend_Fig12_EDRMF", fig_w=BW_PAIR_WIDTH,
    )


# ---------------------------------------------------------------------------
# Variant catalogue — exposed for the validation/variation-table scripts.
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
        (GENIE_LABELS[t], CASCADE_COLORS[t], lambda obs, _t=t: _hk_or_dune(
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
    os.makedirs(OUT_DIR, exist_ok=True)
    for mode in ("abs", "rel"):
        print(f"\n##### mode = {mode} #####\n")
        for flavour in ("numu", "numubar"):
            print(f"=== {flavour}[{mode}] FSI vs noFSI ===")
            make_fsi_compare_figure(flavour, mode)
            print(f"=== {flavour}[{mode}] pi-abs ===")
            make_pi_abs_figure(flavour, mode)
            print(f"=== {flavour}[{mode}] NN_mfp ===")
            make_mfp_compare_figure(flavour, mode)
            print(f"=== {flavour}[{mode}] GENIE cascade tunes ===")
            make_cascade_compare_figure(flavour, mode)
            print(f"=== {flavour}[{mode}] NEUT EDRMF vs RPWIA ===")
            make_edrmf_compare_figure(flavour, mode)
    print("\n=== standalone legends ===")
    _save_all_legends()
    print("\nDONE")
