try:
    import uproot as up
except ImportError:
    Err("Uproot not installed. Cannot perform analysis.")

import warnings

try:
    import matplotlib.pyplot as plt
except ImportError:
    Err("Matplotlib not installed. Cannot create plots.")

import numpy as np
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

try:
    import scienceplots  # noqa: F401  # registers "science" style with matplotlib
    plt.style.use(["science", "notebook"])  #, "grid"]
except Exception:
    plt.style.use("default")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

def Log(string):
    print("\033[94m[LOG]\033[0m :: ", string)

def Warn(string):
    print("\033[93m[WARNING]\033[0m :: ", string)

def Err(string):
    print("\033[91m[ERROR]\033[0m :: ", string)

def Print(string):
    print("\033[92m[OUTPUT]\033[0m :: ", string)


# ----------------------------------------
# Colour schemes
# ----------------------------------------
light_blue = '#56B4E9'
medium_blue = '#0072B2'
dark_blue = '#084594'
light_red = '#E69F00'
dark_red = '#D55E00'
light_green = '#009E73'
dark_green = '#007D4B'
vivid_purple = '#CC79A7'
bright_yellow = '#F0E442'

# ----------------------------------------
# Load OscProb shared library
# ----------------------------------------
import os
import ROOT
# Try a sequence of candidate paths so the module imports both on the
# original developer's machine and on shared installs (lxplus etc.).
_OSCPROB_DEFAULT = (
    '/home/jm721/Desktop/PhDWork/FirstYear/Minoo_Project/UCM_codes/'
    'precomputed_tables/Graphs/Full_EDRMF_RNG_check/IOP_paper/OscProb/'
    'build/lib64/libOscProb.so'
)
_OSCPROB_CANDIDATES = []
if os.environ.get('OSCPROB_LIB'):
    _OSCPROB_CANDIDATES.append(os.environ['OSCPROB_LIB'])
if os.environ.get('OSCPROB'):
    _OSCPROB_CANDIDATES.append(os.path.join(os.environ['OSCPROB'], 'lib', 'libOscProb.so'))
_OSCPROB_CANDIDATES.append(_OSCPROB_DEFAULT)
for _cand in _OSCPROB_CANDIDATES:
    if os.path.exists(_cand):
        ROOT.gSystem.Load(_cand)
        break
else:
    Warn(f'OscProb library not found in any of {_OSCPROB_CANDIDATES}; '
         'oscillation features will fail at runtime.')

# ----------------------------------------
# Create global PMNS_Fast object
# ----------------------------------------
pmns = ROOT.OscProb.PMNS_Fast()

# ----------------------------------------
# Set oscillation parameters (PDG-ish)
# ----------------------------------------
theta12 = 0.583
theta13 = 0.149
theta23 = 0.857
# deltaCP = 3.44
deltaCP = -np.pi/2

dm21 = 7.41e-5
dm32 = 2.437e-3

pmns.SetMix(theta12, theta23, theta13, deltaCP)
pmns.SetDeltaMsqrs(dm21, dm32)

# ----------------------------------------
# Set T2K baseline (km)
# ----------------------------------------
L = 295.0
# L = 1285.0 // DUNE
pmns.SetPath(L, 2.8)   # 2.8 g/cm3 density for Earth

# ----------------------------------------
# Arrays for custom legend lines
# ----------------------------------------

custom_lines = []
labels = []

# ----------------------------------------
# Functions to oscillate given energy
# input: neutrino energy
# output: oscillation probability
# ----------------------------------------
# Flavour codes:
# 0: nue
# 1: numu
# 2: nutau
# ----------------------------------------

def OscProb_mumu(E_in: float) -> float:
    return pmns.Prob(1, 1, E_in, L)  # νμ → νμ

def OscProb_mue(E_in: float) -> float:
    return pmns.Prob(1, 0, E_in, L)  # νμ → νe


def plot_branch(ax_main, filename: str, kinematic: str, bin_width: float, label: str, color: str):
    infile = up.open(filename)
    TBranch_kinematic = infile["FlatTree_VARS;1"][kinematic].array()
    XSec_scale_factor = max(infile["FlatTree_VARS;1"]["fScaleFactor"].array())


    bins = np.arange(0, 10, step=bin_width)
    weights = XSec_scale_factor * np.ones_like(TBranch_kinematic) / bin_width

    counts, edges = np.histogram(TBranch_kinematic, bins=bins, weights=weights)
    sumw2, _      = np.histogram(TBranch_kinematic, bins=bins, weights=weights**2)

    errors  = np.sqrt(sumw2)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax_main.hist(TBranch_kinematic, bins=np.arange(0, 10, step=bin_width), histtype='step', weights=XSec_scale_factor*np.ones_like(TBranch_kinematic)/(bin_width), color=color,linewidth=1.2, label = label)
    # counts, edges  = np.histogram(TBranch_kinematic, bins=(np.arange(0, 10, step=bin_width)), weights=(XSec_scale_factor*np.ones_like(TBranch_kinematic)/(bin_width)))
    # errors = np.sqrt(np.histogram(TBranch_kinematic, bins=(np.arange(0, 10, step=bin_width)), weights=(XSec_scale_factor*np.ones_like(TBranch_kinematic)/(bin_width))**2)[0])
    # centers = 0.5 * (edges[1:] + edges[:-1])
    ax_main.errorbar(centers, counts, yerr=errors, color=color, linestyle='')
    ax_main.legend(loc = "upper right")

    # Create a matching line handle for legend
    custom_lines.append(Line2D([0], [0], color=color, lw=2))
    labels.append(label)
    return

# =============================================================================
# Shared paper style + plotting helpers (added 2026-05).
#
# Existing scripts still work unchanged. New scripts (and incremental
# refactors of the older ones) should call these helpers so the colour
# key, line styles, legend conventions and axis labels stay consistent
# across every figure in the paper.
# =============================================================================

PAPER_STYLE = dict(
    # Colours (Wong CB-friendly, mirroring the constants above).
    fsi_color   = dark_red,       # solid red  -> with FSI
    nofsi_color = dark_blue,      # solid blue -> no FSI
    nominal     = '#444444',      # dark grey  -> nominal / unweighted reference
    pi_with     = light_red,      # 'with charged-pi mass'   (full E for pi+/-)
    pi_without  = medium_blue,    # 'without charged-pi mass' (KE for pi+/-)
    cascade_colors = dict(
        G18_10a='#0072B2',  # hN
        G18_10b='#D55E00',  # hA
        G18_10c='#009E73',  # INCL
        G18_10d='#CC79A7',  # G4BC
    ),
    # Histogram drawing defaults.
    histtype    = 'step',
    lw          = 1.6,
    grid_alpha  = 0.25,
    zero_color  = '0.4',
    zero_ls     = '--',
    zero_lw     = 0.7,
    title_fs    = 11,
    legend_fs   = 9,
    # Axis labels (use these so HK/DUNE plots agree on naming).
    yaxis_density   = r'$\mathrm{d}\sigma/\mathrm{d}E$ [cm$^2$/nucleon/MeV]',
    yaxis_density_GeV = r'$\mathrm{d}\sigma/\mathrm{d}E$ [cm$^2$/nucleon/GeV]',
    yaxis_events    = 'Number of events',
    xaxis_bias_HK   = r'$E_\nu^{\mathrm{reco}} - E_\nu^{\mathrm{true}}$ [MeV]',
    xaxis_bias_DUNE = r'$E_\nu^{\mathrm{reco}} - E_\nu^{\mathrm{true}}$ [GeV]',
    xaxis_bias_QE_HK = r'$E_\nu^{\mathrm{QE}} - E_\nu^{\mathrm{true}}$ [MeV]',
)

PAPER_BINNING = dict(
    HK_bias_MeV   = (-1000., 1000., 10.),    # (xmin, xmax, width)
    DUNE_bias_GeV = (-3.,    1.,    0.04),
)


def paper_bins(kind):
    """Return numpy bin edges for a named binning ('HK_bias_MeV' or
    'DUNE_bias_GeV')."""
    xmin, xmax, w = PAPER_BINNING[kind]
    return np.arange(xmin, xmax + 1e-9, step=w)


def hist_paper(ax, x, *, bins, weights=None, kind=None, label=None,
               color=None, ls='-', lw=None):
    """Wrap ax.hist with the project default style.

    `kind` is one of {'fsi','nofsi','nominal','pi_with','pi_without'} (used
    only when `color` is None). Otherwise pass an explicit `color`.
    """
    if color is None:
        if kind in ('fsi','nofsi','nominal','pi_with','pi_without'):
            color = PAPER_STYLE[f'{kind}_color' if kind in ('fsi','nofsi') else kind]
        else:
            color = PAPER_STYLE['nominal']
    if lw is None:
        lw = PAPER_STYLE['lw']
    return ax.hist(
        x, bins=bins, weights=weights,
        histtype=PAPER_STYLE['histtype'],
        color=color, linewidth=lw, linestyle=ls, label=label,
    )


def style_axis(ax, *, kind='bias_HK', density=True, show_zero=True, title=None):
    """Apply a uniform axis style. `kind` is a key suffix into PAPER_STYLE
    (e.g. 'bias_HK', 'bias_DUNE'). Set `density=False` for raw event counts."""
    ax.grid(alpha=PAPER_STYLE['grid_alpha'])
    if show_zero:
        ax.axvline(0, color=PAPER_STYLE['zero_color'],
                   ls=PAPER_STYLE['zero_ls'], lw=PAPER_STYLE['zero_lw'])
    ax.set_xlabel(PAPER_STYLE[f'xaxis_{kind}'])
    if density:
        ylab = (PAPER_STYLE['yaxis_density'] if 'HK' in kind
                else PAPER_STYLE['yaxis_density_GeV'])
    else:
        ylab = PAPER_STYLE['yaxis_events']
    ax.set_ylabel(ylab)
    if title is not None:
        ax.set_title(title, fontsize=PAPER_STYLE['title_fs'])


def with_ratio_panel(figsize=(8, 6), height_ratios=(3, 1)):
    """Two-row figure: main plot on top, ratio panel below, sharing x.

    Returns (fig, ax_main, ax_ratio). The ratio panel comes pre-styled
    with a horizontal y=1 reference line and 'ratio to nominal' on the y-axis.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
    ax_main.tick_params(labelbottom=False)
    ax_ratio.axhline(1.0, color=PAPER_STYLE['zero_color'],
                     ls=PAPER_STYLE['zero_ls'], lw=PAPER_STYLE['zero_lw'])
    ax_ratio.set_ylabel('ratio to nominal')
    return fig, ax_main, ax_ratio


def hybrid_box_whisker(ax, y, x_data, weights, color, *,
                       hist_range, nbins, half=0.18,
                       fill_alpha=0.22, whisker_alpha=0.50,
                       box_lw=1.4, percentiles=(5, 16, 84, 95),
                       show_mean=True):
    """Hybrid histogram silhouette + 1-sigma box outline + faint 90% whiskers
    + median (circle) and weighted mean (diamond) markers.

    Single-row drawer; call it once per row of a multi-row comparison panel.
    `percentiles` is (whisker_lo, box_lo, box_hi, whisker_hi).
    """
    keep = (weights > 0) & np.isfinite(x_data)
    x = np.asarray(x_data)[keep]
    w = np.asarray(weights)[keep]
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    pos = (cw - 0.5 * ws) / cw[-1]
    p = lambda q: float(np.interp(q / 100.0, pos, xs))
    p_lo, p16, p84, p_hi = (p(q) for q in percentiles)
    median = p(50.0)
    mean = float(np.average(xs, weights=ws))

    counts, edges = np.histogram(xs, bins=nbins, range=hist_range, weights=ws)
    if counts.max() > 0:
        counts = counts / counts.max()
    xstep = np.empty(2 * len(counts))
    xstep[0::2] = edges[:-1]
    xstep[1::2] = edges[1:]
    ystep = np.repeat(counts, 2)
    poly_x = np.concatenate([xstep, xstep[::-1]])
    poly_y = np.concatenate([y - half * ystep, y + half * ystep[::-1]])
    ax.fill(poly_x, poly_y, color=color, alpha=fill_alpha, linewidth=0,
            zorder=2)

    bh = 0.65 * half
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((p16, y - bh), p84 - p16, 2 * bh,
                           fill=False, edgecolor=color, linewidth=box_lw,
                           zorder=4))
    ax.plot([p_lo, p16], [y, y], color=color, lw=1.0, alpha=whisker_alpha, zorder=3)
    ax.plot([p84, p_hi], [y, y], color=color, lw=1.0, alpha=whisker_alpha, zorder=3)
    ax.plot([p_lo] * 2, [y - bh / 3, y + bh / 3], color=color, lw=0.9,
            alpha=whisker_alpha, zorder=3)
    ax.plot([p_hi] * 2, [y - bh / 3, y + bh / 3], color=color, lw=0.9,
            alpha=whisker_alpha, zorder=3)

    ax.plot(median, y, 'o', color='white', markersize=5,
            markeredgecolor=color, markeredgewidth=1.4, zorder=6)
    if show_mean:
        ax.plot(mean, y, 'D', color='white', markersize=4.5,
                markeredgecolor=color, markeredgewidth=1.4, zorder=6)

    return dict(median=median, mean=mean, p_lo=p_lo, p16=p16, p84=p84, p_hi=p_hi)
