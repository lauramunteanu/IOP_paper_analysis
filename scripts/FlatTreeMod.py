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
# Font sizes are calibrated for IOP single-column print width (86 mm / 3.4").
# Each Fig*_plot.py uses make_fig / make_fig_ratio / make_fig_stacked (defined
# below) so figures are born at the right physical size and labels render at
# their pt size on the page (no shrink-to-fit downscaling).
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    # Force 'standard' (no per-fig content-box cropping) — overrides the
    # 'tight' that scienceplots' 'science' style sets. With 'tight', numu
    # vs numubar PDFs come out at different widths because their y-tick
    # label widths differ. With 'standard' the saved canvas is exactly
    # figsize and the make_fig* helpers' layout='constrained' fits labels
    # inside that canvas.
    "savefig.bbox": "standard",
    # Force scientific notation for very small/large numbers (without these,
    # matplotlib's autoscale can fail on differential-xsec values ~1e-42).
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 4),    # use sci notation only for |x| < 1e-3 or |x| >= 1e4 (so axes that go up to ~9999 stay plain)
})

def Log(string):
    print("\033[94m[LOG]\033[0m :: ", string)

def Warn(string):
    print("\033[93m[WARNING]\033[0m :: ", string)

def Err(string):
    print("\033[91m[ERROR]\033[0m :: ", string)

def Print(string):
    print("\033[92m[OUTPUT]\033[0m :: ", string)


# ----------------------------------------
# Colour schemes — Paul Tol "vibrant" palette (CB-friendly, softer than Wong).
# Legacy names (dark_red, dark_blue, ...) are aliased to the new palette so
# existing references resolve automatically.
# ----------------------------------------
tol_red     = '#CC3311'
tol_blue    = '#0077BB'
tol_orange  = '#EE7733'
tol_teal    = '#009988'
tol_magenta = '#EE3377'
tol_cyan    = '#33BBEE'
tol_grey    = '#BBBBBB'
tol_dark    = '#555555'

# Paul Tol "muted" palette — 8 CB-friendly colours for categorical
# breakdowns (e.g. by-mode, by-neutron-multiplicity stacked plots).
TOL_MUTED = [
    '#CC6677',  # rose
    '#332288',  # indigo
    '#DDCC77',  # sand
    '#117733',  # dark green
    '#88CCEE',  # light blue
    '#44AA99',  # teal
    '#999933',  # olive
    '#882255',  # wine
]

# Legacy aliases (don't break existing imports across the repo):
dark_red       = tol_red
dark_blue      = tol_blue
light_blue     = tol_cyan
medium_blue    = tol_blue
light_red      = tol_orange
light_green    = tol_teal
dark_green     = '#117733'    # Tol muted dark green (replaces older olive)
vivid_purple   = tol_magenta
bright_yellow  = '#F0E442'

# CB-friendly pastel red/blue used for +/- energy-shift overlays in the
# Fig1 spectrum plots. Red = Tol muted "rose"; blue = Wong's "sky blue"
# (slightly darker than Tol muted's #88CCEE for better legibility).
pastel_red     = '#CC6677'
pastel_blue    = '#56B4E9'

# Fig1 osc-parameter variation pair — both in the green/teal family with
# clear contrast. Tol muted's medium teal vs dark forest.
osc_inc_color  = '#44AA99'   # +0.4% Δm² or +20° δCP — medium teal
osc_dec_color  = '#117733'   # -0.4% Δm² or -20° δCP — dark forest

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
# Set oscillation parameters: PDG 2025 review Table 14.7, top-left section
# (Ref. [193] w/o SK-ATM & IC24, NO best-fit), δCP fixed to −π/2 by paper
# convention. Angles in radians; mass-squared splittings in eV².
#   sin²θ12 = 3.07e-1  → θ12 = 0.58784 rad (33.68°)
#   sin²θ13 = 2.195e-2 → θ13 = 0.14870 rad ( 8.52°)
#   sin²θ23 = 5.61e-1  → θ23 = 0.84649 rad (48.50°)
# ----------------------------------------
theta12 = 0.58784
theta13 = 0.14870
theta23 = 0.84649
deltaCP = -np.pi/2

dm21 = 7.49e-5
dm32 = 2.459e-3

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


# ----------------------------------------
# Vectorised per-event oscillation weights for use as histogram weights in
# the bias / spectrum plots that currently don't apply oscillation. Resets
# the global pmns object (path, mixing, mass-squared splittings) so callers
# don't need to manage state.
# ----------------------------------------
_BASELINE_KM = {'HK': 295.0, 'DUNE': 1285.0}
_EARTH_DENSITY = 2.8


# Canonical flux histograms used to generate every NUISFLAT sample in this
# paper — read once by `posc_flux_avg` to compute the flux-averaged osc
# probability used as the dσ/dE scale-factor correction (see derivation in
# Bias_level_study/validate_osc_fscalefactor.py).
_FLUX_FILES = {
    ('HK',   'FHC'): ('/eos/project-n/neutrino-generators/PublicFluxFiles/'
                      't2kflux_2016_plus250kA.root',  'enu_sk_numu'),
    ('HK',   'RHC'): ('/eos/project-n/neutrino-generators/PublicFluxFiles/'
                      't2kflux_2016_minus250kA.root', 'enu_sk_numub'),
    ('DUNE', 'FHC'): ('/eos/project-n/neutrino-generators/PublicFluxFiles/'
                      'histos_g4lbne_v3r5p10_QGSP_BERT_OfficialEngDesignSept2021'
                      '_neutrino_DUNEFD_fastmc.root',  'numu_flux'),
    ('DUNE', 'RHC'): ('/eos/project-n/neutrino-generators/PublicFluxFiles/'
                      'histos_g4lbne_v3r5p10_QGSP_BERT_OfficialEngDesignSept2021'
                      '_antineutrino_DUNEFD_fastmc.root', 'numubar_flux'),
}


def _polarity_from_flav(flav):
    """FHC (νμ source flux) for 'numu'/'nue'; RHC (ν̄μ source flux) for 'numubar'/'nuebar'."""
    return 'RHC' if flav.endswith('bar') else 'FHC'


_POSC_AVG_CACHE = {}


def posc_flux_avg(exp, flav, lep_pdg=13):
    """⟨P_osc⟩_Φ = ∫Φ(E)·P_osc(E) dE / ∫Φ(E) dE.

    Source flux is the canonical generator flux for ``(exp, FHC/RHC)`` derived
    from ``flav``. Oscillation channel is selected by ``lep_pdg``: 13 →
    νμ→νμ survival (anti-ν for RHC samples); 11 → νμ→νe appearance.

    Cached after first call per (exp, polarity, lep_pdg).
    """
    polarity = _polarity_from_flav(flav)
    key = (exp, polarity, lep_pdg)
    if key in _POSC_AVG_CACHE:
        return _POSC_AVG_CACHE[key]

    flux_path, hist_name = _FLUX_FILES[(exp, polarity)]
    f = ROOT.TFile.Open(flux_path)
    h = f.Get(hist_name)
    nb = h.GetNbinsX()
    centres_GeV = np.array([h.GetBinCenter(i + 1) for i in range(nb)], dtype=float)
    contents    = np.array([h.GetBinContent(i + 1) for i in range(nb)], dtype=float)
    f.Close()

    L_use = _BASELINE_KM[exp]
    pmns.SetPath(L_use, _EARTH_DENSITY)
    pmns.SetMix(theta12, theta23, theta13, deltaCP)
    pmns.SetDeltaMsqrs(dm21, dm32)
    is_nubar = (polarity == 'RHC')
    pmns.SetIsNuBar(is_nubar)
    src, dst = (1, 1) if lep_pdg == 13 else (1, 0)
    p_osc = np.array([pmns.Prob(src, dst, float(E), L_use) for E in centres_GeV])
    pmns.SetIsNuBar(False)   # reset global state

    avg = float((contents * p_osc).sum() / contents.sum())
    _POSC_AVG_CACHE[key] = avg
    return avg


def _osc_weights(Enu_true_GeV, filename, exp, lep_pdg):
    """Common machinery for osc_weights_mumu / osc_weights_mue. Sets pmns
    state correctly for the (exp, ν vs ν̄, channel) of the sample."""
    if exp is None:
        if filename is None:
            raise ValueError("either exp= or filename= must be supplied")
        exp, flav = detect_exp_flav(filename)
    else:
        flav = detect_exp_flav(filename)[1] if filename else 'numu'
    L_use = _BASELINE_KM[exp]
    pmns.SetPath(L_use, _EARTH_DENSITY)
    pmns.SetMix(theta12, theta23, theta13, deltaCP)
    pmns.SetDeltaMsqrs(dm21, dm32)
    is_nubar = (_polarity_from_flav(flav) == 'RHC')
    pmns.SetIsNuBar(is_nubar)
    src, dst = (1, 1) if lep_pdg == 13 else (1, 0)
    out = np.array([pmns.Prob(src, dst, float(E), L_use) for E in Enu_true_GeV])
    pmns.SetIsNuBar(False)   # reset global state
    return out


def osc_weights_mumu(Enu_true_GeV, filename=None, *, exp=None):
    """Per-event νμ→νμ (or ν̄μ→ν̄μ for RHC) survival probability array.
    ``exp`` is 'HK' or 'DUNE'; if omitted, detected from ``filename``.
    ν vs ν̄ is detected from ``filename`` (numubar/nuebar → ν̄)."""
    return _osc_weights(Enu_true_GeV, filename, exp, lep_pdg=13)


def osc_weights_mue(Enu_true_GeV, filename=None, *, exp=None):
    """Per-event νμ→νe (or ν̄μ→ν̄e) appearance probability array. Use on
    νe/ν̄e samples (generated with the νμ flux × νe σ)."""
    return _osc_weights(Enu_true_GeV, filename, exp, lep_pdg=11)


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


# =============================================================================
# IOP figure-size standards. Single column = 86 mm = 3.39"; double = 178 mm = 7.01".
# Sizing the figure at creation time (instead of letting LaTeX shrink an
# oversized canvas) keeps label pt size honest on the printed page.
# =============================================================================
# Aspect ratios deliberately leaner (~20% shorter than the prior set) so
# the figures pack more efficiently when placed in LaTeX subfigure grids
# at width=\linewidth -- previously the tall single_ratio canvas left
# whitespace above/below each subfigure when three were placed side by side.
FIG_SIZES = {
    'single':         (3.5, 2.2),    # one panel, single column
    'single_ratio':   (3.5, 3.2),    # main + ratio strip, single column (Fig3/4/7)
    'double':         (7.0, 2.6),    # one wide panel, double column
    'double_ratio':   (7.0, 3.7),    # main + ratio, double column
    'double_stacked': (7.0, 4.2),    # two stacked panels (Fig5 noFSI/FSI)
    'box':            (5.0, 7.0),    # tall, narrow box-and-whisker comparison (unchanged)
}


def make_fig(kind='single', **subplots_kw):
    """Single-panel figure sized for IOP publication. Returns (fig, ax).
    Uses layout='constrained' so labels fit inside the figsize canvas
    without changing its dimensions (so paired numu/numubar PDFs are
    pixel-identical in size)."""
    subplots_kw.setdefault('layout', 'constrained')
    return plt.subplots(figsize=FIG_SIZES[kind], **subplots_kw)


def make_fig_ratio(kind='single_ratio', height_ratios=(3, 1), hspace=0.05):
    """Two-row main + ratio figure with shared x. Returns (fig, (ax_main, ax_ratio)).
    Uses layout='constrained' for stable canvas dimensions."""
    return plt.subplots(
        2, 1, sharex=True, figsize=FIG_SIZES[kind], layout='constrained',
        gridspec_kw={'height_ratios': list(height_ratios), 'hspace': hspace},
    )


def make_fig_stacked(kind='double_stacked', sharex=True, sharey=False, hspace=0.08):
    """Two stacked panels (rows) figure. Returns (fig, (ax_top, ax_bot)).
    Uses layout='constrained' for stable canvas dimensions."""
    return plt.subplots(
        2, 1, sharex=sharex, sharey=sharey, figsize=FIG_SIZES[kind],
        layout='constrained',
        gridspec_kw={'hspace': hspace},
    )


def save_strip_legend(handles, out_dir, fname, fig_w=7.0, fig_h=0.5,
                      ncol=None, frameon=False):
    """Save a thin horizontal-strip legend, no title. Designed to drop above
    a side-by-side pair of subfigures in a LaTeX figure with
    \\includegraphics[width=\\linewidth]{out_dir/fname.pdf}.
    Default frameon=False (no border). Writes both .png (200 dpi) and .pdf.

    ``bbox_inches="tight"`` crops the canvas to the actual legend extent so
    the LaTeX subfigure that includes this PDF gets no extra horizontal
    padding (otherwise the wide ``fig_w`` would leave whitespace on both
    sides of the centred legend)."""
    if ncol is None:
        ncol = len(handles)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.legend(handles=handles, loc="center", ncol=ncol,
               frameon=frameon, framealpha=1.0,
               handletextpad=0.5, columnspacing=2.0,
               borderpad=0.1, borderaxespad=0.0)
    plt.savefig(f"{out_dir}/{fname}.png", dpi=200,
                bbox_inches="tight", pad_inches=0.02)
    plt.savefig(f"{out_dir}/{fname}.pdf",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    Print(f"saved {out_dir}/{fname}.pdf")


def save_spectra_legend(nominal_handle, osc_handles, shift_handles,
                        out_dir, fname, fig_w=7.0, fig_h=0.7):
    """3-column legend for the Fig1 spectrum plots. Column 1 = the single
    'Nominal ...' entry centered vertically; column 2 = the 2 osc-parameter
    variants stacked; column 3 = the 2 energy-shift variants stacked.
    No frame. Centered around (0.20 / 0.55 / 0.85) figure-x. ``bbox_inches=
    "tight"`` on savefig crops the canvas to actual legend extent so the
    LaTeX subfigure that includes this PDF gets no extra horizontal padding."""
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    leg1 = ax.legend(handles=[nominal_handle], loc="center",
                     bbox_to_anchor=(0.20, 0.5),
                     frameon=False, handletextpad=0.5,
                     borderpad=0.1, borderaxespad=0.0)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=osc_handles, loc="center",
                     bbox_to_anchor=(0.55, 0.5), ncol=1,
                     frameon=False, handletextpad=0.5,
                     borderpad=0.1, borderaxespad=0.0)
    ax.add_artist(leg2)
    ax.legend(handles=shift_handles, loc="center",
              bbox_to_anchor=(0.85, 0.5), ncol=1,
              frameon=False, handletextpad=0.5,
              borderpad=0.1, borderaxespad=0.0)
    plt.savefig(f"{out_dir}/{fname}.png", dpi=200,
                bbox_inches="tight", pad_inches=0.02)
    plt.savefig(f"{out_dir}/{fname}.pdf",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    Print(f"saved {out_dir}/{fname}.pdf")


def save_bw_legend(metric_handles, color_handles, out_dir, fname,
                   fig_w=7.0, fig_h=1.0):
    """Combined BW legend: row 1 = metric markers, row 2 = variant colours,
    both centered horizontally, no frame."""
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    leg1 = ax.legend(handles=metric_handles, loc="center",
                     bbox_to_anchor=(0.5, 0.75),
                     ncol=len(metric_handles),
                     frameon=False, handletextpad=0.5, columnspacing=2.0)
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, loc="center",
              bbox_to_anchor=(0.5, 0.25),
              ncol=len(color_handles),
              frameon=False, handletextpad=0.5, columnspacing=2.0)
    plt.savefig(f"{out_dir}/{fname}.png", dpi=200)
    plt.savefig(f"{out_dir}/{fname}.pdf")
    plt.close(fig)
    Print(f"saved {out_dir}/{fname}.pdf")


# Global scale for dσ/dE histograms. Multiply weights by DSIGMA_DE_SCALE
# before histogramming and label the axis with DSIGMA_DE_LABEL — this works
# around matplotlib's autoscale failure on raw values ~1e-42 cm²/nucleon/MeV.
DSIGMA_DE_SCALE = 1.0e42
DSIGMA_DE_LABEL = r"$\mathrm{d}\sigma/\mathrm{d}E$ [10$^{-42}$ cm$^2$/nucleon/MeV]"


# =============================================================================
# Two scaling modes:
#   - bias plots             -> dσ/dE   (cm²/nucleon/MeV × 10⁴²)
#   - Enu-spectrum plots     -> Events/MeV (integrate to EXPECTED_EVENTS)
# Use the helper that matches the plot kind. Don't mix.
# =============================================================================

EXPECTED_EVENTS = {
    # DUNE — IOP-paper canonical normalisations (2026-05-15). The 624 / 336
    # factor scales the supplied per-336-d-equivalent numbers to the 624 d
    # exposure used in the paper.
    ('DUNE', 'numu'):    7235 * 624 / 336.0,
    ('DUNE', 'numubar'): 2656 * 624 / 336.0,
    ('DUNE', 'nue'):     1395 * 624 / 336.0,
    ('DUNE', 'nuebar'):   164 * 624 / 336.0,
    # HK — IOP-paper canonical normalisations (2026-05-15).
    ('HK',   'numu'):     8845.1,
    ('HK',   'numubar'): 12027.2,
    ('HK',   'nue'):      2474.7,
    ('HK',   'nuebar'):   1542.7,
}

EVENT_RATE_LABEL = r"Events / bin"


def detect_exp_flav(filename):
    """Auto-detect ('DUNE'|'HK', 'numu'|'numubar'|'nue'|'nuebar') from a path."""
    fn = filename.lower()
    exp = 'DUNE' if 'dune' in fn else 'HK'
    if 'nuebar' in fn:
        flav = 'nuebar'
    elif 'nue' in fn:
        flav = 'nue'
    elif 'numubar' in fn or '/dune_numub_' in fn or '_numub_' in fn:
        flav = 'numubar'
    else:
        flav = 'numu'
    return exp, flav


def make_weights_dxsec(arr, bin_width, fScaleFactor=None):
    """Per-event constant weight for differential cross-section bias plots.
    Returns fScaleFactor * DSIGMA_DE_SCALE / bin_width — multiply by
    np.ones_like(observable) to get the per-event array. Y-axis is
    dσ/dE [10⁻⁴² cm²/nucleon/MeV] (use DSIGMA_DE_LABEL).
    """
    if fScaleFactor is None:
        fScaleFactor = float(np.max(arr['fScaleFactor']))
    return fScaleFactor * DSIGMA_DE_SCALE / bin_width


def bias_sel_arr(arr, observable, *, vertex=False, lep_pdg=13):
    """Return the boolean selection mask that ``bias_arr(arr, observable,
    ...)`` uses internally. Lets callers filter ``arr['Enu_true']`` or any
    other branch the same way the bias values were filtered — needed to
    compute per-event oscillation weights of length matching ``bias_arr``."""
    if observable == "qe":
        return is_cc0pi_arr(arr, vertex=vertex, lep_pdg=lep_pdg)
    if observable in ("had", "avail"):
        # Match enu_had_arr's cc_mask: arr['cc'] & |PDGLep|==lep_pdg.
        return (np.asarray(arr['cc'], dtype=bool)
                & (np.abs(np.asarray(arr['PDGLep'])) == lep_pdg))
    raise ValueError(f"unknown observable {observable!r}")


def weighted_quantile(values, quantiles, weights=None):
    """Weighted quantile(s) of ``values``. Matches ``np.quantile`` default
    (linear interpolation) when ``weights`` is None or uniform. ``quantiles``
    is a scalar or array of probabilities in [0, 1]. Returns same shape as
    ``quantiles``."""
    values    = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if weights is None:
        return np.quantile(values, quantiles)
    weights = np.asarray(weights, dtype=float)
    order   = np.argsort(values)
    values  = values[order]
    weights = weights[order]
    cum = np.cumsum(weights) - 0.5 * weights
    total = cum[-1] + 0.5 * weights[-1]
    if total <= 0:
        return np.quantile(values, quantiles)
    cum /= total
    return np.interp(quantiles, cum, values)


def make_weights_dxsec_osc(arr, bin_width, observable, filename,
                            *, vertex=False, lep_pdg=13, fScaleFactor=None):
    """Per-event weight = fScaleFactor × DSIGMA_DE_SCALE / bin_width
                           × P_osc(E_true) / ⟨P_osc⟩_Φ.

    Length matches ``bias_arr(arr, observable, vertex=vertex, lep_pdg=lep_pdg)``.

    Oscillation channel chosen by ``lep_pdg``: 13 (default) → νμ→νμ
    survival on a νμ/ν̄μ sample; 11 → νμ→νe appearance on a νe/ν̄e sample.
    ``filename`` is used to auto-detect (HK, DUNE) and (FHC, RHC) from the
    file name — driving baseline, ν vs ν̄ NuBar flag, and the flux histogram
    used to compute ⟨P_osc⟩_Φ.

    The 1/⟨P_osc⟩_Φ factor renormalises the (per-event) NUISANCE fScaleFactor
    — which was computed against the unoscillated flux — to be valid for the
    oscillated flux. Validated end-to-end in
    Bias_level_study/validate_osc_fscalefactor.py:
    matched-target (H₂O) ratio of rescaled-unosc'd vs osc-flux-generated runs
    integrates to 1.006 over [200, 2000] MeV, with ~6% bin-by-bin spread that
    is consistent with the limited stats of the 100 k-event validation
    sample.

    The resulting y-axis IS a proper differential cross-section dσ/dE in
    [10⁻⁴² cm²/nucleon/MeV] after osc reweighting — same units as
    ``make_weights_dxsec``."""
    sel = bias_sel_arr(arr, observable, vertex=vertex, lep_pdg=lep_pdg)
    Enu_t_sel = np.asarray(arr['Enu_true'])[sel]
    osc_fn = osc_weights_mue if lep_pdg == 11 else osc_weights_mumu
    osc_w = osc_fn(Enu_t_sel, filename=filename)
    if fScaleFactor is None:
        fScaleFactor = float(np.max(arr['fScaleFactor']))
    exp, flav = detect_exp_flav(filename)
    avg = posc_flux_avg(exp, flav, lep_pdg=lep_pdg)
    return fScaleFactor * DSIGMA_DE_SCALE / bin_width * osc_w / avg


def make_weights_event_rate(arr, filename, bin_width=None):
    """Per-event constant weight for Enu-spectrum plots scaled to expected
    event yield. Returns target / N_gen — caller multiplies by
    np.ones_like(observable). Y-axis is 'Events / bin' (EVENT_RATE_LABEL).
    Histogram bins integrate to the expected event yield. For oscillated
    spectra, multiply this scalar by the per-event probability array before
    passing to ax.hist as weights. The `bin_width` argument is unused and
    kept only for backwards compatibility with older callers.
    """
    del bin_width  # unused — kept for API compat
    exp, flav = detect_exp_flav(filename)
    target = EXPECTED_EVENTS[(exp, flav)]
    n_gen  = len(arr['Enu_true'])
    return target / n_gen


def expected_events(filename, channel=None):
    """Lookup EXPECTED_EVENTS using auto-detected (exp, flav) from filename,
    or override the flavour with ``channel`` (for oscillated spectra where
    the target flavour differs from the source: e.g., νμ→νe at HK uses
    EXPECTED_EVENTS[('HK','nue')])."""
    exp, flav = detect_exp_flav(filename)
    if channel is not None:
        flav = channel
    return EXPECTED_EVENTS[(exp, flav)]


def paper_bins(kind):
    """Return numpy bin edges for a named binning ('HK_bias_MeV' or
    'DUNE_bias_GeV')."""
    xmin, xmax, w = PAPER_BINNING[kind]
    return np.arange(xmin, xmax + 1e-9, step=w)


# =============================================================================
# Pre/post-FSI particle access. NUISFLAT trees carry both:
#   post-FSI:  nfsp,   pdg,      E,      px,      py,      pz
#   vertex:    nvertp, pdg_vert, E_vert, px_vert, py_vert, pz_vert
#
# IMPORTANT: vertex=True does NOT give a clean "noFSI" view. NuWro applies a
# binding-energy / separation-energy correction (Ef + kaskada_w, ~25-40 MeV
# per outgoing nucleon) at cascade exit, which means vertex and post-FSI
# energies disagree by tens of MeV per nucleon even when no rescattering
# occurred. For "noFSI" comparisons in published plots, ALWAYS load the
# corresponding noFSI sample file (use noFSI_path(...)) and call with
# vertex=False. vertex=True is kept for diagnostics / particle-stack
# inspection only.
# =============================================================================


def noFSI_path(filename):
    """Map a *_FSI.flat.root path to its *_noFSI.flat.root sibling."""
    if "_FSI.flat.root" not in filename:
        raise ValueError(f"Expected *_FSI.flat.root path, got: {filename}")
    return filename.replace("_FSI.flat.root", "_noFSI.flat.root")

def particles(tree, *, vertex=False):
    """Return (n, pdg, E, px, py, pz) arrays from a NUISFLAT entry."""
    if vertex:
        return (tree.nvertp, tree.pdg_vert, tree.E_vert,
                tree.px_vert, tree.py_vert, tree.pz_vert)
    return (tree.nfsp, tree.pdg, tree.E, tree.px, tree.py, tree.pz)


def is_cc0pi(tree, *, vertex=False):
    """CC + no charged-pi/pi0 in the chosen particle set. Computed directly
    from the pdg stack (post-FSI for vertex=False, pdg_vert for vertex=True)
    so the helper works on any NUISFLAT file regardless of whether the
    generator wrote a `flagCC0pi` branch."""
    if not tree.cc:
        return False
    n, pdg, _, _, _, _ = particles(tree, vertex=vertex)
    for j in range(n):
        a = abs(int(pdg[j]))
        if a == 211 or a == 111:
            return False
    return True


# =============================================================================
# Uproot + pickle cache.
# Replaces the slow PyROOT GetEntry loop. ``load_arrays(filename, branches)``
# returns a dict of awkward/numpy arrays for the requested branches and caches
# the result on disk (keyed on file mtime + branch list) so subsequent runs
# read from pickle in <1 s instead of re-decoding the ROOT tree.
#
# Vectorised CC0pi / particle helpers operate on those arrays directly.
# =============================================================================

import os as _os
import hashlib as _hashlib
import pickle as _pickle

try:
    import uproot as _uproot
    import awkward as _ak
except ImportError:
    _uproot = None
    _ak = None
    Warn("uproot/awkward not importable; load_arrays() will not work.")

# Public aliases so `from FlatTreeMod import *` brings them in
ak = _ak
uproot = _uproot

# Cache directory — defaults to the project-n shared cache so everyone
# running these scripts shares the same load_arrays() pickles. Home-EOS
# has a tighter quota and ran out during the 10M-stats regen; project-n
# has 1.7 PB free and the cache is generically useful for anyone touching
# the IOP-paper figures. Override with $IOP_PAPER_CACHE if needed.
_CACHE_DIR = _os.environ.get(
    'IOP_PAPER_CACHE',
    '/eos/project-n/neutrino-generators/iop_paper_cache'
)
_os.makedirs(_CACHE_DIR, exist_ok=True)

# Branches commonly used by every Fig*.py. Pulling them all in one pass means
# the cache is reusable across figs that read the same file.
DEFAULT_BRANCHES = (
    'Enu_true', 'Enu_QE', 'ELep', 'CosLep', 'PDGLep',
    'Mode', 'cc', 'fScaleFactor',
    # flagCC0pi removed -- we derive CC0π from the pdg stack inside
    # is_cc0pi / is_cc0pi_arr so the same code path works on GENIE NUISFLAT
    # files (which don't write that branch).
    'nfsp',   'pdg',      'E',      'px',      'py',      'pz',
    'nvertp', 'pdg_vert', 'E_vert', 'px_vert', 'py_vert', 'pz_vert',
    'ninitp',
)



# ---------------------------------------------------------------------------
# E_nu^QE per FSI IOP paper Eq. 9
# Numerator   : m_p^2 - m_l^2 - (m_n - Eb)^2 + 2 E_l (m_n - Eb)
# Denominator : 2 (m_n - Eb - E_l + p_l^z)
# For antineutrino interactions m_p and m_n are swapped.
# Eb = 27 MeV (oxygen) per the paper.
# ---------------------------------------------------------------------------
M_PROTON_GeV  = 0.938272
M_NEUTRON_GeV = 0.939565
M_MUON_GeV    = 0.105658
M_ELECTRON_GeV = 0.000511
EB_IOP_GeV    = 0.027

def compute_enu_qe(arr, eb=EB_IOP_GeV):
    """Reconstructed Enu_QE per the IOP paper. Returns array in GeV.
    Requires arr to contain ELep (GeV), CosLep, PDGLep."""
    pdgl = arr['PDGLep']
    EL   = arr['ELep']
    cosL = arr['CosLep']
    abs_pdgl = np.abs(pdgl)
    m_l = np.where(abs_pdgl == 13, M_MUON_GeV, M_ELECTRON_GeV)
    is_antinu = pdgl < 0
    M_init  = np.where(is_antinu, M_PROTON_GeV,  M_NEUTRON_GeV)
    M_final = np.where(is_antinu, M_NEUTRON_GeV, M_PROTON_GeV)
    pl  = np.sqrt(np.maximum(EL*EL - m_l*m_l, 0.0))
    plz = pl * cosL
    Em  = M_init - eb
    num = M_final*M_final - m_l*m_l - Em*Em + 2.0*Em*EL
    den = 2.0*(Em - EL + plz)
    safe = np.where(np.abs(den) > 1e-9, den, 1e-9)
    return num / safe


# Bump this whenever the post-load array transformation changes so old caches
# are not re-used. The cache key includes this string.
_CACHE_SCHEMA = 'iop_enuqe_v1'

def _cache_key(filename, branches, max_events):
    h = _hashlib.sha1()
    try:
        st = _os.stat(filename)
        h.update(f"{filename}|{st.st_size}|{int(st.st_mtime)}".encode())
    except OSError:
        h.update(filename.encode())
    h.update(",".join(sorted(branches)).encode())
    h.update(str(max_events).encode())
    h.update(_CACHE_SCHEMA.encode())
    return h.hexdigest()


USE_10M_STATS = True
"""Module-level toggle: when True, any single-file NuWro morestats path
passed to load_arrays() is transparently rewritten to its 10M-stats glob
equivalent and dispatched to load_arrays_glob() (which applies the 1/N
fScaleFactor scaling). EDRMF / RPWIA / GENIE paths do not match the
rewrite pattern, so they fall through unchanged.

Set FlatTreeMod.USE_10M_STATS = False at the top of a script to revert
to the 1M morestats samples for that run."""


def _maybe_rewrite_to_10M(filename):
    """Return the high_stats_10M glob equivalent of a morestats single-file
    NuWro path, or None if the path doesn't match the rewrite pattern."""
    if not USE_10M_STATS:
        return None
    if 'nuwro_25031_morestats' not in filename:
        return None
    # NEUT / RPWIA / EDRMF / GENIE samples are NOT in high_stats_10M.
    if any(tok in filename for tok in ('EDRMF', 'RPWIA', 'NEUT_', 'GENIE', 'genie')):
        return None
    if '*' in filename or '?' in filename:
        # Already a glob — caller knows what it's doing, don't double-rewrite.
        return None
    return filename.replace(
        'nuwro_25031_morestats', 'high_stats_10M'
    ).replace('.flat.root', '_*.flat.root')


def load_arrays(filename, branches=DEFAULT_BRANCHES, max_events=None,
                tree_name='FlatTree_VARS'):
    """Load (cached) NUISFLAT branches from `filename` as awkward arrays.

    Returns a dict ``{branch_name: awkward_array}``. Cached on disk under
    ~/.cache/iop_paper/ keyed on (file mtime, size, branch list, max_events).
    Subsequent calls return the cache instead of reading the ROOT file.

    If `USE_10M_STATS` is True and the input is a NuWro morestats path,
    the call is transparently redirected to load_arrays_glob() on the
    corresponding 10-chunk high_stats_10M glob (with 1/N fScaleFactor
    scaling). NEUT / GENIE paths are unaffected.
    """
    rewritten = _maybe_rewrite_to_10M(filename)
    if rewritten is not None:
        return load_arrays_glob(rewritten, branches=branches,
                                max_events=max_events, tree_name=tree_name)
    if _uproot is None:
        raise RuntimeError("uproot not available")
    branches = tuple(branches)
    cache_path = _os.path.join(_CACHE_DIR,
                               _cache_key(filename, branches, max_events) + '.pkl')
    if _os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            try:
                return _pickle.load(f)
            except Exception:
                Warn(f"corrupt cache, re-reading: {cache_path}")
    Print(f"Reading (uproot): {filename}")
    with _uproot.open(filename) as fin:
        tree = fin[tree_name]
        # Filter out branches not present in this tree (e.g. GENIE NUISFLAT
        # files don't carry `flagCC0pi`). The cache key is built from the
        # resolved subset, so a NUISFLAT-derived file gets its own cache
        # entry distinct from a NuWro-derived one.
        available = set(tree.keys())
        kept = tuple(b for b in branches if b in available)
        if len(kept) != len(branches):
            missing = [b for b in branches if b not in available]
            Warn(f"{filename}: skipping branches not in tree: {missing}")
            cache_path = _os.path.join(_CACHE_DIR,
                                       _cache_key(filename, kept, max_events) + '.pkl')
            if _os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    try:
                        return _pickle.load(f)
                    except Exception:
                        Warn(f"corrupt cache, re-reading: {cache_path}")
        kw = dict(library='ak')
        if max_events is not None and max_events >= 0:
            kw['entry_stop'] = int(max_events)
        arrays = tree.arrays(list(kept), **kw)
    out = {b: arrays[b] for b in kept}
    # Override Enu_QE with the IOP-paper formula (Eq. 9, Eb = 27 MeV) when
    # the necessary branches are present. NUISANCE's precomputed Enu_QE may
    # use a different binding energy or omit the antineutrino mass swap.
    if all(b in out for b in ('ELep', 'CosLep', 'PDGLep')):
        out['Enu_QE'] = compute_enu_qe(out)
    try:
        with open(cache_path, 'wb') as f:
            _pickle.dump(out, f, protocol=_pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        Warn(f"could not write cache {cache_path}: {e}")
    return out


def load_arrays_multifile(filenames, branches=DEFAULT_BRANCHES, max_events=None,
                          tree_name='FlatTree_VARS', scale_fScaleFactor=True):
    """Load and concatenate NUISFLAT branches from multiple files.

    Use this when one conceptual sample is split across N NUISFLAT files —
    e.g. the 10M-stats production at /eos/project-n/.../high_stats_10M/
    where each variant is 10 × 1M-event chunks.

    Each file is loaded individually (cached) via ``load_arrays`` and the
    awkward arrays are concatenated along the event axis. Returns a dict
    with the same keys as ``load_arrays``.

    ``scale_fScaleFactor`` (default True): divides the combined
    fScaleFactor by len(filenames). NUISANCE writes fScaleFactor per file
    as if that file were the entire production, so naively summing N
    chunks would over-count the flux normalisation by N. With scaling on,
    a 10-chunk 10M sample reproduces the event-rate prediction of any one
    1M chunk (within ~1/√N stats).

    Pass a single string for convenience; that path delegates to
    ``load_arrays`` unchanged.
    """
    if isinstance(filenames, str):
        return load_arrays(filenames, branches=branches, max_events=max_events,
                           tree_name=tree_name)
    files = list(filenames)
    if not files:
        raise ValueError("filenames is empty")
    if len(files) == 1:
        return load_arrays(files[0], branches=branches, max_events=max_events,
                           tree_name=tree_name)

    per_file = [load_arrays(f, branches=branches, max_events=max_events,
                            tree_name=tree_name) for f in files]
    # Intersect keys across files (a missing-branch file shouldn't crash
    # the concat — drop the key for everyone).
    keys = set(per_file[0].keys())
    for d in per_file[1:]:
        keys &= set(d.keys())

    combined = {}
    for key in sorted(keys):
        combined[key] = ak.concatenate([d[key] for d in per_file], axis=0)

    if scale_fScaleFactor and 'fScaleFactor' in combined:
        combined['fScaleFactor'] = combined['fScaleFactor'] / float(len(files))
    return combined


def load_arrays_glob(glob_pattern, **kwargs):
    """Convenience: glob → sorted file list → load_arrays_multifile.

    Example
    -------
    >>> arr = load_arrays_glob(
    ...     "/eos/project-n/neutrino-generators/generatorOutput/"
    ...     "FSIIOPPaperinputs/high_stats_10M/HK/HK_numu_FSI_*.flat.root")
    """
    import glob as _glob
    files = sorted(_glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"no files matched {glob_pattern!r}")
    return load_arrays_multifile(files, **kwargs)


def particles_arr(arr, *, vertex=False):
    """Vectorised counterpart of `particles()`. Given the dict from
    load_arrays, returns (n, pdg, E, px, py, pz) jagged arrays.

    For vertex=True, the initial-state entries (incoming neutrino + target
    nucleon, the first `ninitp` entries of each event's vertex stack) are
    stripped so the returned arrays contain only the outgoing pre-FSI
    particles. This is what's wanted for hadronic-energy / CC0π scans
    that should not double-count the initial state."""
    if vertex:
        # Per-particle mask: keep entries whose local index is >= ninitp.
        keep = ak.local_index(arr['pdg_vert'], axis=1) >= arr['ninitp']
        pdg = arr['pdg_vert'][keep]
        return (
            ak.sum(keep, axis=1),
            pdg,
            arr['E_vert'][keep],
            arr['px_vert'][keep],
            arr['py_vert'][keep],
            arr['pz_vert'][keep],
        )
    return (arr['nfsp'], arr['pdg'], arr['E'],
            arr['px'], arr['py'], arr['pz'])


def is_cc0pi_arr(arr, *, vertex=False, lep_pdg=13):
    """Vectorised CC0pi mask. Returns a 1D numpy bool array of length nevents.

    Both branches now compute the selection directly from the particle stack
    (`pdg` for vertex=False / post-FSI; `pdg_vert` for vertex=True / pre-FSI).
    The post-FSI branch used to read the pre-computed `flagCC0pi` field but
    GENIE NUISFLAT files don't carry it -- computing from `pdg` works for any
    NUISFLAT-format file regardless of generator.

    ``lep_pdg`` selects the expected final-state lepton: 13 (μ±, default) for
    νμ/ν̄μ samples, 11 (e±) for νe/ν̄e appearance samples. The selection
    requires the **first** particle in the final-state stack to have
    ``|pdg|==lep_pdg`` -- this is the literal-first-particle convention also
    used by the Fig 6 mis-reco cut (drop events where the primary lepton was
    overwritten by a hard radiated γ→e+e-). Differs from NUISFLAT's ``PDGLep``
    branch, which records the *highest-energy* charged lepton and so can
    disagree on rare DIS-with-hard-radiation events; this stricter criterion
    removes the resulting +tail. Impact on a νμ/ν̄μ NuWro file: ~0.004% of
    CC events.
    """
    cc = np.asarray(arr['cc'], dtype=bool)
    pdg_field = 'pdg_vert' if vertex else 'pdg'
    apdg = abs(arr[pdg_field])
    has_chpi = ak.any(apdg == 211, axis=1)
    has_pi0  = ak.any(apdg == 111, axis=1)
    no_pi = ~ak.to_numpy(has_chpi | has_pi0)
    if vertex:
        # The vertex stack is laid out as [incoming ν, target, outgoing lepton,
        # hadrons...]; the outgoing-lepton position varies per channel.  Use
        # NUISANCE's PDGLep scalar (the post-FSI primary lepton) — it agrees
        # with the vertex-stack primary lepton for the events we care about.
        is_target_lep = np.abs(np.asarray(arr['PDGLep'])) == lep_pdg
    else:
        # Post-FSI stack: position 0 is the primary lepton (matches the Fig 6
        # mis-reco cut convention pdg[:, 0] == 13 / 11).
        first_pdg = ak.to_numpy(ak.fill_none(ak.firsts(apdg), -1))
        is_target_lep = first_pdg == lep_pdg
    return cc & no_pi & is_target_lep


def diff_enu_qe_arr(arr, *, vertex=False, scale_to_MeV=True):
    """Vectorised (Enu_QE - Enu_true) for events passing CC0pi selection.
    Returns a numpy array (in MeV by default)."""
    sel = is_cc0pi_arr(arr, vertex=vertex)
    diff = ak.to_numpy(arr['Enu_QE'] - arr['Enu_true'])[sel]
    return diff * 1000.0 if scale_to_MeV else diff


def enu_had_arr(arr, *, vertex=False, lep_pdg=13):
    """Vectorised hadronic-energy reconstruction matching NUISANCE
    ``GetErecoil_MINERvA_LowRecoil`` for the hadronic part.

    Returns ``(bias_wo, bias_with, valid_mask)`` *filtered to CC events*:
      - bias_wo   = E_ν^reco(no pion-mass subtraction)           − Enu_true
      - bias_with = E_ν^reco(π± with full E, p still kinetic)    − Enu_true
      - valid_mask is the CC selection AND ``|PDGLep|==lep_pdg``.

    Per-particle contribution (everything else contributes 0):
      proton    (2212):              T = E − m
      π±        (211):               T = E − m   (def 1, "no π mass")
                                     E           (def 2, "with π mass")
      π0        (111):               E
      e±        (11):                E
      γ         (22):                E
      d/t/α     (1000010020,
                 1000010030,
                 1000020040):        T = E − m   (both definitions)
      neutrons / heavier nuclei / |pdg|>3000 / strange / etc.: skipped

    Light nuclear fragments (d, t, α) are treated like protons: kinetic
    energy only, in both ``add_wo`` and ``add_with``. Their rest mass is
    bound nuclear binding and not detected calorimetrically. Restoring
    them lifts GENIE G18_10c E_had/E_avail by ~19 MeV/event (~1.2 %) and
    G18_10d by ~5 MeV/event (~0.3 %); NuWro and G18_10a are unaffected
    since neither produces these fragments in the samples used here.
    Lepton energy ELep is added separately (so the histogrammed quantity is
    full E_ν^reco = ELep + Σ_hadronic, not just the recoil).

    ``lep_pdg`` selects the expected primary lepton: 13 (default) for
    νμ/ν̄μ samples; 11 for νe/ν̄e appearance samples. The cut removes the
    ~0.004% misID-tail (NUISFLAT's ELep filled with the wrong lepton's
    energy from hard radiation, producing an unphysical +50..+3000 MeV bias
    tail)."""
    n, pdg, E, px, py, pz = particles_arr(arr, vertex=vertex)
    apdg = abs(pdg)
    p2 = px*px + py*py + pz*pz
    mass2 = E*E - p2
    mass = ak.where(mass2 > 0, np.sqrt(ak.where(mass2 > 0, mass2, 0.0)), 0.0)

    is_p     = apdg == 2212
    is_chpi  = apdg == 211
    is_pi0   = apdg == 111
    # is_e catches SECONDARY electrons (from γ → e+e- conversion / hard
    # radiation) which legitimately belong in the EM-shower part of the
    # hadronic recoil.  The PRIMARY lepton's energy is added separately via
    # arr['ELep'] below — excluding position 0 here avoids double-counting
    # on νe/ν̄e samples (post-FSI primary electron lives at pdg[0]). No
    # effect on νμ/ν̄μ samples since their primary is pdg=±13, never in
    # is_e. For vertex=True position 0 is the incoming neutrino (pdg=±12/14),
    # never in is_e either, so the exclusion is a no-op there — accepted as
    # a small mis-handling of the hypothetical vertex=True × νe combination,
    # which isn't used by any current figure script.
    is_primary = ak.local_index(apdg) == 0
    is_e     = (apdg == 11) & ~is_primary
    is_gamma = apdg == 22
    # Light nuclear fragments (deuteron, triton, alpha): treat their KE
    # like the proton's in both definitions — rest mass is invisible to
    # the detector, only dE/dx is recorded.
    is_lightnuc = (apdg == 1000010020) | (apdg == 1000010030) | (apdg == 1000020040)
    full_E_set = is_pi0 | is_e | is_gamma                # always +E
    kin_set_wo   = is_p | is_chpi | is_lightnuc          # kinetic in def 1
    kin_set_with = is_p | is_lightnuc                    # kinetic in def 2

    # Definition 1 (no π mass): p, π± and light nuclei contribute kinetic energy.
    add_wo   = kin_set_wo   * (E - mass) + full_E_set * E
    # Definition 2 (with π mass): p and light nuclei kinetic, π± full E.
    add_with = kin_set_with * (E - mass) + (is_chpi | full_E_set) * E

    enuhad_wo   = ak.to_numpy(arr['ELep']) + ak.to_numpy(ak.sum(add_wo,   axis=1))
    enuhad_with = ak.to_numpy(arr['ELep']) + ak.to_numpy(ak.sum(add_with, axis=1))
    Enu_true    = ak.to_numpy(arr['Enu_true'])
    cc_mask     = np.asarray(arr['cc'], dtype=bool) & (np.abs(np.asarray(arr['PDGLep'])) == lep_pdg)
    return (enuhad_wo - Enu_true)[cc_mask], (enuhad_with - Enu_true)[cc_mask], cc_mask


# ---------------------------------------------------------------------------
# Output directory placeholder. Set the $OUTPUT_PLOTS_DIR environment variable
# to redirect every figure-script savefig into a single staging tree
# (e.g. /eos/home-l/lamuntea/FSI_IOP_paper/iop_plots). Per-figure subdirs
# (Fig1_plots/, Fig2_plots/, ...) are preserved under that root. If unset,
# scripts fall back to the existing relative dirs under scripts/.
# ---------------------------------------------------------------------------
OUTPUT_PLOTS_DIR = os.environ.get("OUTPUT_PLOTS_DIR", "")


def outpath(subdir, filename):
    """Resolve a figure-output path.

    Returns ``$OUTPUT_PLOTS_DIR/<subdir>/<filename>`` if OUTPUT_PLOTS_DIR is set,
    else ``<subdir>/<filename>`` (relative to the script's cwd). Creates the
    parent directory if needed so callers can plt.savefig(...) without
    pre-mkdir-ing.
    """
    full_dir = os.path.join(OUTPUT_PLOTS_DIR, subdir) if OUTPUT_PLOTS_DIR else subdir
    os.makedirs(full_dir, exist_ok=True)
    return os.path.join(full_dir, filename)


# ---------------------------------------------------------------------------
# Relative-bias support — paper-wide constants and unified bias accessor.
# Relative bias delta = (E_reco - E_true) / E_true (dimensionless). Used
# alongside the existing absolute MeV bias by parameterising every bias-plot
# script on kind in {"abs", "rel"}.
# ---------------------------------------------------------------------------
REL_BIAS_XLIM = (-0.9, 0.3)
DSIGMA_DREL_LABEL = r"$\mathrm{d}\sigma/\mathrm{d}\delta$ [10$^{-42}$ cm$^{2}$/nucleon]"


def bias_arr(arr, observable, *, kind="abs", vertex=False, lep_pdg=13):
    """Unified neutrino-energy bias accessor used by every bias-plot script.

    observable : {'qe', 'had', 'avail'}
        - 'qe'    -- HK CC0pi, Enu_QE per IOP paper Eq. 9
        - 'had'   -- DUNE Enu^had with full charged-pion energy
        - 'avail' -- DUNE Enu^avail with charged-pion kinetic energy only
    kind : {'abs', 'rel'}
        - 'abs' -- (E_reco - E_true) in MeV (matches diff_enu_qe_arr semantics)
        - 'rel' -- (E_reco - E_true) / E_true, dimensionless
    lep_pdg : int (only used for observable='qe')
        13 (default) for νμ/ν̄μ samples, 11 for νe/ν̄e samples — forwarded to
        the underlying ``is_cc0pi_arr`` selection.

    Returns a 1-D numpy array of length = events passing the observable's
    intrinsic selection (CC0pi for 'qe'; CC for 'had'/'avail').
    """
    if observable == "qe":
        sel = is_cc0pi_arr(arr, vertex=vertex, lep_pdg=lep_pdg)
        enu_qe_GeV   = ak.to_numpy(arr['Enu_QE'])[sel]
        enu_true_GeV = ak.to_numpy(arr['Enu_true'])[sel]
        diff_GeV = enu_qe_GeV - enu_true_GeV
        if kind == "abs":
            return diff_GeV * 1000.0
        safe = np.where(enu_true_GeV > 0, enu_true_GeV, 1.0)
        return np.where(enu_true_GeV > 0, diff_GeV / safe, 0.0)

    if observable in ("had", "avail"):
        bias_wo_GeV, bias_with_GeV, cc_mask = enu_had_arr(arr, vertex=vertex, lep_pdg=lep_pdg)
        diff_GeV = bias_with_GeV if observable == "had" else bias_wo_GeV
        if kind == "abs":
            return diff_GeV * 1000.0
        enu_true_GeV = ak.to_numpy(arr['Enu_true'])[np.asarray(cc_mask, dtype=bool)]
        safe = np.where(enu_true_GeV > 0, enu_true_GeV, 1.0)
        return np.where(enu_true_GeV > 0, diff_GeV / safe, 0.0)

    raise ValueError(f"unknown observable {observable!r}; use 'qe', 'had' or 'avail'")


_BIAS_XLABEL = {
    ("qe",    "abs"): r"$E_{\nu}^{\rm QE} - E_{\nu}^{\rm true}$ [MeV]",
    ("qe",    "rel"): r"$(E_{\nu}^{\rm QE} - E_{\nu}^{\rm true}) / E_{\nu}^{\rm true}$",
    ("had",   "abs"): r"$E_{\nu}^{\rm had} - E_{\nu}^{\rm true}$ [MeV]",
    ("had",   "rel"): r"$(E_{\nu}^{\rm had} - E_{\nu}^{\rm true}) / E_{\nu}^{\rm true}$",
    ("avail", "abs"): r"$E_{\nu}^{\rm avail} - E_{\nu}^{\rm true}$ [MeV]",
    ("avail", "rel"): r"$(E_{\nu}^{\rm avail} - E_{\nu}^{\rm true}) / E_{\nu}^{\rm true}$",
    ("reco",  "abs"): r"$E_{\nu}^{\rm reco} - E_{\nu}^{\rm true}$ [MeV]",
    ("reco",  "rel"): r"$(E_{\nu}^{\rm reco} - E_{\nu}^{\rm true}) / E_{\nu}^{\rm true}$",
}


def bias_xlabel(observable, kind):
    """Return the LaTeX x-axis label for a bias plot.
    observable in {'qe', 'had', 'avail', 'reco'}; kind in {'abs', 'rel'}."""
    try:
        return _BIAS_XLABEL[(observable, kind)]
    except KeyError as e:
        raise ValueError(f"unknown (observable, kind) = {e.args[0]!r}")


def bias_ylabel(kind):
    """Return the dsigma/dE y-axis label for a 1-D bias plot.
    Matches DSIGMA_DE_LABEL (MeV-scaled) for kind='abs' and a dimensionless
    analogue for kind='rel'."""
    return DSIGMA_DE_LABEL if kind == "abs" else DSIGMA_DREL_LABEL


def enu_true_arr(arr, observable, *, vertex=False, lep_pdg=13):
    """Return Enu_true in GeV, filtered to the same selection bias_arr uses
    for `observable` in {'qe', 'had', 'avail'}.

    Used by validation_distributions.py to render the Enu_true input
    distribution alongside the bias histogram so the inputs to the bias
    calculation can be eyeballed per-variant."""
    if observable == "qe":
        sel = is_cc0pi_arr(arr, vertex=vertex, lep_pdg=lep_pdg)
        return ak.to_numpy(arr['Enu_true'])[sel]
    if observable in ("had", "avail"):
        _, _, cc_mask = enu_had_arr(arr, vertex=vertex, lep_pdg=lep_pdg)
        return ak.to_numpy(arr['Enu_true'])[np.asarray(cc_mask, dtype=bool)]
    raise ValueError(f"unknown observable {observable!r}; use 'qe', 'had' or 'avail'")


def enu_reco_arr(arr, observable, *, vertex=False, lep_pdg=13):
    """Return Enu_reco in GeV (the appropriate reconstruction for the given
    observable), filtered to the same selection as bias_arr.

      'qe'    -> Enu_QE (IOP paper Eq. 9, written into arr['Enu_QE'] by load_arrays)
      'had'   -> ELep + ehad_with (charged-pion full E)
      'avail' -> ELep + ehad_wo   (charged-pion KE only)
    """
    if observable == "qe":
        sel = is_cc0pi_arr(arr, vertex=vertex, lep_pdg=lep_pdg)
        return ak.to_numpy(arr['Enu_QE'])[sel]
    if observable in ("had", "avail"):
        bias_wo_GeV, bias_with_GeV, cc_mask = enu_had_arr(arr, vertex=vertex, lep_pdg=lep_pdg)
        diff_GeV = bias_with_GeV if observable == "had" else bias_wo_GeV
        enu_true_GeV = ak.to_numpy(arr['Enu_true'])[np.asarray(cc_mask, dtype=bool)]
        return diff_GeV + enu_true_GeV
    raise ValueError(f"unknown observable {observable!r}; use 'qe', 'had' or 'avail'")


def auto_ratio_ylim(ax_ratio, counts_nom, *,
                    peak_frac=0.01, padding=1.1, floor=0.005):
    """Set ax_ratio's y-limits to ± padding × max(|ratio − 1|), restricted to
    bins where ``counts_nom`` exceeds ``peak_frac × peak(counts_nom)``.

    Use this on the bottom panel of figures that overlay variant/nominal
    ratios so the y-range adapts to the actual variation while ignoring the
    wild bin-to-bin fluctuations in the low-stat tails. ``floor`` is a
    minimum half-range so the y-axis doesn't collapse to nothing when
    variations are tiny.

    Operates on the Line2D objects already drawn into ``ax_ratio`` via
    ``ax_ratio.step(...)`` -- ``ax_ratio.hlines(1, ...)`` adds a
    LineCollection so it doesn't appear in ``get_lines()`` and the
    reference at 1 is correctly ignored.
    """
    counts_nom = np.asarray(counts_nom)
    if counts_nom.size == 0:
        return
    peak = float(np.nanmax(counts_nom))
    if peak <= 0:
        return
    mask = counts_nom > peak_frac * peak
    max_dev = float(floor)
    for line in ax_ratio.get_lines():
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if ydata.size == 0:
            continue
        # Line ydata may be a hair shorter than mask (some scripts drop
        # the underflow bin); align to whichever is shorter.
        n = min(len(ydata), len(mask))
        m = mask[:n]
        valid = ydata[:n][m]
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            continue
        dev = float(np.max(np.abs(valid - 1.0)))
        if np.isfinite(dev):
            max_dev = max(max_dev, dev)
    half = padding * max_dev
    ax_ratio.set_ylim(1.0 - half, 1.0 + half)


def smooth_shift_ratio(counts, edges, shift,
                       count_floor_frac=1e-3, ratio_clip=10.0,
                       order=4):
    """Analytical H(E - shift) / H(E) via Taylor expansion of log-counts.

    Uses the order-N Taylor expansion of ln H around E:
        ln H(E - Δ) ≈ ln H(E) + Σ_{k=1..N}  (-Δ)^k / k!  ·  d^k ln H / dE^k
    then exponentiates to get the ratio.  N is the ``order`` arg (1..4).

    For an exponentially-falling spectrum H(E) ~ exp(-aE) the order-1
    answer is already exact; higher orders matter at the spectrum
    peak / shoulder where d²(ln H)/dE², etc., are non-negligible. Order
    4 is the practical limit — numerical finite-difference noise grows
    rapidly with each extra derivative.

    `edges` and `shift` must be in the same units; `counts` is the
    unshifted weighted bin contents. `shift` may be a scalar or a per-bin
    array (numpy broadcasts) — the latter handles per-event reconstructions
    where the shift scales with energy (e.g. Δ = frac × E).

    Numerical guard: in bins where `counts` < `count_floor_frac` × peak,
    the `np.maximum(counts, 1e-30)` floor makes log H drop to ~-69 and the
    centered-difference gradient at the live/dead boundary becomes huge,
    blowing the exp() up by many orders of magnitude. There's no real
    information in those near-empty bins, so we pin the ratio to 1.0
    (no shift correction) and additionally clip the final ratio to
    [1/ratio_clip, ratio_clip] as a safety net inside the live region.
    """
    if not (1 <= int(order) <= 4):
        raise ValueError(f"order must be in [1, 4], got {order}")
    centers = 0.5 * (edges[:-1] + edges[1:])
    safe = np.maximum(counts, 1e-30)
    log_H = np.log(safe)

    # Build successive derivatives d^k ln H / dE^k via repeated centered
    # finite differences. Each derivative amplifies numerical noise; if
    # order > 1 the user accepts that trade-off.
    deriv = [log_H]
    for _ in range(int(order)):
        deriv.append(np.gradient(deriv[-1], centers))
    # deriv[k] for k >= 1 is d^k ln H / dE^k.

    # log-ratio = Σ_{k=1..N} (-Δ)^k / k! · d^k ln H / dE^k
    from math import factorial
    log_ratio = np.zeros_like(centers, dtype=float)
    for k in range(1, int(order) + 1):
        log_ratio = log_ratio + ((-shift) ** k / factorial(k)) * deriv[k]
    ratio = np.exp(log_ratio)

    counts_arr = np.asarray(counts)
    peak = float(counts_arr.max()) if counts_arr.size else 0.0
    if peak > 0:
        alive = counts_arr > count_floor_frac * peak
        ratio = np.where(alive, ratio, 1.0)
    return np.clip(ratio, 1.0 / ratio_clip, ratio_clip)


# Default method for energy-shift systematics in the Fig1 ratio panels.
#   'taylor'   - analytical exp(-Δ d ln H / dE) — smooth (default)
#   'explicit' - per-event histogram of x + shift — bin-to-bin Poisson noise
# Override at runtime via:  FlatTreeMod.ENERGY_SHIFT_METHOD = 'explicit'
# or per-call via the `method` arg of `plot_osc_shift_e`.
ENERGY_SHIFT_METHOD = 'taylor'


def plot_osc_shift_e(ax, ax_ratio, *, shift, label, color, counts_nom, bins,
                     x_unshifted, weights, method=None, lw=1.5, ls='--'):
    """Plot an energy-shifted variant on (ax, ax_ratio).

    Top panel: a curve at the shifted energy values.
    Ratio panel: H(E - shift) / H(E).

    method is one of 'taylor' / 'explicit'; default = ENERGY_SHIFT_METHOD.
    The default linestyle is dashed so the energy-shift variants are
    visually distinct from the (solid) oscillation-parameter variants.
    """
    if method is None:
        method = ENERGY_SHIFT_METHOD
    centers = 0.5 * (bins[:-1] + bins[1:])
    if method == 'taylor':
        # Visual-shift curve: same counts, x translated by +shift.
        ax.step(centers + shift, counts_nom, where='mid',
                color=color, lw=lw, ls=ls, label=label)
        ratio = smooth_shift_ratio(counts_nom, bins, shift)
    elif method == 'explicit':
        counts, _ = np.histogram(x_unshifted + shift, bins=bins, weights=weights)
        ax.step(centers, counts, where='mid',
                color=color, lw=lw, ls=ls, label=label)
        ratio = counts / np.maximum(counts_nom, 1e-30)
    else:
        raise ValueError(f"unknown method {method!r}; use 'taylor' or 'explicit'")
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ax_ratio.step(centers, ratio, where='mid', color=color, lw=lw, ls=ls)
    return ratio


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
