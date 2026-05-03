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
    "savefig.bbox": "tight",
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


# =============================================================================
# IOP figure-size standards. Single column = 86 mm = 3.39"; double = 178 mm = 7.01".
# Sizing the figure at creation time (instead of letting LaTeX shrink an
# oversized canvas) keeps label pt size honest on the printed page.
# =============================================================================
FIG_SIZES = {
    'single':         (3.5, 2.6),    # one panel, single column
    'single_ratio':   (3.5, 4.0),    # main + ratio strip, single column (Fig3/4/7)
    'double':         (7.0, 3.0),    # one wide panel, double column
    'double_ratio':   (7.0, 4.5),    # main + ratio, double column
    'double_stacked': (7.0, 5.0),    # two stacked panels (Fig5 noFSI/FSI)
    'box':            (5.0, 7.0),    # tall, narrow box-and-whisker comparison
}


def make_fig(kind='single', **subplots_kw):
    """Single-panel figure sized for IOP publication. Returns (fig, ax)."""
    return plt.subplots(figsize=FIG_SIZES[kind], **subplots_kw)


def make_fig_ratio(kind='single_ratio', height_ratios=(3, 1), hspace=0.05):
    """Two-row main + ratio figure with shared x. Returns (fig, (ax_main, ax_ratio))."""
    return plt.subplots(
        2, 1, sharex=True, figsize=FIG_SIZES[kind],
        gridspec_kw={'height_ratios': list(height_ratios), 'hspace': hspace},
    )


def make_fig_stacked(kind='double_stacked', sharex=True, sharey=False, hspace=0.08):
    """Two stacked panels (rows) figure. Returns (fig, (ax_top, ax_bot))."""
    return plt.subplots(
        2, 1, sharex=sharex, sharey=sharey, figsize=FIG_SIZES[kind],
        gridspec_kw={'hspace': hspace},
    )


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
    ('DUNE', 'numu'):    15000,
    ('DUNE', 'numubar'):  9000,
    ('DUNE', 'nue'):      1500,
    ('DUNE', 'nuebar'):   1000,
    ('HK',   'numu'):     9000,
    ('HK',   'numubar'):  7000,
    ('HK',   'nue'):      2000,
    ('HK',   'nuebar'):   1000,
}

EVENT_RATE_LABEL = r"Events / MeV"


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


def make_weights_event_rate(arr, filename, bin_width):
    """Per-event constant weight for Enu-spectrum plots scaled to expected
    event yield. Returns target / N_gen / bin_width — caller multiplies by
    np.ones_like(observable). Y-axis is 'Events / MeV' (EVENT_RATE_LABEL).
    For oscillated spectra, multiply this scalar by the per-event probability
    array before passing to ax.hist as weights.
    """
    exp, flav = detect_exp_flav(filename)
    target = EXPECTED_EVENTS[(exp, flav)]
    n_gen  = len(arr['Enu_true'])
    return target / n_gen / bin_width


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

# Cache directory — defaults to /eos (huge quota) since AFS home is small.
# Override with $IOP_PAPER_CACHE if needed.
_CACHE_DIR = _os.environ.get(
    'IOP_PAPER_CACHE',
    '/eos/home-l/lamuntea/.cache/iop_paper'
)
_os.makedirs(_CACHE_DIR, exist_ok=True)

# Branches commonly used by every Fig*.py. Pulling them all in one pass means
# the cache is reusable across figs that read the same file.
DEFAULT_BRANCHES = (
    'Enu_true', 'Enu_QE', 'ELep', 'Mode', 'cc', 'fScaleFactor',
    # flagCC0pi removed -- we derive CC0π from the pdg stack inside
    # is_cc0pi / is_cc0pi_arr so the same code path works on GENIE NUISFLAT
    # files (which don't write that branch).
    'nfsp',   'pdg',      'E',      'px',      'py',      'pz',
    'nvertp', 'pdg_vert', 'E_vert', 'px_vert', 'py_vert', 'pz_vert',
    'ninitp',
)


def _cache_key(filename, branches, max_events):
    h = _hashlib.sha1()
    try:
        st = _os.stat(filename)
        h.update(f"{filename}|{st.st_size}|{int(st.st_mtime)}".encode())
    except OSError:
        h.update(filename.encode())
    h.update(",".join(sorted(branches)).encode())
    h.update(str(max_events).encode())
    return h.hexdigest()


def load_arrays(filename, branches=DEFAULT_BRANCHES, max_events=None,
                tree_name='FlatTree_VARS'):
    """Load (cached) NUISFLAT branches from `filename` as awkward arrays.

    Returns a dict ``{branch_name: awkward_array}``. Cached on disk under
    ~/.cache/iop_paper/ keyed on (file mtime, size, branch list, max_events).
    Subsequent calls return the cache instead of reading the ROOT file.
    """
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
    try:
        with open(cache_path, 'wb') as f:
            _pickle.dump(out, f, protocol=_pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        Warn(f"could not write cache {cache_path}: {e}")
    return out


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


def is_cc0pi_arr(arr, *, vertex=False):
    """Vectorised CC0pi mask. Returns a 1D numpy bool array of length nevents.

    Both branches now compute the selection directly from the particle stack
    (`pdg` for vertex=False / post-FSI; `pdg_vert` for vertex=True / pre-FSI).
    The post-FSI branch used to read the pre-computed `flagCC0pi` field but
    GENIE NUISFLAT files don't carry it -- computing from `pdg` works for any
    NUISFLAT-format file regardless of generator.
    """
    cc = np.asarray(arr['cc'], dtype=bool)
    pdg_field = 'pdg_vert' if vertex else 'pdg'
    apdg = abs(arr[pdg_field])
    has_chpi = ak.any(apdg == 211, axis=1)
    has_pi0  = ak.any(apdg == 111, axis=1)
    no_pi = ~ak.to_numpy(has_chpi | has_pi0)
    return cc & no_pi


def diff_enu_qe_arr(arr, *, vertex=False, scale_to_MeV=True):
    """Vectorised (Enu_QE - Enu_true) for events passing CC0pi selection.
    Returns a numpy array (in MeV by default)."""
    sel = is_cc0pi_arr(arr, vertex=vertex)
    diff = ak.to_numpy(arr['Enu_QE'] - arr['Enu_true'])[sel]
    return diff * 1000.0 if scale_to_MeV else diff


def enu_had_arr(arr, *, vertex=False):
    """Vectorised hadronic-energy reconstruction matching NUISANCE
    ``GetErecoil_MINERvA_LowRecoil`` for the hadronic part.

    Returns ``(bias_wo, bias_with, valid_mask)`` *filtered to CC events*:
      - bias_wo   = E_ν^reco(no pion-mass subtraction)           − Enu_true
      - bias_with = E_ν^reco(π± with full E, p still kinetic)    − Enu_true
      - valid_mask is the CC selection (arr['cc']).

    Per-particle contribution (everything else contributes 0):
      proton  (2212):       T = E − m
      π±      (211):        T = E − m   (def 1, "no π mass")
                            E           (def 2, "with π mass")
      π0      (111):        E
      e±      (11):         E
      γ       (22):         E
      neutrons / heavy / |pdg|>3000 / strange / etc.: skipped
    Lepton energy ELep is added separately (so the histogrammed quantity is
    full E_ν^reco = ELep + Σ_hadronic, not just the recoil)."""
    n, pdg, E, px, py, pz = particles_arr(arr, vertex=vertex)
    apdg = abs(pdg)
    p2 = px*px + py*py + pz*pz
    mass2 = E*E - p2
    mass = ak.where(mass2 > 0, np.sqrt(ak.where(mass2 > 0, mass2, 0.0)), 0.0)

    is_p     = apdg == 2212
    is_chpi  = apdg == 211
    is_pi0   = apdg == 111
    is_e     = apdg == 11
    is_gamma = apdg == 22
    full_E_set = is_pi0 | is_e | is_gamma                # always +E

    # Definition 1 (no π mass): p and π± both contribute kinetic energy.
    add_wo   = (is_p | is_chpi) * (E - mass) + full_E_set * E
    # Definition 2 (with π mass): p kinetic, π± full E.
    add_with = is_p * (E - mass) + (is_chpi | full_E_set) * E

    enuhad_wo   = ak.to_numpy(arr['ELep']) + ak.to_numpy(ak.sum(add_wo,   axis=1))
    enuhad_with = ak.to_numpy(arr['ELep']) + ak.to_numpy(ak.sum(add_with, axis=1))
    Enu_true    = ak.to_numpy(arr['Enu_true'])
    cc_mask     = np.asarray(arr['cc'], dtype=bool)
    return (enuhad_wo - Enu_true)[cc_mask], (enuhad_with - Enu_true)[cc_mask], cc_mask


def smooth_shift_ratio(counts, edges, shift):
    """Analytical H(E - shift) / H(E) via centered finite-difference of log-counts.

    Uses the first-order Taylor expansion
        H(E - Δ) / H(E) ≈ exp(-Δ · d ln H / dE)
    which for small Δ vs the spectrum width gives a smooth ratio with no
    per-bin Poisson scatter from the explicit-shift implementation.

    `edges` and `shift` must be in the same units; `counts` is the
    unshifted weighted bin contents.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    safe = np.maximum(counts, 1e-30)
    dlogH_dE = np.gradient(np.log(safe), centers)
    return np.exp(-shift * dlogH_dE)


# Default method for energy-shift systematics in the Fig1 ratio panels.
#   'taylor'   - analytical exp(-Δ d ln H / dE) — smooth (default)
#   'explicit' - per-event histogram of x + shift — bin-to-bin Poisson noise
# Override at runtime via:  FlatTreeMod.ENERGY_SHIFT_METHOD = 'explicit'
# or per-call via the `method` arg of `plot_osc_shift_e`.
ENERGY_SHIFT_METHOD = 'taylor'


def plot_osc_shift_e(ax, ax_ratio, *, shift, label, color, counts_nom, bins,
                     x_unshifted, weights, method=None, lw=1.5, ls='-'):
    """Plot an energy-shifted variant on (ax, ax_ratio).

    Top panel: a curve at the shifted energy values.
    Ratio panel: H(E - shift) / H(E).

    method is one of 'taylor' / 'explicit'; default = ENERGY_SHIFT_METHOD.
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
