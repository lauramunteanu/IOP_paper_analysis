"""LaTeX variation table across the 5 plot categories from fsi_iop_plots.py.

For each (category, flavour, observable), prints the range of weighted medians
and weighted means across the variants in that category (max − min). Cells are
shaded pale red where the range exceeds:
   5 MeV  for HK observables
  15 MeV  for DUNE observables

Categories (mirroring fsi_iop_plots.py, all corresponding to the 5 plots):
  - FSI vs noFSI         (2 variants)
  - π_abs ±31%           (3 variants: 069, nominal, 131)
  - NN MFP ±30%          (3 variants: 0.7, nominal, 1.3)
  - GENIE cascade        (4 variants: G18_10a/b/c/d)
  - EDRMF vs RPWIA       (2 variants -- RPWIA placeholders; row prints '--' if
                          its files don't resolve)

Shares the on-disk pickle cache with fsi_iop_plots.py:
   /eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw/_cache/

Path-fallback to *.bak.preOPpatch / *.bak.preNNmfp keeps this runnable while
the live regen is in flight.
"""
import os
import sys
import numpy as np
import awkward as ak

# Share the FlatTreeMod on-disk cache (~/.cache/iop_paper/) instead of using
# a separate one — keeps this script in lock-step with fsi_iop_plots.py and
# all the canonical figures. load_arrays auto-busts on file mtime/size change.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from FlatTreeMod import load_arrays  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (kept in sync with fsi_iop_plots.py)
# ---------------------------------------------------------------------------
BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/nuwro_25031"
NEUT_BASE = "/eos/home-l/lamuntea/FSI_IOP_paper/neut_runs"
GENIE_FILES_BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs"  # existing GENIE NUISFLAT samples — separate from NuWro BASE
MAX_EVENTS = None

THRESH_HK   = 5.0   # MeV
THRESH_DUNE = 15.0  # MeV


# ---------------------------------------------------------------------------
# File catalogues (must match fsi_iop_plots.py)
# ---------------------------------------------------------------------------
NUWRO_HK = {
    "numu":    {"FSI":   f"{BASE}/HK/HK_numu_FSI.flat.root",
                "noFSI": f"{BASE}/HK/HK_numu_noFSI.flat.root"},
    "numubar": {"FSI":   f"{BASE}/HK/HK_numubar_FSI.flat.root",
                "noFSI": f"{BASE}/HK/HK_numubar_noFSI.flat.root"},
}
NUWRO_DUNE = {
    "numu":    {"FSI":   f"{BASE}/DUNE/DUNE_numu_FSI.flat.root",
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
    ("numu",    t): f"{GENIE_FILES_BASE}/HK/GENIE/T2KSK_unosc_FHC_numu_H2O_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
}
GENIE_HK.update({
    ("numubar", t): f"{GENIE_FILES_BASE}/HK/GENIE/T2KSK_unosc_RHC_numubar_H2O_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
})
GENIE_DUNE = {
    ("numu",    t): f"{GENIE_FILES_BASE}/DUNE/GENIE/DUNEFD_unosc_FHC_numu_Ar40_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
}
GENIE_DUNE.update({
    ("numubar", t): f"{GENIE_FILES_BASE}/DUNE/GENIE/DUNEFD_unosc_RHC_numubar_Ar40_GENIEv3_G18_{t}_00_000_1M_0000_NUISFLAT.root"
    for t in ("10a", "10b", "10c", "10d")
})
NEUT_EDRMF = {
    ("numu",    "hk"):   f"{NEUT_BASE}/HK_numu_H2O_EDRMF/EDRMF.flat.root",
    ("numubar", "hk"):   f"{NEUT_BASE}/HK_numubar_H2O_EDRMF/EDRMF.flat.root",
    ("numu",    "dune"): f"{NEUT_BASE}/DUNE_numu_Ar40_EDRMF/EDRMF.flat.root",
    ("numubar", "dune"): f"{NEUT_BASE}/DUNE_numubar_Ar40_EDRMF/EDRMF.flat.root",
}
NEUT_RPWIA = {
    ("numu",    "hk"):   f"{NEUT_BASE}/HK_numu_H2O_RPWIA/RPWIA.flat.root",
    ("numubar", "hk"):   f"{NEUT_BASE}/HK_numubar_H2O_RPWIA/RPWIA.flat.root",
    ("numu",    "dune"): f"{NEUT_BASE}/DUNE_numu_Ar40_RPWIA/RPWIA.flat.root",
    ("numubar", "dune"): f"{NEUT_BASE}/DUNE_numubar_Ar40_RPWIA/RPWIA.flat.root",
}

OBS_LABELS_LATEX = {
    "hk_qe":    r"HK $E_{\nu}^{\mathrm{QE}}$",
    "dune_epi": r"DUNE $E_{\nu}^{\mathrm{avail}}$",
    "dune_tpi": r"DUNE $E_{\nu}^{\mathrm{had}}$",
}
FLAV_LATEX = {"numu": r"$\nu_\mu$", "numubar": r"$\bar\nu_\mu$"}
THRESH = {"hk_qe": THRESH_HK, "dune_epi": THRESH_DUNE, "dune_tpi": THRESH_DUNE}


# ---------------------------------------------------------------------------
# Path resolver + cache + readers (lifted from fsi_iop_plots.py)
# ---------------------------------------------------------------------------
def resolve_path(path):
    """Live files only. Returns None if missing (so EDRMF/RPWIA placeholder
    rows skip cleanly). The .bak.preOPpatch fallback was removed once the
    OP-patched regen landed on 2026-05-03."""
    if path and os.path.exists(path):
        return path
    return None


def weighted_percentile(x, w, q):
    keep = (w > 0) & np.isfinite(x)
    x, w = x[keep], w[keep]
    if len(x) == 0:
        return np.nan
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    p = (cw - 0.5 * ws) / cw[-1]
    return float(np.interp(q / 100.0, p, xs))


def weighted_mean(x, w):
    keep = (w > 0) & np.isfinite(x)
    if keep.sum() == 0:
        return np.nan
    return float(np.average(x[keep], weights=w[keep]))


def _read_hk(path):
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


def _read_dune(path):
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


# In-process memoisation for repeated calls in one run; disk cache lives in
# load_arrays (shared with fsi_iop_plots.py and all canonical scripts).
_CACHE = {}


def get_hk(path):
    p = resolve_path(path)
    if p is None: return None
    if ("hk", p) not in _CACHE:
        _CACHE[("hk", p)] = _read_hk(p)
    return _CACHE[("hk", p)]


def get_dune(path):
    p = resolve_path(path)
    if p is None: return None
    if ("dune", p) not in _CACHE:
        _CACHE[("dune", p)] = _read_dune(p)
    return _CACHE[("dune", p)]


def get_xw(path, obs_key):
    if obs_key == "hk_qe":
        out = get_hk(path)
        if out is None: return None
        x, w, _ = out
        return x, w
    out = get_dune(path)
    if out is None: return None
    x_tpi, x_epi, w, _ = out
    return (x_tpi, w) if obs_key == "dune_tpi" else (x_epi, w)


# ---------------------------------------------------------------------------
# Variant catalogue per category
# Each category yields a list of (variant_label, path_for_obs_key_callable).
# path_for_obs_key_callable(flavour, obs_key) -> path or None
# ---------------------------------------------------------------------------
def _hk_or_dune(obs_key, hk_path, dune_path):
    return hk_path if obs_key == "hk_qe" else dune_path


def fsi_paths(flav, obs_key, which):
    return _hk_or_dune(obs_key,
                       NUWRO_HK[flav][which],
                       NUWRO_DUNE[flav][which])


def piabs_paths(flav, obs_key, scale):
    if scale == "100":
        return _hk_or_dune(obs_key,
                           NUWRO_HK[flav]["FSI"],
                           NUWRO_DUNE[flav]["FSI"])
    return _hk_or_dune(obs_key,
                       NUWRO_HK_PIABS[(flav, scale)],
                       NUWRO_DUNE_PIABS[(flav, scale)])


def mfp_paths(flav, obs_key, scale):
    if scale == "1p0":
        return _hk_or_dune(obs_key,
                           NUWRO_HK[flav]["FSI"],
                           NUWRO_DUNE[flav]["FSI"])
    return _hk_or_dune(obs_key,
                       NUWRO_HK_MFP[(flav, scale)],
                       NUWRO_DUNE_MFP[(flav, scale)])


def genie_paths(flav, obs_key, tune):
    return _hk_or_dune(obs_key,
                       GENIE_HK[(flav, tune)],
                       GENIE_DUNE[(flav, tune)])


def edrmf_paths(flav, obs_key, kind):
    catalogue = NEUT_EDRMF if kind == "EDRMF" else NEUT_RPWIA
    return catalogue[(flav, "hk")] if obs_key == "hk_qe" else catalogue[(flav, "dune")]


CATEGORIES = [
    (r"FSI vs no FSI",
     ["FSI", "noFSI"],
     lambda flav, obs, v: fsi_paths(flav, obs, v)),
    (r"$\pi_{\mathrm{abs}}$ $\pm 31\%$",
     ["069", "100", "131"],
     lambda flav, obs, v: piabs_paths(flav, obs, v)),
    (r"NN MFP $\pm 30\%$",
     ["0p7", "1p0", "1p3"],
     lambda flav, obs, v: mfp_paths(flav, obs, v)),
    (r"GENIE cascade (10a-10d)",
     ["10a", "10b", "10c", "10d"],
     lambda flav, obs, v: genie_paths(flav, obs, v)),
    (r"EDRMF vs RPWIA",
     ["EDRMF", "RPWIA"],
     lambda flav, obs, v: edrmf_paths(flav, obs, v)),
]


# ---------------------------------------------------------------------------
# Main: build (category, flavour, observable) rows
# ---------------------------------------------------------------------------
def category_row(cat_label, variants, path_fn, flav, obs_key):
    medians, means = [], []
    for v in variants:
        path = path_fn(flav, obs_key, v)
        res = get_xw(path, obs_key)
        if res is None:
            continue
        x, w = res
        medians.append(weighted_percentile(x, w, 50))
        means.append(weighted_mean(x, w))
    if len(medians) < 2:
        return None, None
    med_range = max(medians) - min(medians)
    mean_range = max(means) - min(means)
    return med_range, mean_range


# Compute all rows
print("% (loading data; first run may take several minutes)")
rows = []
for cat_label, variants, path_fn in CATEGORIES:
    for flav in ("numu", "numubar"):
        for obs_key in ("hk_qe", "dune_epi", "dune_tpi"):
            med, mean = category_row(cat_label, variants, path_fn, flav, obs_key)
            rows.append((cat_label, flav, obs_key, med, mean))


# ---------------------------------------------------------------------------
# LaTeX output (printed to stdout AND saved to OUT_DIR/variation_table.tex)
# ---------------------------------------------------------------------------
OUT_DIR = "/eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw"
out_path = os.path.join(OUT_DIR, "variation_table.tex")
os.makedirs(OUT_DIR, exist_ok=True)

# Build all output lines once, write to file, also echo to stdout.
# Uses booktabs (\toprule, \midrule, \bottomrule) for nicer separators,
# rotated multirow group cells, and pale-red shading where a range exceeds
# the per-experiment threshold.
lines = []
lines.append(r"% Variation table -- requires:")
lines.append(r"%   \usepackage{booktabs}")
lines.append(r"%   \usepackage{multirow}")
lines.append(r"%   \usepackage[table]{xcolor}")
lines.append(r"%   \usepackage{graphicx}   % for \rotatebox")
lines.append(r"%   \usepackage{siunitx}    % optional, for unit alignment")
lines.append("")
lines.append(r"\setlength{\tabcolsep}{6pt}")
lines.append(r"\renewcommand{\arraystretch}{1.15}")
lines.append(r"\begin{tabular}{@{}c c l S[table-format=3.1] S[table-format=3.1]@{}}")
lines.append(r"\toprule")
lines.append(r"\textbf{Group} & \textbf{Flavour} & \textbf{Observable} & "
             r"{\textbf{median range}} & {\textbf{mean range}} \\")
lines.append(r"               &                  &                     & "
             r"{[MeV]}              & {[MeV]}             \\")
lines.append(r"\midrule")

cat_counts = {}
for r in rows:
    cat_counts[r[0]] = cat_counts.get(r[0], 0) + 1

last_cat = None
n_so_far = 0
for cat_label, flav, obs_key, med, mean in rows:
    is_last_in_cat = False
    n_so_far += 1
    if cat_label != last_cat:
        if last_cat is not None:
            lines.append(r"\midrule")
        last_cat = cat_label
        cat_cell = (rf"\multirow{{{cat_counts[cat_label]}}}{{*}}"
                    rf"{{\rotatebox[origin=c]{{90}}{{\textbf{{{cat_label}}}}}}}")
    else:
        cat_cell = ""

    flav_cell = FLAV_LATEX[flav]
    obs_cell  = OBS_LABELS_LATEX[obs_key]
    thresh    = THRESH[obs_key]

    def fmt(v):
        # siunitx S column needs raw numbers; cellcolor still works.
        if v is None or not np.isfinite(v):
            return "{--}"
        col = r"\cellcolor{red!12}" if v > thresh else ""
        return f"{col}{v:.1f}"

    lines.append(rf"{cat_cell} & {flav_cell} & {obs_cell} & {fmt(med)} & {fmt(mean)} \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

content = "\n".join(lines) + "\n"
with open(out_path, "w") as f:
    f.write(content)

print()
print(content)
print(f"% wrote {out_path}")
