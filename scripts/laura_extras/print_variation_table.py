"""LaTeX variation table across the 5 plot categories from fsi_iop_plots.py.

Two tables are written, one per bias mode:
  variation_table_abs.tex   --  bias in MeV (shading threshold:  5 MeV HK / 15 MeV DUNE)
  variation_table_rel.tex   --  bias dimensionless (E_reco-E_true)/E_true
                                (shading threshold: 0.01 HK / 0.005 DUNE)

For each (category, flavour, observable), prints the mean-driven (max,min) variant
pair: report mean[v_max] - mean[v_min] AND |median[v_max] - median[v_min]| for
that same pair (the median sub-column is NOT independently re-extremised).

Categories (mirroring fsi_iop_plots.py):
  - FSI vs noFSI         (2 variants)
  - pi_abs +-31%          (3 variants: 069, nominal, 131)
  - NN MFP +-30%          (3 variants: 0.7, nominal, 1.3)
  - GENIE cascade        (4 variants: G18_10a/b/c/d)
  - EDRMF vs RPWIA       (2 variants -- numubar EDRMF cells emit blank)

Shares the on-disk pickle cache with fsi_iop_plots.py via FlatTreeMod.load_arrays
+ FlatTreeMod.bias_arr.
"""
import os
import sys
import numpy as np
import awkward as ak

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from FlatTreeMod import load_arrays, bias_arr, is_cc0pi_arr  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (kept in sync with fsi_iop_plots.py)
# ---------------------------------------------------------------------------
BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/nuwro_25031_morestats"
NEUT_BASE = "/eos/home-l/lamuntea/FSI_IOP_paper/neut_runs"
GENIE_FILES_BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs"
MAX_EVENTS = None

# Per-mode shading thresholds. abs in MeV; rel dimensionless.
THRESH_BY_MODE = {
    "abs": {"hk": 5.0,   "dune": 15.0},
    "rel": {"hk": 0.01,  "dune": 0.005},
}

# Number-formatting per mode (LaTeX cell values).
FMT_BY_MODE = {
    "abs": "{:.1f}",
    "rel": "{:.3f}",
}

UNITS_BY_MODE = {
    "abs": "[MeV]",
    "rel": "[dimensionless]",
}


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
_NEUT_FILES_BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs"
NEUT_EDRMF = {
    ("numu",    "hk"):   f"{_NEUT_FILES_BASE}/HK/NEUT_HK_EDRMF_numu.flat.root",
    ("numubar", "hk"):   None,
    ("numu",    "dune"): f"{_NEUT_FILES_BASE}/DUNE/DUNE_EDRMF_numu.root",
    ("numubar", "dune"): None,
}
NEUT_RPWIA = {
    ("numu",    "hk"):   f"{_NEUT_FILES_BASE}/HK/NEUT_HK_RPWIA_numu.flat.root",
    ("numubar", "hk"):   None,
    ("numu",    "dune"): f"{_NEUT_FILES_BASE}/DUNE/DUNE_RPWIA_numu.root",
    ("numubar", "dune"): None,
}

OBS_LABELS_LATEX = {
    "hk_qe":    r"HK $E_{\nu}^{\mathrm{QE}}$",
    "dune_epi": r"DUNE $E_{\nu}^{\mathrm{avail}}$",
    "dune_tpi": r"DUNE $E_{\nu}^{\mathrm{had}}$",
}
FLAV_LATEX = {"numu": r"$\nu_\mu$", "numubar": r"$\bar\nu_\mu$"}
EXP_BY_OBS = {"hk_qe": "hk", "dune_epi": "dune", "dune_tpi": "dune"}


def resolve_path(path):
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


# ---------------------------------------------------------------------------
# Per-observable readers go through FlatTreeMod.bias_arr so the abs / rel
# distinction is honoured uniformly.
# ---------------------------------------------------------------------------
def _read_hk(path, mode):
    a = load_arrays(path, max_events=MAX_EVENTS)
    x = bias_arr(a, "qe", kind=mode, vertex=False)
    sel = is_cc0pi_arr(a, vertex=False)
    w = np.asarray(a["fScaleFactor"][sel])
    return x, w


def _read_dune(path, mode, observable):
    """observable in {'had', 'avail'}."""
    a = load_arrays(path, max_events=MAX_EVENTS)
    x = bias_arr(a, observable, kind=mode, vertex=False)
    sel = np.asarray(a["cc"], dtype=bool)
    w = np.asarray(a["fScaleFactor"][sel])
    return x, w


_CACHE = {}


def get_xw(path, obs_key, mode):
    """Returns (x, w) or None if file unresolvable."""
    p = resolve_path(path)
    if p is None:
        return None
    key = (p, obs_key, mode)
    if key not in _CACHE:
        if obs_key == "hk_qe":
            _CACHE[key] = _read_hk(p, mode)
        else:
            observable = "had" if obs_key == "dune_tpi" else "avail"
            _CACHE[key] = _read_dune(p, mode, observable)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Path callables per category
# ---------------------------------------------------------------------------
def _hk_or_dune(obs_key, hk_path, dune_path):
    return hk_path if obs_key == "hk_qe" else dune_path


def fsi_paths(flav, obs_key, which):
    return _hk_or_dune(obs_key, NUWRO_HK[flav][which], NUWRO_DUNE[flav][which])


def piabs_paths(flav, obs_key, scale):
    if scale == "100":
        return _hk_or_dune(obs_key, NUWRO_HK[flav]["FSI"], NUWRO_DUNE[flav]["FSI"])
    return _hk_or_dune(obs_key,
                       NUWRO_HK_PIABS[(flav, scale)],
                       NUWRO_DUNE_PIABS[(flav, scale)])


def mfp_paths(flav, obs_key, scale):
    if scale == "1p0":
        return _hk_or_dune(obs_key, NUWRO_HK[flav]["FSI"], NUWRO_DUNE[flav]["FSI"])
    return _hk_or_dune(obs_key,
                       NUWRO_HK_MFP[(flav, scale)],
                       NUWRO_DUNE_MFP[(flav, scale)])


def genie_paths(flav, obs_key, tune):
    return _hk_or_dune(obs_key, GENIE_HK[(flav, tune)], GENIE_DUNE[(flav, tune)])


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
# Cell computation: mean-driven pair selection
# ---------------------------------------------------------------------------
def category_row(variants, path_fn, flav, obs_key, mode):
    """Pick the (max_mean, min_mean) variant pair; report median and mean
    range for THAT pair."""
    medians, means = [], []
    for v in variants:
        path = path_fn(flav, obs_key, v)
        res = get_xw(path, obs_key, mode)
        if res is None:
            continue
        x, w = res
        medians.append(weighted_percentile(x, w, 50))
        means.append(weighted_mean(x, w))
    if len(medians) < 2:
        return None, None
    i_max = int(np.argmax(means))
    i_min = int(np.argmin(means))
    mean_range = means[i_max] - means[i_min]                # >= 0 by construction
    med_range  = abs(medians[i_max] - medians[i_min])       # same pair as mean
    return med_range, mean_range


# ---------------------------------------------------------------------------
# LaTeX table emission
# ---------------------------------------------------------------------------
CAT_HEADERS = [
    (r"FSI vs no FSI",                    r"FSI / no FSI"),
    (r"$\pi_{\mathrm{abs}}$ $\pm 31\%$",   r"$\pi_{\mathrm{abs}}$"),
    (r"NN MFP $\pm 30\%$",                 r"NN MFP"),
    (r"GENIE cascade (10a-10d)",           r"FSI model"),
    (r"EDRMF vs RPWIA",                    r"EDRMF"),
]
OBS_ORDER = ["hk_qe", "dune_epi", "dune_tpi"]


def fmt(v, thresh, num_fmt):
    if v is None or not np.isfinite(v):
        return "--"
    col = r"\cellcolor{red!12}" if v > thresh else ""
    return f"{col}{num_fmt.format(v)}"


def emit_table(mode, rows, out_path):
    """Build the LaTeX table for one bias mode and write it to out_path."""
    num_fmt = FMT_BY_MODE[mode]
    units = UNITS_BY_MODE[mode]
    thresh_by_exp = THRESH_BY_MODE[mode]

    by_key = {(r[0], r[1], r[2]): (r[3], r[4]) for r in rows}

    lines = []
    lines.append(rf"% Variation table ({mode}) -- requires:")
    lines.append(r"%   \usepackage{booktabs}")
    lines.append(r"%   \usepackage{multirow}")
    lines.append(r"%   \usepackage[table]{xcolor}")
    lines.append(r"%   \usepackage{graphicx}   % for \rotatebox")
    lines.append("")
    n_cat = len(CAT_HEADERS)
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    col_spec = "@{}c l " + " ".join("r r" for _ in range(n_cat)) + "@{}"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    hdr_top = [r"\textbf{Flavour}", r"\textbf{Observable}"]
    for _, h in CAT_HEADERS:
        hdr_top.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{h}}}}}")
    lines.append(" & ".join(hdr_top) + r" \\")
    cmid_parts = []
    for i in range(n_cat):
        c1 = 3 + 2 * i
        c2 = c1 + 1
        cmid_parts.append(rf"\cmidrule(lr){{{c1}-{c2}}}")
    lines.append("".join(cmid_parts))
    hdr_sub = ["", ""]
    for _ in CAT_HEADERS:
        hdr_sub.extend(["median", "mean"])
    lines.append(" & ".join(hdr_sub) + r" \\")
    units_row = ["", ""] + [units] * (2 * n_cat)
    lines.append(" & ".join(units_row) + r" \\")
    lines.append(r"\midrule")

    for flav_idx, flav in enumerate(["numu", "numubar"]):
        if flav_idx > 0:
            lines.append(r"\midrule")
        for obs_idx, obs_key in enumerate(OBS_ORDER):
            if obs_idx == 0:
                flav_cell = (rf"\multirow{{{len(OBS_ORDER)}}}{{*}}"
                             rf"{{\rotatebox[origin=c]{{90}}{{\textbf{{{FLAV_LATEX[flav]}}}}}}}")
            else:
                flav_cell = ""
            obs_cell = OBS_LABELS_LATEX[obs_key]
            thresh = thresh_by_exp[EXP_BY_OBS[obs_key]]
            cells = [flav_cell, obs_cell]
            for cat_label, _ in CAT_HEADERS:
                if cat_label.startswith("EDRMF") and flav == "numubar":
                    cells.extend(["", ""])
                    continue
                med, mean = by_key.get((cat_label, flav, obs_key), (None, None))
                cells.append(fmt(med, thresh, num_fmt))
                cells.append(fmt(mean, thresh, num_fmt))
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    content = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(content)
    return content


# ---------------------------------------------------------------------------
# Run both modes and emit two .tex files
# ---------------------------------------------------------------------------
_root = os.environ.get("OUTPUT_PLOTS_DIR")
OUT_DIR = (os.path.join(_root, "BW_summaries") if _root
           else "/eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw")
os.makedirs(OUT_DIR, exist_ok=True)

for mode in ("abs", "rel"):
    print(f"\n##### mode = {mode} #####")
    print("% (loading data; first run may take several minutes)")
    rows = []
    for cat_label, variants, path_fn in CATEGORIES:
        for flav in ("numu", "numubar"):
            for obs_key in ("hk_qe", "dune_epi", "dune_tpi"):
                med, mean = category_row(variants, path_fn, flav, obs_key, mode)
                rows.append((cat_label, flav, obs_key, med, mean))

    out_path = os.path.join(OUT_DIR, f"variation_table_{mode}.tex")
    content = emit_table(mode, rows, out_path)
    print(content)
    print(f"% wrote {out_path}")
