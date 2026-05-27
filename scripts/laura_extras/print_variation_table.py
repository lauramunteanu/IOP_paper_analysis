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
from FlatTreeMod import (load_arrays, bias_arr, is_cc0pi_arr,  # noqa: E402
                         osc_weights_mumu, osc_weights_mue, bias_sel_arr)

# ---------------------------------------------------------------------------
# Configuration (kept in sync with fsi_iop_plots.py)
# ---------------------------------------------------------------------------
BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs/nuwro_25031_morestats"
NEUT_BASE = "/eos/home-l/lamuntea/FSI_IOP_paper/neut_runs"
GENIE_FILES_BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs"
MAX_EVENTS = None

# Per-mode shading thresholds. Stored in display units: abs in MeV, rel in
# percent. rel = 0.5% for both HK and DUNE matches the critical bias level
# established in the bias_level_study (cell is flagged if the displayed value
# is >= 0.5%).
THRESH_BY_MODE = {
    "abs": {"hk": 5.0,   "dune": 15.0},
    "rel": {"hk": 0.5,   "dune": 0.5},
}

UNITS_BY_MODE = {
    "abs": "[MeV]",
    "rel": r"[\%]",
}


NUWRO_HK = {
    "numu":    {"FSI":   f"{BASE}/HK/HK_numu_FSI.flat.root",
                "noFSI": f"{BASE}/HK/HK_numu_noFSI.flat.root"},
    "numubar": {"FSI":   f"{BASE}/HK/HK_numubar_FSI.flat.root",
                "noFSI": f"{BASE}/HK/HK_numubar_noFSI.flat.root"},
    "nue":     {"FSI":   f"{BASE}/HK/HK_nue_FSI.flat.root",
                "noFSI": f"{BASE}/HK/HK_nue_noFSI.flat.root"},
    "nuebar":  {"FSI":   f"{BASE}/HK/HK_nuebar_FSI.flat.root",
                "noFSI": f"{BASE}/HK/HK_nuebar_noFSI.flat.root"},
}
NUWRO_DUNE = {
    "numu":    {"FSI":   f"{BASE}/DUNE/DUNE_numu_FSI.flat.root",
                "noFSI": f"{BASE}/DUNE/DUNE_numu_noFSI.flat.root"},
    "numubar": {"FSI":   f"{BASE}/DUNE/DUNE_numub_FSI.flat.root",
                "noFSI": f"{BASE}/DUNE/DUNE_numub_noFSI.flat.root"},
    "nue":     {"FSI":   f"{BASE}/DUNE/DUNE_nue_FSI.flat.root",
                "noFSI": f"{BASE}/DUNE/DUNE_nue_noFSI.flat.root"},
    "nuebar":  {"FSI":   f"{BASE}/DUNE/DUNE_nueb_FSI.flat.root",
                "noFSI": f"{BASE}/DUNE/DUNE_nueb_noFSI.flat.root"},
}
NUWRO_HK_PIABS = {
    ("numu",    "069"): f"{BASE}/HK/piabs/HK_numu_piabs069_FSI.flat.root",
    ("numu",    "131"): f"{BASE}/HK/piabs/HK_numu_piabs131_FSI.flat.root",
    ("numubar", "069"): f"{BASE}/HK/piabs/HK_numubar_piabs069_FSI.flat.root",
    ("numubar", "131"): f"{BASE}/HK/piabs/HK_numubar_piabs131_FSI.flat.root",
    ("nue",     "069"): f"{BASE}/HK/piabs/HK_nue_piabs069_FSI.flat.root",
    ("nue",     "131"): f"{BASE}/HK/piabs/HK_nue_piabs131_FSI.flat.root",
    ("nuebar",  "069"): f"{BASE}/HK/piabs/HK_nuebar_piabs069_FSI.flat.root",
    ("nuebar",  "131"): f"{BASE}/HK/piabs/HK_nuebar_piabs131_FSI.flat.root",
}
NUWRO_DUNE_PIABS = {
    ("numu",    "069"): f"{BASE}/DUNE/piabs/DUNE_numu_piabs069_FSI.flat.root",
    ("numu",    "131"): f"{BASE}/DUNE/piabs/DUNE_numu_piabs131_FSI.flat.root",
    ("numubar", "069"): f"{BASE}/DUNE/piabs/DUNE_numub_piabs069_FSI.flat.root",
    ("numubar", "131"): f"{BASE}/DUNE/piabs/DUNE_numub_piabs131_FSI.flat.root",
    ("nue",     "069"): f"{BASE}/DUNE/piabs/DUNE_nue_piabs069_FSI.flat.root",
    ("nue",     "131"): f"{BASE}/DUNE/piabs/DUNE_nue_piabs131_FSI.flat.root",
    ("nuebar",  "069"): f"{BASE}/DUNE/piabs/DUNE_nueb_piabs069_FSI.flat.root",
    ("nuebar",  "131"): f"{BASE}/DUNE/piabs/DUNE_nueb_piabs131_FSI.flat.root",
}
NUWRO_HK_MFP = {
    ("numu",    "0p7"): f"{BASE}/HK/ChangeMFP/HK_numu_0p7MFP_FSI.flat.root",
    ("numu",    "1p3"): f"{BASE}/HK/ChangeMFP/HK_numu_1p3MFP_FSI.flat.root",
    ("numubar", "0p7"): f"{BASE}/HK/ChangeMFP/HK_numubar_0p7MFP_FSI.flat.root",
    ("numubar", "1p3"): f"{BASE}/HK/ChangeMFP/HK_numubar_1p3MFP_FSI.flat.root",
    ("nue",     "0p7"): f"{BASE}/HK/ChangeMFP/HK_nue_0p7MFP_FSI.flat.root",
    ("nue",     "1p3"): f"{BASE}/HK/ChangeMFP/HK_nue_1p3MFP_FSI.flat.root",
    ("nuebar",  "0p7"): f"{BASE}/HK/ChangeMFP/HK_nuebar_0p7MFP_FSI.flat.root",
    ("nuebar",  "1p3"): f"{BASE}/HK/ChangeMFP/HK_nuebar_1p3MFP_FSI.flat.root",
}
NUWRO_DUNE_MFP = {
    ("numu",    "0p7"): f"{BASE}/DUNE/ChangeMFP/DUNE_numu_0p7MFP_FSI.flat.root",
    ("numu",    "1p3"): f"{BASE}/DUNE/ChangeMFP/DUNE_numu_1p3MFP_FSI.flat.root",
    ("numubar", "0p7"): f"{BASE}/DUNE/ChangeMFP/DUNE_numub_0p7MFP_FSI.flat.root",
    ("numubar", "1p3"): f"{BASE}/DUNE/ChangeMFP/DUNE_numub_1p3MFP_FSI.flat.root",
    ("nue",     "0p7"): f"{BASE}/DUNE/ChangeMFP/DUNE_nue_0p7MFP_FSI.flat.root",
    ("nue",     "1p3"): f"{BASE}/DUNE/ChangeMFP/DUNE_nue_1p3MFP_FSI.flat.root",
    ("nuebar",  "0p7"): f"{BASE}/DUNE/ChangeMFP/DUNE_nueb_0p7MFP_FSI.flat.root",
    ("nuebar",  "1p3"): f"{BASE}/DUNE/ChangeMFP/DUNE_nueb_1p3MFP_FSI.flat.root",
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
    "hk_qe":    r"Hyper-K $E_{\nu}^{\mathrm{QE}}$",
    "dune_epi": r"DUNE $E_{\nu}^{\mathrm{avail}}$",
    "dune_tpi": r"DUNE $E_{\nu}^{\mathrm{had}}$",
}
FLAV_LATEX = {
    "numu":    r"$\nu_\mu$",
    "numubar": r"$\bar\nu_\mu$",
    "nue":     r"$\nu_e$",
    "nuebar":  r"$\bar\nu_e$",
}
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
def _read_hk(path, mode, lep_pdg=13, force_appearance=False):
    a = load_arrays(path, max_events=MAX_EVENTS)
    x = bias_arr(a, "qe", kind=mode, vertex=False, lep_pdg=lep_pdg)
    sel = is_cc0pi_arr(a, vertex=False, lep_pdg=lep_pdg)
    w_xsec = np.asarray(a["fScaleFactor"][sel])
    Enu_t  = np.asarray(a['Enu_true'])[sel]
    # force_appearance: events have μ final state (lep_pdg=13) but the osc
    # weight is the νμ→νe appearance probability anyway. Used by the GENIE
    # numu-as-nue proxy — no real νe GENIE files exist.
    if force_appearance:
        osc_fn = osc_weights_mue
    else:
        osc_fn = osc_weights_mue if lep_pdg == 11 else osc_weights_mumu
    w = w_xsec * osc_fn(Enu_t, filename=path)
    return x, w


def _read_dune(path, mode, observable, lep_pdg=13, force_appearance=False):
    """observable in {'had', 'avail'}."""
    a = load_arrays(path, max_events=MAX_EVENTS)
    x = bias_arr(a, observable, kind=mode, vertex=False, lep_pdg=lep_pdg)
    sel = bias_sel_arr(a, observable, vertex=False, lep_pdg=lep_pdg)
    w_xsec = np.asarray(a["fScaleFactor"])[sel]
    Enu_t  = np.asarray(a['Enu_true'])[sel]
    if force_appearance:
        osc_fn = osc_weights_mue
    else:
        osc_fn = osc_weights_mue if lep_pdg == 11 else osc_weights_mumu
    w = w_xsec * osc_fn(Enu_t, filename=path)
    return x, w


_CACHE = {}


def get_xw(path, obs_key, mode, lep_pdg=13, force_appearance=False):
    """Returns (x, w) or None if file unresolvable."""
    p = resolve_path(path)
    if p is None:
        return None
    key = (p, obs_key, mode, lep_pdg, force_appearance)
    if key not in _CACHE:
        if obs_key == "hk_qe":
            _CACHE[key] = _read_hk(p, mode, lep_pdg=lep_pdg,
                                    force_appearance=force_appearance)
        else:
            observable = "had" if obs_key == "dune_tpi" else "avail"
            _CACHE[key] = _read_dune(p, mode, observable, lep_pdg=lep_pdg,
                                      force_appearance=force_appearance)
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
    # Real GENIE νe / ν̄e cascade samples don't exist for this paper.
    # Proxy: read the matching GENIE νμ / ν̄μ file and reweight per-event as
    # νμ→νe appearance (handled by force_appearance=True in the main loop).
    # The events still have μ in the final state — this captures the cascade-
    # model spread under νe-appearance flux weighting only.
    src_flav = {"nue": "numu", "nuebar": "numubar"}.get(flav, flav)
    return _hk_or_dune(obs_key, GENIE_HK[(src_flav, tune)], GENIE_DUNE[(src_flav, tune)])


def edrmf_paths(flav, obs_key, kind):
    # Real NEUT EDRMF/RPWIA samples exist only for FHC νμ. For the νe
    # channel proxy through the numu NEUT files (force_appearance=True in
    # selection_for). No numubar NEUT samples exist, so ν̄e stays empty.
    catalogue = NEUT_EDRMF if kind == "EDRMF" else NEUT_RPWIA
    src_flav = {"nue": "numu", "nuebar": "numubar"}.get(flav, flav)
    return catalogue[(src_flav, "hk")] if obs_key == "hk_qe" else catalogue[(src_flav, "dune")]


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
def category_row(variants, path_fn, flav, obs_key, mode, lep_pdg=13,
                 force_appearance=False):
    """Pick the (max_mean, min_mean) variant pair; report median and mean
    range for THAT pair."""
    medians, means = [], []
    for v in variants:
        path = path_fn(flav, obs_key, v)
        res = get_xw(path, obs_key, mode, lep_pdg=lep_pdg,
                     force_appearance=force_appearance)
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
# First column = lookup key matching CATEGORIES; second column = header text.
CAT_HEADERS = [
    (r"FSI vs no FSI",                    r"FSI / no FSI"),
    (r"$\pi_{\mathrm{abs}}$ $\pm 31\%$",  r"$\pi_{\mathrm{abs}}\pm31\%$"),
    (r"NN MFP $\pm 30\%$",                r"NN MFP $\pm30\%$"),
    (r"GENIE cascade (10a-10d)",          r"INC model var."),
    (r"EDRMF vs RPWIA",                   r"Nuc. Pot. on/off"),
]
OBS_ORDER = ["hk_qe", "dune_epi", "dune_tpi"]


def fmt(v, thresh, mode):
    """Format one cell value in display units (rel: percent; abs: MeV) and
    shade the cell red if its rounded display value is >= the threshold."""
    if v is None or not np.isfinite(v):
        return "--"
    disp = v * 100.0 if mode == "rel" else v
    # Compare on the value rounded to display precision so a cell shown as
    # 0.5 always crosses the 0.5% threshold regardless of FP noise.
    flagged = round(disp, 1) >= round(thresh, 1)
    text = f"{disp:.1f}"
    return rf"\cellcolor{{red!12}}{text}" if flagged else text


# Captions for the two tables. Both use `Hyper-K' / `DUNE' / `INC model var.'
# / `Nuc. Pot.' terminology in the paper text. The abs caption mentions the
# 5 / 15 MeV thresholds; the rel caption swaps that for the 0.5% common one.
_CAPTION_ABS = r"The maximum shift in the mean and median neutrino energy estimation bias due to different FSI variations for the Hyper-K and DUNE neutrino and antineutrino cases. The numbers reported for the INC model variation is derived from the two INC models that give the largest spread in the mean. Red boxes indicate that the variation is larger than 5 MeV or 15 MeV for the Hyper-K and DUNE cases respectively, which is broadly indicative of how well the neutrino energy reconstruction scale must be controlled (see \autoref{sec:enurec}). ``Nuc. Pot.'' stands for the nuclear potential considered in \autoref{subsec:beyondcasc}, for which the table reports a shift derived considering only CCQE interactions."

_CAPTION_REL = r"The maximum shift in the mean and median neutrino energy estimation bias (expressed as a fraction of $E_\nu^{\rm true}$, in percent) due to different FSI variations for the Hyper-K and DUNE neutrino and antineutrino cases. The numbers reported for the INC model variation is derived from the two INC models that give the largest spread in the mean. Red boxes indicate that the variation is larger than $0.5\%$, which is broadly indicative of how well the neutrino energy reconstruction scale must be controlled (see \autoref{sec:enurec}). ``Nuc. Pot.'' stands for the nuclear potential considered in \autoref{subsec:beyondcasc}, for which the table reports a shift derived considering only CCQE interactions."
CAPTION_BY_MODE = {"abs": _CAPTION_ABS, "rel": _CAPTION_REL}
LABEL_BY_MODE   = {"abs": r"tab:FSIVar",  "rel": r"tab:FSIVar_rel"}


def _label_for(mode, flavours):
    """LaTeX label for the (mode, flavours) combination. Distinct per channel."""
    if flavours == ("numu", "numubar"):
        return LABEL_BY_MODE[mode]
    if flavours == ("nue", "nuebar"):
        return f"{LABEL_BY_MODE[mode]}_nue"
    return f"{LABEL_BY_MODE[mode]}_{'_'.join(flavours)}"


def emit_table(mode, rows, out_path, flavours=("numu", "numubar")):
    """Build the LaTeX table for one bias mode and write it to out_path.

    The emitted .tex is a full ``\\begin{table}[tb]`` block wrapping the
    tabular -- ready to drop into the paper via ``\\input{}`` -- with the
    caption / label / centering / size matching the FSIVar template.
    """
    units = UNITS_BY_MODE[mode]
    thresh_by_exp = THRESH_BY_MODE[mode]

    by_key = {(r[0], r[1], r[2]): (r[3], r[4]) for r in rows}

    lines = []
    lines.append(rf"% Variation table ({mode}) -- requires:")
    lines.append(r"%   \usepackage{booktabs}")
    lines.append(r"%   \usepackage{multirow}")
    lines.append(r"%   \usepackage[table]{xcolor}")
    lines.append(r"%   \usepackage{graphicx}   % for \rotatebox")
    lines.append(r"%   \usepackage{hyperref}   % for \autoref in caption")
    lines.append("")
    lines.append(r"\begin{table}[tb]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    n_cat = len(CAT_HEADERS)
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

    for flav_idx, flav in enumerate(flavours):
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
                cells.append(fmt(med, thresh, mode))
                cells.append(fmt(mean, thresh, mode))
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\normalsize")
    lines.append(rf"\caption{{{CAPTION_BY_MODE[mode]}}}")
    lines.append(rf"\label{{{_label_for(mode, flavours)}}}")
    lines.append(r"\end{table}")

    content = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(content)
    return content


# Combined-table caption mentions both threshold conventions in one breath.
_CAPTION_COMBINED = r"The shift in the mean and median neutrino energy estimation bias due to different FSI variations for the Hyper-K and DUNE neutrino and antineutrino cases. The top half reports the absolute shift in MeV, and the bottom half the same shift expressed as a fraction of $E_\nu^{\rm true}$ (in percent). The numbers reported for the INC model variation are derived from the two INC models that give the largest spread in the mean. Red boxes indicate that the variation is larger than 5\,MeV / 15\,MeV (absolute, Hyper-K and DUNE respectively) or $0.5\%$ (relative), which is broadly indicative of how well the neutrino energy reconstruction scale must be controlled (see \autoref{sec:enurec}). ``Nuc. Pot.'' stands for the nuclear potential considered in \autoref{subsec:beyondcasc}, for which the table reports a shift derived considering only CCQE interactions."


def _emit_data_rows(lines, by_key, mode):
    """Emit the 6 data rows (2 flavours × 3 observables) for one bias mode."""
    thresh_by_exp = THRESH_BY_MODE[mode]
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
                cells.append(fmt(med, thresh, mode))
                cells.append(fmt(mean, thresh, mode))
            lines.append(" & ".join(cells) + r" \\")


def emit_combined_table(rows_abs, rows_rel, out_path):
    """Build a single LaTeX table holding both the abs and rel variation
    rows (12 data rows total) and write it to out_path.

    Layout: shared header (no units row, since units differ per section),
    then a ``Absolute bias [MeV]'' section header spanning the full width,
    then the 6 abs rows, then a ``Relative bias [\\%]'' section header,
    then the 6 rel rows.
    """
    by_key_abs = {(r[0], r[1], r[2]): (r[3], r[4]) for r in rows_abs}
    by_key_rel = {(r[0], r[1], r[2]): (r[3], r[4]) for r in rows_rel}

    n_cat = len(CAT_HEADERS)
    n_cols = 2 + 2 * n_cat  # 12

    lines = []
    lines.append(r"% Combined variation table (abs + rel) -- requires:")
    lines.append(r"%   \usepackage{booktabs}")
    lines.append(r"%   \usepackage{multirow}")
    lines.append(r"%   \usepackage[table]{xcolor}")
    lines.append(r"%   \usepackage{graphicx}   % for \rotatebox")
    lines.append(r"%   \usepackage{hyperref}   % for \autoref in caption")
    lines.append("")
    lines.append(r"\begin{table}[tb]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
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
    lines.append(r"\midrule")

    # ---- Absolute section ----
    lines.append(rf"\multicolumn{{{n_cols}}}{{c}}{{\textbf{{Absolute bias [MeV]}}}} \\")
    lines.append(r"\midrule")
    _emit_data_rows(lines, by_key_abs, "abs")

    # ---- Relative section ----
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{n_cols}}}{{c}}{{\textbf{{Relative bias [\%]}}}} \\")
    lines.append(r"\midrule")
    _emit_data_rows(lines, by_key_rel, "rel")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\normalsize")
    lines.append(rf"\caption{{{_CAPTION_COMBINED}}}")
    lines.append(r"\label{tab:FSIVar_combined}")
    lines.append(r"\end{table}")

    content = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(content)
    return content


# ---------------------------------------------------------------------------
# Run both modes and emit two .tex files (+ a combined one)
# ---------------------------------------------------------------------------
_root = os.environ.get("OUTPUT_PLOTS_DIR")
OUT_DIR = (os.path.join(_root, "BW_summaries") if _root
           else "/eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw")
os.makedirs(OUT_DIR, exist_ok=True)

# Channels: (label, flavour pair). νμ-channel: real νμ/ν̄μ samples with
# disappearance weighting. νe-channel: real νe/ν̄e samples with appearance
# weighting (or GENIE-numu-as-νe proxy where real νe samples don't exist).
CHANNELS = [
    ("numu", ("numu",  "numubar")),
    ("nue",  ("nue",   "nuebar")),
]


def selection_for(channel, cat_label):
    """Return (lep_pdg, force_appearance) for (channel, category).

    Most cells follow the natural rule: numu channel → (13, False); nue
    channel → (11, False) on real νe/ν̄e samples.

    The exception is the GENIE-cascade row in the nue channel: real GENIE
    νe / ν̄e samples don't exist, so we proxy with the GENIE νμ / ν̄μ files
    (events have μ final state — selection stays at lep_pdg=13) but apply
    νμ→νe appearance weighting via force_appearance=True. The cell reflects
    the cascade-model spread under appearance-flux weighting; bias kinematics
    are νμ-CC, not νe-CC."""
    # GENIE νe and EDRMF/RPWIA νe both proxy through νμ NEUT/GENIE files
    # (no real νe samples for either). Events are νμ-CC (μ final state) →
    # lep_pdg=13 for the selection, force_appearance=True for the osc weight.
    if channel == "nue" and (cat_label.startswith("GENIE")
                              or cat_label.startswith("EDRMF")):
        return (13, True)
    if channel == "nue":
        return (11, False)
    return (13, False)


_all_rows = {}
for channel, flavours in CHANNELS:
    for mode in ("abs", "rel"):
        print(f"\n##### channel = {channel}  mode = {mode} #####")
        print("% (loading data; first run may take several minutes)")
        rows = []
        for cat_label, variants, path_fn in CATEGORIES:
            lp, force_app = selection_for(channel, cat_label)
            for flav in flavours:
                for obs_key in ("hk_qe", "dune_epi", "dune_tpi"):
                    med, mean = category_row(variants, path_fn, flav,
                                              obs_key, mode, lep_pdg=lp,
                                              force_appearance=force_app)
                    rows.append((cat_label, flav, obs_key, med, mean))
        _all_rows[(channel, mode)] = rows

        out_path = os.path.join(OUT_DIR, f"variation_table_{channel}_{mode}.tex")
        content = emit_table(mode, rows, out_path, flavours=flavours)
        print(content)
        print(f"% wrote {out_path}")

# Combined table — kept for backwards compat: numu channel only, abs + rel.
# Guard: only emit if the numu channel was iterated this run. Allows
# nue-only runs to skip the combined table (its content is unchanged).
if ("numu", "abs") in _all_rows and ("numu", "rel") in _all_rows:
    _combined_path = os.path.join(OUT_DIR, "variation_table_combined.tex")
    emit_combined_table(_all_rows[("numu", "abs")], _all_rows[("numu", "rel")],
                        _combined_path)
    print(f"\n% wrote {_combined_path}")
else:
    print("\n% skipping combined table — numu channel not iterated this run")
