"""Print HK CC0π composition stats in LaTeX format.

For HK NuWro SF FSI nominal samples (νμ FHC and ν̄μ RHC), prints:
  - % of CCQE interactions in the CC0π sample
  - % of CC0π interactions of total CC
  - % of CC0π interactions of total (all interactions in the file)

CC0π is defined post-FSI: cc=1 AND no π± AND no π0 in the final-state pdg
list. CCQE is |Mode|==1 (Mode 1 = νμ QE, Mode -1 = ν̄μ QE).

Counts are fScaleFactor-weighted (cross-section style). Reads the same
canonical NuWro SF nominal files as fsi_iop_plots.py and falls back to
.bak.preOPpatch if the live file is missing -- so it can run while a regen
is in flight.
"""
import os
import sys
import numpy as np
import awkward as ak

# Share Jake's mtime-keyed cache via FlatTreeMod.load_arrays.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from FlatTreeMod import load_arrays  # noqa: E402

BASE = "/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs"

HK_FILES = {
    "numu":    f"{BASE}/HK/HK_numu_FSI.flat.root",
    "numubar": f"{BASE}/HK/HK_numubar_FSI.flat.root",
}


def resolve_path(path):
    """Live files only. The .bak fallback was removed once the OP-patched
    regen landed on 2026-05-03."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def compute_hk_cc0pi_stats(path):
    p = resolve_path(path)
    a = load_arrays(p)

    cc = np.asarray(a["cc"]).astype(bool)
    mode = np.asarray(a["Mode"])
    w = np.asarray(a["fScaleFactor"])

    apdg = np.abs(a["pdg"])
    n_chpi = ak.to_numpy(ak.sum(apdg == 211, axis=1))
    n_pi0  = ak.to_numpy(ak.sum(apdg == 111, axis=1))
    cc0pi = cc & (n_chpi == 0) & (n_pi0 == 0)
    is_qe = np.abs(mode) == 1

    sum_total      = float(w.sum())
    sum_cc         = float(w[cc].sum())
    sum_cc0pi      = float(w[cc0pi].sum())
    sum_ccqe_cc0pi = float(w[cc0pi & is_qe].sum())

    return {
        "ccqe_in_cc0pi": 100.0 * sum_ccqe_cc0pi / sum_cc0pi if sum_cc0pi else 0.0,
        "cc0pi_of_cc":   100.0 * sum_cc0pi / sum_cc if sum_cc else 0.0,
        "cc0pi_of_tot":  100.0 * sum_cc0pi / sum_total if sum_total else 0.0,
        "_n_events":     len(cc),
        "_path":         p,
    }


stats = {f: compute_hk_cc0pi_stats(p) for f, p in HK_FILES.items()}

print()
print(r"% --- HK CC0π composition (NuWro SF FSI nominal, fScaleFactor-weighted) ---")
for f in ("numu", "numubar"):
    print(rf"% {f:<8} N_events={stats[f]['_n_events']:>9d}  src=...{stats[f]['_path'][-50:]}")
print()
print(r"\begin{tabular}{lcc}")
print(r"\hline")
print(r"Quantity & $\nu_\mu$ FHC & $\bar\nu_\mu$ RHC \\")
print(r"\hline")
print(rf"CCQE / CC0$\pi$       & {stats['numu']['ccqe_in_cc0pi']:5.1f}\% & {stats['numubar']['ccqe_in_cc0pi']:5.1f}\% \\")
print(rf"CC0$\pi$ / CC         & {stats['numu']['cc0pi_of_cc']:5.1f}\% & {stats['numubar']['cc0pi_of_cc']:5.1f}\% \\")
print(rf"CC0$\pi$ / total      & {stats['numu']['cc0pi_of_tot']:5.1f}\% & {stats['numubar']['cc0pi_of_tot']:5.1f}\% \\")
print(r"\hline")
print(r"\end{tabular}")
