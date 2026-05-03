"""For events where the vertex stack and post-FSI stack contain *identical*
particle multiplicities (no FSI rescatter happened — only the binding/separation
energy adjustment), compare the per-nucleon energy difference."""
from FlatTreeMod import *
import numpy as np
import awkward as ak

filename = "../../Remade_April26/DUNE/DUNE_numu_FSI.flat.root"
arr = load_arrays(filename, max_events=200000)
ninitp = ak.to_numpy(arr["ninitp"])

# Build outgoing-only vertex pdg jagged
keep_v = ak.local_index(arr["pdg_vert"], axis=1) >= arr["ninitp"]
pdg_vert_out = arr["pdg_vert"][keep_v]
E_vert_out   = arr["E_vert"][keep_v]
pdg_post     = arr["pdg"]
E_post       = arr["E"]

cc = np.asarray(arr["cc"], dtype=bool)
n_evt = len(arr["Enu_true"])

# multiplicity counters
def counts_per_event(pdgs, ids):
    """Return (n_evt, len(ids)) array of counts per (event, species)."""
    out = np.zeros((n_evt, len(ids)), dtype=int)
    for j, pdg in enumerate(ids):
        out[:, j] = ak.to_numpy(ak.sum(pdgs == pdg, axis=1))
    return out

species = [13, -13, 11, -11, 22, 211, -211, 111, 2212, 2112, 14, -14]
mv = counts_per_event(pdg_vert_out, species)
mp = counts_per_event(pdg_post,     species)
identical = cc & np.all(mv == mp, axis=1)
print(f"CC events: {cc.sum()}")
print(f"CC events with identical species multiplicity (vertex outgoing == post-FSI): {identical.sum()}")

# For these events, compute (sum of nucleon E)_vertex - (sum of nucleon E)_post
# only for nucleons (proton+neutron). The expected difference: ~30 MeV per outgoing nucleon.
is_nuc_v = (abs(pdg_vert_out) == 2212) | (abs(pdg_vert_out) == 2112)
is_nuc_p = (abs(pdg_post)     == 2212) | (abs(pdg_post)     == 2112)
sum_E_nuc_v = ak.to_numpy(ak.sum(E_vert_out * is_nuc_v, axis=1)) * 1000.0
sum_E_nuc_p = ak.to_numpy(ak.sum(E_post     * is_nuc_p, axis=1)) * 1000.0
n_nuc_v = ak.to_numpy(ak.sum(is_nuc_v, axis=1))

dE = (sum_E_nuc_v - sum_E_nuc_p)[identical]
n_nuc = n_nuc_v[identical]
mask = n_nuc > 0
per_nuc = dE[mask] / n_nuc[mask]

print()
print("== per-event total nucleon E_vertex - E_postFSI (MeV), identical-species CC events ==")
print(f"  mean    = {dE[mask].mean():.2f}")
print(f"  median  = {np.median(dE[mask]):.2f}")
print(f"  std     = {dE[mask].std():.2f}")
print()
print("== per outgoing nucleon (MeV), same events ==")
print(f"  mean    = {per_nuc.mean():.2f}")
print(f"  median  = {np.median(per_nuc):.2f}")
print(f"  std     = {per_nuc.std():.2f}")
hist, edges = np.histogram(per_nuc, bins=np.arange(-10, 100, 5))
for c, lo, hi in zip(hist, edges[:-1], edges[1:]):
    bar = "#" * int(round(40 * c / max(hist.max(), 1)))
    print(f"  [{lo:>4.0f},{hi:>4.0f}) {c:>6}  {bar}")
print()
print(f"  fraction with dE_per_nuc < 0:        {(per_nuc < 0).mean()*100:.1f}%")
print(f"  fraction with 10 < dE_per_nuc < 50:  {((per_nuc >= 10) & (per_nuc <= 50)).mean()*100:.1f}%")
