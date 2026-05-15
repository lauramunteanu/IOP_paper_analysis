from FlatTreeMod import *
import numpy as np
import awkward as ak

filename = "../../Remade_April26/nuwro_25031_morestats/DUNE/DUNE_numu_FSI.flat.root"
arr = load_arrays(filename, max_events=2000)

ninitp = ak.to_numpy(arr["ninitp"])
def has_chpi_vertex(i):
    pdg_vt = arr["pdg_vert"][i]
    keep = ak.local_index(pdg_vt, axis=0) >= int(ninitp[i])
    pdg_out = ak.to_numpy(pdg_vt[keep])
    return np.any(np.abs(pdg_out) == 211)

cc = np.asarray(arr["cc"], dtype=bool)
candidates = [i for i in range(len(arr["Enu_true"])) if cc[i] and has_chpi_vertex(i)][:6]
print("picked event idxs:", candidates)

NAMES = {2212:"p", 2112:"n", 211:"pi+", -211:"pi-", 111:"pi0",
         13:"mu-", -13:"mu+", 11:"e-", -11:"e+", 22:"gam",
         14:"nu_mu", -14:"nubar_mu",
         311:"K0", 321:"K+", -321:"K-", 130:"K0L", 310:"K0S",
         2114:"Delta0", 1114:"Delta-", 2214:"Delta+", 2224:"Delta++"}

def fmt_pdg(p):
    return NAMES.get(int(p), str(int(p)))

def stack(pdgs, Es, pxs, pys, pzs):
    pdgs = ak.to_numpy(pdgs); Es = ak.to_numpy(Es)
    pxs = ak.to_numpy(pxs); pys = ak.to_numpy(pys); pzs = ak.to_numpy(pzs)
    masses = np.sqrt(np.maximum(Es*Es - (pxs*pxs+pys*pys+pzs*pzs), 0.0))
    return pdgs, Es, masses

def contrib(pdg, E, mass, definition):
    apdg = abs(pdg)
    is_p    = apdg == 2212
    is_chpi = apdg == 211
    is_pi0  = apdg == 111
    is_e    = apdg == 11
    is_gam  = apdg == 22
    if definition == "wo":
        if is_p or is_chpi:      return E - mass
        if is_pi0 or is_e or is_gam: return E
    else:
        if is_p:                 return E - mass
        if is_chpi or is_pi0 or is_e or is_gam: return E
    return 0.0

HDR = "  {0:<8} {1:>10}  {2:>8}  {3:>10}  {4:>10}".format("particle", "E", "m", "+wo", "+with")

for idx in candidates:
    Enu_true = float(arr["Enu_true"][idx]) * 1000.0
    ELep     = float(arr["ELep"][idx])     * 1000.0
    print()
    print("========== event {0}  Enu_true={1:.1f} MeV  ELep={2:.1f} MeV ==========".format(idx, Enu_true, ELep))

    np_init = int(ninitp[idx])
    keep_v = ak.local_index(arr["pdg_vert"][idx], axis=0) >= np_init
    pdgs_v, Es_v, m_v = stack(arr["pdg_vert"][idx][keep_v], arr["E_vert"][idx][keep_v],
                              arr["px_vert"][idx][keep_v], arr["py_vert"][idx][keep_v], arr["pz_vert"][idx][keep_v])
    pdgs_p, Es_p, m_p = stack(arr["pdg"][idx], arr["E"][idx],
                              arr["px"][idx], arr["py"][idx], arr["pz"][idx])

    print(" --- vertex (noFSI) stack ---")
    print(HDR)
    sum_wo_v = sum_with_v = 0.0
    for pdg, E, m in zip(pdgs_v, Es_v, m_v):
        E_M, m_M = E*1000.0, m*1000.0
        cwo   = contrib(pdg, E_M, m_M, "wo")
        cwith = contrib(pdg, E_M, m_M, "with")
        sum_wo_v += cwo; sum_with_v += cwith
        print("  {0:<8} {1:>10.1f}  {2:>8.1f}  {3:>10.1f}  {4:>10.1f}".format(
            fmt_pdg(pdg), E_M, m_M, cwo, cwith))

    print(" --- post-FSI stack ---")
    print(HDR)
    sum_wo_p = sum_with_p = 0.0
    for pdg, E, m in zip(pdgs_p, Es_p, m_p):
        E_M, m_M = E*1000.0, m*1000.0
        cwo   = contrib(pdg, E_M, m_M, "wo")
        cwith = contrib(pdg, E_M, m_M, "with")
        sum_wo_p += cwo; sum_with_p += cwith
        print("  {0:<8} {1:>10.1f}  {2:>8.1f}  {3:>10.1f}  {4:>10.1f}".format(
            fmt_pdg(pdg), E_M, m_M, cwo, cwith))

    print(" recoil_wo:   noFSI={0:.1f}   FSI={1:.1f}   Enu_true={2:.1f}".format(
        ELep + sum_wo_v, ELep + sum_wo_p, Enu_true))
    print(" bias_wo:     noFSI={0:+.1f}   FSI={1:+.1f}".format(
        (ELep + sum_wo_v) - Enu_true, (ELep + sum_wo_p) - Enu_true))
    print(" bias_with:   noFSI={0:+.1f}   FSI={1:+.1f}".format(
        (ELep + sum_with_v) - Enu_true, (ELep + sum_with_p) - Enu_true))
