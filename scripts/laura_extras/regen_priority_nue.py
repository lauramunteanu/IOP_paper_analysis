"""Priority driver: nue/nuebar BW summary plots — Fig8-12 × {nue, nuebar} × {abs, rel}.

Mirrors the nue/nuebar block of fsi_iop_plots.py's __main__ but skips
numu/numubar (already produced earlier today). For Fig 8/9/10 the real
νe/ν̄e samples are used (lep_pdg=11); Fig 11/12 proxy through the νμ
NEUT/GENIE files with force_appearance=True (no real νe samples exist
for those cells)."""

import os
import fsi_iop_plots as m

os.makedirs(m.OUT_DIR, exist_ok=True)
print(f"MAX_EVENTS = {m.MAX_EVENTS}\n")

for mode in ("abs", "rel"):
    print(f"\n##### mode = {mode} (nue/nuebar subset) #####\n")
    for flavour in ("nue", "nuebar"):
        print(f"=== {flavour}[{mode}] FSI vs noFSI ===")
        m.make_fsi_compare_figure(flavour, mode, lep_pdg=11)
        print(f"=== {flavour}[{mode}] pi-abs ===")
        m.make_pi_abs_figure(flavour, mode, lep_pdg=11)
        print(f"=== {flavour}[{mode}] NN_mfp ===")
        m.make_mfp_compare_figure(flavour, mode, lep_pdg=11)
        print(f"=== {flavour}[{mode}] GENIE cascade tunes (numu proxy) ===")
        m.make_cascade_compare_figure(flavour, mode, lep_pdg=13,
                                       force_appearance=True)
        print(f"=== {flavour}[{mode}] NEUT EDRMF vs RPWIA (numu proxy) ===")
        m.make_edrmf_compare_figure(flavour, mode, lep_pdg=13,
                                     force_appearance=True)

print("\nDONE (nue/nuebar subset)")
