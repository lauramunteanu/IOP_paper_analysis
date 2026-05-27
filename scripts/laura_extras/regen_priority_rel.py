"""Priority driver: only the BW plots Laura asked for —
   Fig8-12 × {numu, numubar} × rel mode + standalone legends.

Imports the figure-rendering helpers from fsi_iop_plots and skips
nue/nuebar entirely. Run from scripts/laura_extras/ so the relative
input paths inside fsi_iop_plots resolve."""

import os
import fsi_iop_plots as m

os.makedirs(m.OUT_DIR, exist_ok=True)
print(f"MAX_EVENTS = {m.MAX_EVENTS}\n")
print(f"##### mode = rel (priority subset) #####\n")

for flavour in ("numu", "numubar"):
    print(f"=== {flavour}[rel] FSI vs noFSI ===")
    m.make_fsi_compare_figure(flavour, "rel")
    print(f"=== {flavour}[rel] pi-abs ===")
    m.make_pi_abs_figure(flavour, "rel")
    print(f"=== {flavour}[rel] NN_mfp ===")
    m.make_mfp_compare_figure(flavour, "rel")
    print(f"=== {flavour}[rel] GENIE cascade tunes ===")
    m.make_cascade_compare_figure(flavour, "rel")
    print(f"=== {flavour}[rel] NEUT EDRMF vs RPWIA ===")
    m.make_edrmf_compare_figure(flavour, "rel")

print("\n=== standalone legends ===")
m._save_all_legends()
print("\nDONE (priority subset)")
