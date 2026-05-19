"""Standalone-strip legends for the Fig1 spectrum-figure pairs.

Output (under scripts/Fig1_plots/):
  legend_Fig1_HK_dm32.{png,pdf}             — HK Δm²₃₂ variation + ±5 MeV shift
  legend_Fig1_HK_dCP.{png,pdf}              — HK δ_CP variation + ±5 MeV shift
  legend_Fig1_DUNE_dm32.{png,pdf}           — DUNE Δm²₃₂ variation + ±15 MeV shift
  legend_Fig1_DUNE_dCP.{png,pdf}            — DUNE δ_CP variation + ±15 MeV shift
  legend_Fig1_HK_dm32_relshift.{png,pdf}    — HK Δm²₃₂ variation + ±0.5%·E^true shift
  legend_Fig1_HK_dCP_relshift.{png,pdf}     — HK δ_CP  variation + ±0.5%·E^true shift
  legend_Fig1_DUNE_dm32_relshift.{png,pdf}  — DUNE Δm²₃₂ variation + ±0.5%·E^true shift
  legend_Fig1_DUNE_dCP_relshift.{png,pdf}   — DUNE δ_CP  variation + ±0.5%·E^true shift

Each is a 3-column horizontal-strip figure (no frame): col 1 = Nominal
(centered vertically), col 2 = osc-parameter variants stacked, col 3 =
energy-shift variants stacked. Sized to span a side-by-side
EnuTrue+EnuQE/Enuhad pair at \\linewidth in LaTeX.
"""
import os
from FlatTreeMod import (
    save_spectra_legend,
    tol_dark, osc_inc_color, osc_dec_color, pastel_red, pastel_blue,
)
from matplotlib.lines import Line2D

OUT_DIR = os.path.join(
    os.environ.get("OUTPUT_PLOTS_DIR",
                   os.path.dirname(os.path.abspath(__file__))),
    "Fig1_plots",
)


def line(color, label, ls='-', lw=1.6):
    return Line2D([0], [0], color=color, lw=lw, linestyle=ls, label=label)


def shift_line(color, label):
    """Energy-shift legend entry -- dashed to match the dashed shift curves
    drawn by FlatTreeMod.plot_osc_shift_e in the corresponding spectra."""
    return line(color, label, ls='--')


def hk_dm32():
    nominal = line(tol_dark, r"Nominal $\Delta m^{2}_{32} = 2.437 \times 10^{-3}$ eV$^{2}$")
    osc = [
        line(osc_inc_color, r"$\Delta m^{2}_{32} + 0.4\%$"),
        line(osc_dec_color, r"$\Delta m^{2}_{32} - 0.4\%$"),
    ]
    shifts = [
        shift_line(pastel_red,  r"$E_{\nu}^{\rm QE} + 5$ MeV"),
        shift_line(pastel_blue, r"$E_{\nu}^{\rm QE} - 5$ MeV"),
    ]
    return nominal, osc, shifts


def hk_dcp():
    nominal = line(tol_dark, r"Nominal $\delta_{CP} = -\pi/2$")
    osc = [
        line(osc_inc_color, r"$\delta_{CP} + 20^{\circ}$"),
        line(osc_dec_color, r"$\delta_{CP} - 20^{\circ}$"),
    ]
    shifts = [
        shift_line(pastel_red,  r"$E_{\nu}^{\rm QE} + 5$ MeV"),
        shift_line(pastel_blue, r"$E_{\nu}^{\rm QE} - 5$ MeV"),
    ]
    return nominal, osc, shifts


def dune_dm32():
    nominal = line(tol_dark, r"Nominal $\Delta m^{2}_{32} = 2.437 \times 10^{-3}$ eV$^{2}$")
    osc = [
        line(osc_inc_color, r"$\Delta m^{2}_{32} + 0.4\%$"),
        line(osc_dec_color, r"$\Delta m^{2}_{32} - 0.4\%$"),
    ]
    shifts = [
        shift_line(pastel_red,  r"$E_{\nu}^{\rm had} + 15$ MeV"),
        shift_line(pastel_blue, r"$E_{\nu}^{\rm had} - 15$ MeV"),
    ]
    return nominal, osc, shifts


def dune_dcp():
    nominal = line(tol_dark, r"Nominal $\delta_{CP} = -\pi/2$")
    osc = [
        line(osc_inc_color, r"$\delta_{CP} + 20^{\circ}$"),
        line(osc_dec_color, r"$\delta_{CP} - 20^{\circ}$"),
    ]
    shifts = [
        shift_line(pastel_red,  r"$E_{\nu}^{\rm had} + 15$ MeV"),
        shift_line(pastel_blue, r"$E_{\nu}^{\rm had} - 15$ MeV"),
    ]
    return nominal, osc, shifts


# --- relshift variants: same osc lines, ±0.5%·E_nu^true shift instead of ±5/15 MeV.
def hk_dm32_relshift():
    nominal, osc, _ = hk_dm32()
    shifts = [
        shift_line(pastel_red,  r"$E_{\nu}^{\rm QE} + 0.5\%\, E_{\nu}^{\rm true}$"),
        shift_line(pastel_blue, r"$E_{\nu}^{\rm QE} - 0.5\%\, E_{\nu}^{\rm true}$"),
    ]
    return nominal, osc, shifts


def hk_dcp_relshift():
    nominal, osc, _ = hk_dcp()
    shifts = [
        shift_line(pastel_red,  r"$E_{\nu}^{\rm QE} + 0.5\%\, E_{\nu}^{\rm true}$"),
        shift_line(pastel_blue, r"$E_{\nu}^{\rm QE} - 0.5\%\, E_{\nu}^{\rm true}$"),
    ]
    return nominal, osc, shifts


def dune_dm32_relshift():
    nominal, osc, _ = dune_dm32()
    shifts = [
        shift_line(pastel_red,  r"$E_{\nu}^{\rm had} + 0.5\%\, E_{\nu}^{\rm true}$"),
        shift_line(pastel_blue, r"$E_{\nu}^{\rm had} - 0.5\%\, E_{\nu}^{\rm true}$"),
    ]
    return nominal, osc, shifts


def dune_dcp_relshift():
    nominal, osc, _ = dune_dcp()
    shifts = [
        shift_line(pastel_red,  r"$E_{\nu}^{\rm had} + 0.5\%\, E_{\nu}^{\rm true}$"),
        shift_line(pastel_blue, r"$E_{\nu}^{\rm had} - 0.5\%\, E_{\nu}^{\rm true}$"),
    ]
    return nominal, osc, shifts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    save_spectra_legend(*hk_dm32(),   out_dir=OUT_DIR, fname="legend_Fig1_HK_dm32")
    save_spectra_legend(*hk_dcp(),    out_dir=OUT_DIR, fname="legend_Fig1_HK_dCP")
    save_spectra_legend(*dune_dm32(), out_dir=OUT_DIR, fname="legend_Fig1_DUNE_dm32")
    save_spectra_legend(*dune_dcp(),  out_dir=OUT_DIR, fname="legend_Fig1_DUNE_dCP")
    save_spectra_legend(*hk_dm32_relshift(),   out_dir=OUT_DIR, fname="legend_Fig1_HK_dm32_relshift")
    save_spectra_legend(*hk_dcp_relshift(),    out_dir=OUT_DIR, fname="legend_Fig1_HK_dCP_relshift")
    save_spectra_legend(*dune_dm32_relshift(), out_dir=OUT_DIR, fname="legend_Fig1_DUNE_dm32_relshift")
    save_spectra_legend(*dune_dcp_relshift(),  out_dir=OUT_DIR, fname="legend_Fig1_DUNE_dCP_relshift")


if __name__ == "__main__":
    main()
