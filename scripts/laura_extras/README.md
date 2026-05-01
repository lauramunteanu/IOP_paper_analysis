# `laura_extras/` — additional bias-summary plots

Extra figures for the FSI IOP paper, sitting alongside the canonical scripts in
`scripts/`. Nothing here modifies Jake's plotting code; the two scripts below are
self-contained and import only matplotlib / numpy / awkward / uproot /
scienceplots.

## Files

- **`fsi_iop_plots.py`** — produces six summary figures comparing per-event
  neutrino-energy-bias distributions across three observables (HK $E_\nu^{QE}$,
  DUNE $E_\nu^{\mathrm{avail}}$, DUNE $E_\nu^{\mathrm{had}}$):

  | Output (× 2: `numu`, `numubar`) | What it shows |
  | --- | --- |
  | `pi_abs_<flavour>.{png,pdf}` | NuWro SF FSI sample, with a $\pm40\%$ reweight of CC inelastic events to mimic the pion-absorption uncertainty. |
  | `fsi_compare_<flavour>.{png,pdf}` | NuWro SF, FSI vs no-FSI samples. |
  | `cascade_compare_<flavour>.{png,pdf}` | GENIE NUISFLAT, four cascade tunes (G18\_10a/b/c/d). |

  Plus four standalone legend files (`legend_quantity.{png,pdf}`,
  `legend_pi_abs.{png,pdf}`, `legend_fsi.{png,pdf}`,
  `legend_cascade.{png,pdf}`).

  Each row of every figure is a translucent weighted-histogram silhouette with
  a 1 σ box outline (16-84 percentile), faint 90 % whiskers (5-95), and
  separate markers for the median (white circle) and the weighted mean
  (white diamond). HK rows use 10 MeV bins over $[-1,+1]$ GeV;
  DUNE rows use 40 MeV bins over $[-3,+1]$ GeV (matching the binning of
  `Fig3_HK_EnuBiasFSI_plot.py` and `Fig4_DUNE_EnuBiasFSI_plot.py` in the parent
  `scripts/` folder). The visible x-axis is always $[-1,+1]$ GeV.

- **`print_stats.py`** — reads the on-disk pickle cache (see below) and prints
  one row per (figure, flavour, observable, variant) with the mean, median,
  $P_{16}$, $P_{84}$, $P_5$, $P_{95}$, 1 σ width and 90 % width, followed by a
  per-group variation summary (range and standard deviation across variants).
  Output is plain text, easy to paste into a paper as a tabular environment.

## How to run

The scripts assume an environment with ROOT-aware `uproot` etc.; on lxplus the
LCG view used during development is

```sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
pip install --user scienceplots   # one-off
```

Inputs live at

```
BASE   = /eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs
OUT_DIR = /eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw
```

Both paths are constants at the top of `fsi_iop_plots.py`; change them to point
at your own working directory before running. Outputs go to `OUT_DIR`.

```sh
python3 fsi_iop_plots.py    # writes 6 figures + 4 legends to OUT_DIR
python3 print_stats.py      # writes the per-row + variation tables to stdout
```

## Pickle cache

To avoid re-reading multi-hundred-MB ROOT files on every iteration, the
loader memoises `(x, w, mode)` arrays into `OUT_DIR/_cache/*.pkl` keyed by
`(reader_kind, basename, MAX_EVENTS)`. The first run primes the cache (a few
minutes); subsequent runs finish in seconds. The cache directory is excluded
from version control (`.gitignore`).

If the input ROOT file content or the reader logic changes, delete
`OUT_DIR/_cache/` before re-running so stale caches don't get reused.

## Differences from the parent `scripts/` figures

The two `Fig3_HK_EnuBiasFSI_plot.py` / `Fig4_DUNE_EnuBiasFSI_plot.py` scripts
draw a single 1-D histogram per `(experiment, observable)` with FSI / no-FSI
overlaid. `Fig_HK_BoxWhisker.py` draws a box-whisker summary across
samples. The figures here merge the two views into a single hybrid layout
that lets you compare three obs by `(experiment, definition)` and several
variants per row in one panel; numerics for the histogram bin widths and the
selection cuts match the parent scripts.
