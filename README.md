# IOP_paper_analysis

Analysis scripts for the FSI IOP paper. Each `scripts/Fig*_plot.py` reads NUISANCE
flat-tree (NUISFLAT) ROOT files and produces one figure (or a small set of
related figures) for the publication. Plotting is done with matplotlib + uproot;
oscillation weights via [OscProb](https://github.com/joaoabcoelho/OscProb).

## 1. Get the input ROOT files

All NUISFLAT inputs (`*_FSI.flat.root`, `*_noFSI.flat.root`, MFP variants,
EDRMF / RPWIA, GENIE cascade tunes, …) are bundled here:

  https://cernbox.cern.ch/s/j10MSE3STy5V8kB

Download the share and unpack so the directory layout matches what the scripts
expect (paths are relative to `scripts/`):

```
<your-workdir>/
├── IOP_paper_analysis/         # this repo
└── Remade_April26/
    ├── HK/
    │   ├── HK_numu_FSI.flat.root
    │   ├── HK_numu_noFSI.flat.root
    │   ├── HK_numubar_FSI.flat.root
    │   ├── HK_numubar_noFSI.flat.root
    │   └── ChangeMFP/                 # 0.7× / 1.3× MFP variants (BoxWhisker)
    └── DUNE/
        ├── DUNE_numu_FSI.flat.root
        ├── DUNE_numu_noFSI.flat.root
        ├── DUNE_numub_FSI.flat.root
        ├── DUNE_numub_noFSI.flat.root
        └── plus30_FSI/                # MFP variants (BoxWhisker)
```

Fig7 (EDRMF vs RPWIA) additionally needs the NEUT samples in `../../FSI/`.

## 2. Set up the environment (lxplus)

```sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
export OSCPROB_LIB=/eos/project-n/neutrino-generators/software/OscProb/lib/libOscProb.so
```

LCG\_107 ships ROOT 6.34.02 (matches the OscProb build above), uproot, awkward,
matplotlib, and a working LaTeX. If `OSCPROB_LIB` is unset, FlatTreeMod also
checks `$OSCPROB/lib/libOscProb.so` and a hardcoded fallback path; without any
of them the Fig1 / Fig1\_DUNE scripts (oscillation weights) will fail at
runtime — the others still work.

For the `scienceplots` style:

```sh
pip install --user scienceplots    # one-off
```

LaTeX must be on `$PATH`; the LCG view provides it.

## 3. Repository layout

```
scripts/
├── FlatTreeMod.py                 # shared helpers — see §5
├── Fig1_dCP_plot.py               # HK Eν spectra, ±20° dCP shift
├── Fig1_dm32_plot.py              # HK Eν spectra, ±0.4% Δm²₃₂ shift
├── Fig1_dCP_DUNE_plot.py          # DUNE Eν spectra, dCP + Δm²₃₂ shifts
├── Fig2_HK_plot.py                # HK Eν^QE − Eν^true, by mode, noFSI
├── Fig2_DUNE_plot.py              # DUNE Eν^had − Eν^true, by neutron content, noFSI
├── Fig2_HK_CC0pi_flag_plot.py    # CC0π flag-based vs computed-mask sanity check
├── Fig3_HK_EnuBiasFSI_plot.py    # HK ν̄μ FSI vs noFSI, with ratio panel
├── Fig4_DUNE_EnuBiasFSI_plot.py  # DUNE FSI vs noFSI, ν/ν̄ × with/without π mass
├── Fig5_EnergyFromNeutrons_plot.py # Σ Tn / q0, stacked by neutron multiplicity
├── Fig6_DUNE_EnuBias_2D_plot.py  # 2D heatmap: bias vs Eν^true (DUNE)
├── Fig6_HK_EnuBias_2D_plot.py    # 2D heatmap: bias vs Eν^true (HK)
├── Fig7_EDRMF_DUNE_plot.py       # ED-RMF vs RPWIA, DUNE
├── Fig7_EDRMF_HK_plot.py         # ED-RMF vs RPWIA, HK
├── Fig_HK_BoxWhisker.py          # box-and-whisker summary across samples
├── Fig{1..7}_plots/               # output PDFs land here
├── BW_plots/                      # BoxWhisker output
└── laura_extras/                  # additional bias-summary plots (own README)
```

## 4. Run a figure

Each script is a self-contained module. From `scripts/`:

```sh
cd scripts/
python3 Fig3_HK_EnuBiasFSI_plot.py   # writes Fig3_plots/Enu_bias_FSIvsNoFSI_numubar.pdf
python3 Fig4_DUNE_EnuBiasFSI_plot.py # writes 4 PDFs to Fig4_plots/
# ... etc.
```

There are no command-line flags — input file paths and event counts are
constants at the bottom of each script (look for `_events = ...`). For an
end-to-end regen of the whole figure set, just iterate over the scripts:

```sh
for s in Fig*.py; do python3 "$s"; done
```

The first invocation per ROOT file is slow (uproot decode); subsequent runs
hit the pickle cache (§6) and finish in seconds.

## 5. `FlatTreeMod.py` — shared helpers

Every Fig script does `from FlatTreeMod import *`. The module provides:

- **Style** — `plt.rcParams` set for IOP single-column print width
  (10 pt labels, 9 pt ticks/legend, Computer Modern serif). Do not override
  font sizes per-figure; rely on the global rcParams + the `make_fig*` helpers.
- **Figure sizes / helpers** — `make_fig(kind)`, `make_fig_ratio(kind)`,
  `make_fig_stacked(kind)` with `kind` ∈
  `{single, single_ratio, double, double_ratio, double_stacked, box}`.
  See `FIG_SIZES` for the (w, h) inches map.
- **Colours** — `dark_red` / `dark_blue` / `light_*` / `vivid_purple` (Paul Tol
  vibrant), `TOL_MUTED` for stacked categorical breakdowns.
- **Loading** — `load_arrays(filename)` returns a dict of awkward/numpy arrays
  for the canonical NUISFLAT branches and caches the result on disk (§6).
  `is_cc0pi_arr(arr)`, `enu_had_arr(arr)`, `diff_enu_qe_arr(arr)`, etc., are
  the vectorised event-selection / observable helpers.
- **Oscillation** — global `pmns` (`OscProb.PMNS_Fast`) preset to PDG-ish
  oscillation parameters and the T2K (HK) baseline. Re-set `pmns.SetPath`,
  `pmns.SetMix`, `pmns.SetDeltaMsqrs` per-figure as needed (Fig1\_DUNE
  re-paths to 1285 km).
- **Weights** — `make_weights_dxsec(arr, bin_width)` for differential xsec
  plots; `make_weights_event_rate(arr, filename, bin_width)` for Eν spectra
  scaled to `EXPECTED_EVENTS`.

## 6. Pickle cache

`load_arrays()` writes one pickle per (file mtime, size, branch list,
`max_events`) combination to:

```
$IOP_PAPER_CACHE             # if set
/eos/home-l/lamuntea/.cache/iop_paper/   # default (override with $IOP_PAPER_CACHE)
```

Subsequent calls with the same key load from disk instead of re-decoding the
ROOT file. **Wipe the cache** if you change the *derivation logic* in
`FlatTreeMod.py` (selection cuts, observable definitions, branch list).
Pure styling changes (fonts, figsize, colours) do not invalidate the cache.

```sh
rm -rf /eos/home-l/lamuntea/.cache/iop_paper/   # or wherever $IOP_PAPER_CACHE points
```

## 7. Common gotchas

- **OscProb version mismatch** — OscProb is built against a specific ROOT
  version. The library at `/eos/project-n/neutrino-generators/software/OscProb`
  is built with ROOT 6.34.x → use LCG\_107 (or 107a). LCG\_106 ships ROOT
  6.32 and will warn `version mismatch, libOscProb.so = 63402, ROOT = 63202`.
- **Missing samples** — most scripts `noFSI_path()`-derive their noFSI input
  from the FSI path. Make sure the cernbox tarball was unpacked with both
  `*_FSI.flat.root` and `*_noFSI.flat.root` siblings.
- **LaTeX not found** — figures use `text.usetex=True`. Re-source the LCG
  view (it brings TeXLive on `$PATH`). Without LaTeX you'll see
  `RuntimeError: latex was not able to process the following string`.
- **Stats\_table.txt** — `Fig_HK_BoxWhisker.py` opens this file in append
  mode. Delete it before re-running if you want a clean table.

For the bias-summary plots and a description of their layout / cache, see
`scripts/laura_extras/README.md`.
