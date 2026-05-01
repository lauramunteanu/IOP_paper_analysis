#!/usr/bin/env bash
# Generate every paper figure in scripts/. Each Fig*.py is invoked through
# run_one_fig.py so that figures are saved to disk regardless of whether the
# script has plt.savefig() in it.
#
# Output:  scripts/output/<Fig*>_<i>.png  (and a one-line status per script)
#
# Usage:   bash scripts/run_all_figs.sh


REPO=/eos/home-l/lamuntea/FSI_IOP_paper/IOP_paper_analysis
SCRIPTS_DIR="$REPO/scripts"
OUT_DIR="$SCRIPTS_DIR/output"
TIMEOUT=600   # seconds per script

mkdir -p "$OUT_DIR"

# One-time data path symlink so Jake's `../../Remade_April26/...` resolves
# to the EOS-hosted samples in /eos/project-n/...
SYMLINK="$REPO/../Remade_April26"
DATA_TARGET="/eos/project-n/neutrino-generators/generatorOutput/FSIIOPPaperinputs"
if [ ! -e "$SYMLINK" ]; then
  ln -sfn "$DATA_TARGET" "$SYMLINK"
  echo "[run_all_figs] created symlink $SYMLINK -> $DATA_TARGET"
fi

# Source the project env (gives us LCG_107 ROOT 6.34, NUISANCE, OscProb, ...)
source /eos/project-n/neutrino-generators/software/setup_all_lxplus.sh >/dev/null 2>&1
export MPLBACKEND=Agg

cd "$SCRIPTS_DIR"
echo
echo "=== running every Fig*.py, output -> $OUT_DIR ==="
printf "%-40s  %-8s  %-7s  %s\n" "script" "status" "ms" "n_pngs"
echo "-------------------------------------------------------------------"
overall=0
for fig in $(ls -1 Fig*.py 2>/dev/null | sort); do
  basename="${fig%.py}"
  log="$OUT_DIR/${basename}.log"
  t0=$(date +%s%3N)
  timeout "$TIMEOUT" python3 "$SCRIPTS_DIR/run_one_fig.py" "$fig" "$OUT_DIR" "$basename" \
    > "$log" 2>&1
  rc=$?
  t1=$(date +%s%3N)
  dt=$((t1 - t0))
  n_png=$(ls -1 "$OUT_DIR/${basename}_"*.png 2>/dev/null | wc -l)
  if [ $rc -eq 0 ] && [ "$n_png" -gt 0 ]; then
    status="OK"
  elif [ $rc -eq 124 ]; then
    status="TIMEOUT"; overall=1
  elif [ $rc -ne 0 ]; then
    status="FAIL"; overall=1
  else
    status="EMPTY"; overall=1
  fi
  printf "%-40s  %-8s  %5d ms  %d\n" "$fig" "$status" "$dt" "$n_png"
done
echo
if [ $overall -eq 0 ]; then
  echo "all scripts produced figures"
else
  echo "some scripts had problems; see per-script logs in $OUT_DIR/"
fi
