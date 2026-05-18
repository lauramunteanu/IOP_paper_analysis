#!/usr/bin/env bash
# Build standalone PDFs from the variation_table_{abs,rel}.tex fragments
# written by print_variation_table.py.
#
# Looks for the .tex fragments in:
#   $OUTPUT_PLOTS_DIR/BW_summaries/   (if OUTPUT_PLOTS_DIR is set)
#   /eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw/   (otherwise)
#
# Writes variation_table_{abs,rel}.pdf alongside the fragments. Uses the
# cvmfs TeXLive 2020 distribution to keep things reproducible.
set -euo pipefail

PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2020/bin/x86_64-linux:$PATH
export PATH

if [[ -n "${OUTPUT_PLOTS_DIR:-}" ]]; then
    OUT_DIR="$OUTPUT_PLOTS_DIR/BW_summaries"
else
    OUT_DIR="/eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw"
fi

if [[ ! -d "$OUT_DIR" ]]; then
    echo "ERROR: output dir not found: $OUT_DIR" >&2
    exit 1
fi

build_one() {
    local mode="$1"
    local frag="$OUT_DIR/variation_table_${mode}.tex"
    if [[ ! -f "$frag" ]]; then
        echo "  skipping: $frag does not exist"
        return
    fi
    local stem="variation_table_${mode}_standalone"
    local wrapper="$OUT_DIR/${stem}.tex"
    cat > "$wrapper" <<'EOF'
\documentclass[12pt]{article}
\usepackage[a3paper,landscape,margin=10mm]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[table]{xcolor}
\usepackage{graphicx}
\pagestyle{empty}
\begin{document}
\noindent
EOF
    echo "\\input{variation_table_${mode}.tex}" >> "$wrapper"
    echo "\\end{document}" >> "$wrapper"

    echo "  building $OUT_DIR/variation_table_${mode}.pdf ..."
    ( cd "$OUT_DIR" && pdflatex -interaction=nonstopmode -halt-on-error \
          -jobname="variation_table_${mode}" "${stem}.tex" > /dev/null )
    # Clean up auxiliary files but keep the standalone .tex wrapper for
    # reproducibility.
    rm -f "$OUT_DIR/variation_table_${mode}.aux" \
          "$OUT_DIR/variation_table_${mode}.log"
    echo "  done -> $OUT_DIR/variation_table_${mode}.pdf"
}

echo "compiling variation tables in $OUT_DIR ..."
build_one abs
build_one rel
echo "all done."
