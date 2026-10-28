#!/usr/bin/env bash
# Regenerate the ten paper figures (PDF and PNG) from the saved inputs in ./inputs.
# Usage: bash make_figures.sh [OUTPUT_DIR]        (default: ./output; set PYTHON to choose the interpreter)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/output}"
PY="${PYTHON:-python}"
export PYTHONUTF8=1 MPLBACKEND=Agg
mkdir -p "$OUT"
for s in fig1_overview_cr fig2_density_runtime_cr fig3_information_cr fig4_tc_decomposition_cr figA_training_diagnostics_cr \
         figA_nonsimplified_stress_cr figA_tc_story_cr figA_mi_sample_size_cr figA_var_calibration_cr \
         figA_rolling_dependence_cr; do
  echo "== $s"
  "$PY" "$HERE/scripts/$s.py" --inputs "$HERE/inputs" --figures-dir "$OUT"
done
