#!/usr/bin/env bash
# Rebuild every released table from the saved outputs in inputs/ and, when a reference directory is
# given (for example the paper's tables/ directory), compare the results with it, ignoring full-line
# LaTeX comments. Requires Python 3.9+ with numpy and scipy; no estimator is run.
#
# usage: bash make_tables.sh OUT_DIR [REFERENCE_DIR]
#        PYTHON=/path/to/python bash make_tables.sh OUT_DIR [REFERENCE_DIR]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:?usage: make_tables.sh OUT_DIR [REFERENCE_DIR]}"
py="${PYTHON:-python3}"
for s in table1 uci ood mi scaling sampling self_consistency ipfp_iterations raw_composition threads; do
    "$py" "$here/scripts/$s.py" --inputs "$here/inputs" --out "$out"
done
if [[ $# -ge 2 ]]; then
    "$py" "$here/scripts/compare_tables.py" "$out" "$2"
fi
