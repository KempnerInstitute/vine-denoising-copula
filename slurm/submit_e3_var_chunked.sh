#!/bin/bash
# Submit chunked E3 VaR backtests for exact rolling protocol in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula}"
PARTITION="${PARTITION:-kempner_eng}"
SEEDS="${SEEDS:-42 123 456}"
CHUNK_DAYS="${CHUNK_DAYS:-360}"   # should be a multiple of REFIT_EVERY
REFIT_EVERY="${REFIT_EVERY:-5}"
WINDOW="${WINDOW:-252}"
N_SIM="${N_SIM:-5000}"
TIME_LIMIT="${E3_SBATCH_TIME:-48:00:00}"

if [ $((CHUNK_DAYS % REFIT_EVERY)) -ne 0 ]; then
  echo "ERROR: CHUNK_DAYS=${CHUNK_DAYS} must be a multiple of REFIT_EVERY=${REFIT_EVERY}"
  exit 1
fi

CHECKPOINT="${PAPER_CHECKPOINT:-}"
if [ -z "${CHECKPOINT}" ]; then
  CHECKPOINT="$(python - <<'PY'
from pathlib import Path
from vdc.utils.paper import choose_best_checkpoint
base = Path("/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula")
ck = choose_best_checkpoint(
    output_bases=[base],
    preferred_methods=["denoiser_cond_enhanced", "denoiser_cond"],
    metric="mean_ise",
    prefer_joint=True,
)
print("" if ck is None else str(ck))
PY
)"
fi

if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: unable to resolve PAPER_CHECKPOINT."
  exit 1
fi

N_DAYS="$(python - <<'PY'
from vdc.data.paths import data_root
from vdc.data.tabular import maybe_load_finance_sp100_returns
R = maybe_load_finance_sp100_returns(data_root())
window = 252
print(int(R.shape[0] - window - 1))
PY
)"

echo "============================================================================"
echo "Submitting chunked E3 VaR jobs"
echo "Date: $(date)"
echo "Partition: ${PARTITION}"
echo "Output base: ${OUTPUT_BASE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Seeds: ${SEEDS}"
echo "Window: ${WINDOW}"
echo "Refit every: ${REFIT_EVERY}"
echo "N sim: ${N_SIM}"
echo "N days: ${N_DAYS}"
echo "Chunk days: ${CHUNK_DAYS}"
echo "Walltime: ${TIME_LIMIT}"
echo "============================================================================"

declare -a JOB_IDS=()

submit() {
  local label="$1"
  shift
  local jid
  jid="$(sbatch --parsable --partition="${PARTITION}" --time="${TIME_LIMIT}" "$@")"
  JOB_IDS+=("${jid}")
  printf "  %-34s %s\n" "${label}" "${jid}"
}

for seed in ${SEEDS}; do
  start=0
  while [ "${start}" -lt "${N_DAYS}" ]; do
    submit "E3 seed=${seed} start=${start}" \
      --export=ALL,OUTPUT_BASE="${OUTPUT_BASE}",PAPER_CHECKPOINT="${CHECKPOINT}",E3_SEED="${seed}",E3_WINDOW="${WINDOW}",E3_REFIT_EVERY="${REFIT_EVERY}",E3_N_SIM="${N_SIM}",E3_START_DAY="${start}",E3_MAX_DAYS="${CHUNK_DAYS}",E3_COPY_TO_PAPER=0,E3_REGENERATE_ARTIFACTS=0 \
      slurm/paper_e3_var.sh
    start=$((start + CHUNK_DAYS))
  done
done

echo ""
echo "Submitted job IDs:"
printf "  %s\n" "${JOB_IDS[@]}"
echo ""
echo "Monitor:"
echo "  squeue -u ${USER}"
echo "  sacct -j $(IFS=,; echo "${JOB_IDS[*]}") --format=JobID,JobName%30,Partition,State,Elapsed,ExitCode -P"
echo ""
echo "After completion, merge chunk JSON files:"
echo "  python drafts/scripts/e3_var_merge_chunks.py --input-glob '/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_e3_var_*/results/e3_var_results.json' --out-json drafts/paper_outputs/e3_var_results.json"

