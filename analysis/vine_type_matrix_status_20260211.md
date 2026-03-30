# Vine-Type Matrix + Regression Integrity Status (2026-02-11)

## Submitted jobs

- E3 D-vine: `59848982`
- E3 C-vine: `59848983`
- E4 D-vine: `59848984`
- E4 C-vine: `59848985`
- E5 D-vine: `59848986`
- E5 C-vine: `59848987`
- E5 R-vine: `59848988`

Current queue state: all `PENDING (QOSGrpGRES)` (account/group GPU GRES limit).

## Post-review correction (same day)

During result review we found that E4/E5 SLURM scripts did not `export OUTPUT_BASE`, so
dataset loading fell back to local `repo/data` instead of `${OUTPUT_BASE}/datasets`.
This made the first E4/E5 matrix runs non-comparable to the staged paper baseline.

Fix applied:
- `slurm/paper_e4_imputation.sh`: added `export OUTPUT_BASE`
- `slurm/paper_e5_anomaly.sh`: added `export OUTPUT_BASE`

Paper cache recovery:
- Restored `drafts/paper_outputs/e4_imputation_results.json` and
  `drafts/paper_outputs/e5_anomaly_results.json` from
  `analysis/baseline_artifacts_20260211_021525/`, then regenerated paper artifacts.

Corrected re-submissions:
- E4 D-vine (copy to paper): `59928012`
- E4 C-vine (no copy): `59928013`
- E5 D-vine (copy to paper): `59928014`
- E5 C-vine (no copy): `59928015`
- E5 R-vine (no copy): `59928016`

## Baseline snapshot before reruns

Baseline copies were saved to:
- `analysis/baseline_artifacts_20260211_021525/e3_var_results.json`
- `analysis/baseline_artifacts_20260211_021525/e4_imputation_results.json`
- `analysis/baseline_artifacts_20260211_021525/e5_anomaly_results.json`

Integrity check against current paper cache:
- `drafts/paper_outputs/e3_var_results.json`: identical hash
- `drafts/paper_outputs/e4_imputation_results.json`: identical hash
- `drafts/paper_outputs/e5_anomaly_results.json`: identical hash

So no paper-output regression has occurred yet from the recent code changes.

## Correctness/regression sanity checks

- Unit tests after h-function fix + new tests: `python -m pytest -q` -> `38 passed`.
- Tiny E4 end-to-end sanity run (real staged dataset path):
  - output: `analysis/quick_checks_20260211_021525/e4_dvine_tiny_power.json`
  - pipeline completed and produced finite metrics.
- Tiny E5 local login-node run was killed by host resource limit (exit `137`) before writing JSON; full E5 checks are delegated to submitted GPU SLURM jobs.

## Placeholder/material checks

- All files referenced by `\IfFileExists{...}` in `drafts/vine_diffusion.tex` currently exist (tables/figures resolve).
- Standalone placeholder table files still exist in `drafts/tables/` (e.g., `tab*_placeholder.tex`) but are not currently used by the main paper includes.

## Watch commands

- Queue status:
  - `squeue -j 59848982,59848983,59848984,59848985,59848986,59848987,59848988 -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'`
- Once completed, compare new vs baseline JSONs:
  - `python analysis/summarize_paper_sweeps.py --manifest analysis/paper_sweep_jobs_20260211.json`
  - plus direct artifact diff against `analysis/baseline_artifacts_20260211_021525/`.
