# Paper Results Status (2026-02-08)

## Update (2026-02-13 09:30 EST)
- Added TC story composite figure (main text):
  - Script: `drafts/scripts/fig_tc_story_composite.py`
  - Output: `drafts/figures/fig_tc_story_composite.pdf`
  - Panels:
    1. Ground-truth edge MI recovery (synthetic Gaussian factor truth vs estimate),
    2. Real-data tree decomposition contrast (Power vs Gas, grouped bars),
    3. Existing TC scaling benchmark panel from `results/tc_benchmark.json`.
- Negative TC handling policy for decomposition comparison:
  - Keep signed TC values as-is (do **not** clamp negative totals to zero),
  - Use per-tree **absolute share of |TC|** for grouped bar comparisons to avoid misleading cancellation artifacts in near-independence regimes.
- Main manuscript now uses this composite in the Information section:
  - `drafts/vine_diffusion.tex` Figure `\ref{fig:tc_decomposition}` now includes `fig_tc_story_composite`.
- Artifact pipeline wiring:
  - `drafts/scripts/paper_artifacts.py` now calls `fig_tc_story_composite.py`.
- Validation:
  - `pdflatex` 2-pass compile succeeds,
  - `analysis/paper_asset_audit_latest.json` remains clean (`ok=true`).

## Update (2026-02-13 03:25 EST)
- Added real-data TC decomposition pipeline:
  - Script upgraded: `drafts/scripts/generate_tc_decomposition_figure.py`
    - supports `--mode uci` using canonical checkpoint + real UCI data,
    - writes figure, JSON artifact, and LaTeX table from the same run.
- Real runs completed with canonical checkpoint
  `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/checkpoints/model_step_190000.pt`:
  - `power` (used in main text): `drafts/paper_outputs/tc_decomposition_uci_power.json`
    - TC = `2.3566` nats, `T1=89.0%`, `T1+T2=97.1%`.
  - `gas` diagnostic: `drafts/paper_outputs/tc_decomposition_uci_gas.json`
    - near-independence regime (`TC≈-0.0010` nats with sign cancellations), not suitable as flagship interpretability panel.
  - `hepmass` diagnostic (subsampled): `drafts/paper_outputs/tc_decomposition_uci_hepmass.json`
    - similarly near-zero signed TC under this estimator setting.
- Figure/table artifacts from real data:
  - Main figure now overwritten with real-data run: `drafts/figures/fig_tc_decomposition.pdf`
  - Additional table: `drafts/tables/tab_tc_decomposition_uci_power.tex`
- Manuscript integration:
  - Added TC decomposition figure in Information section:
    `drafts/vine_diffusion.tex` (Figure `\ref{fig:tc_decomposition}`), with real-data caption values.
- Added reproducible asset audit utility:
  - `analysis/audit_paper_assets.py` -> `analysis/paper_asset_audit_latest.json`
  - Current audit status: all referenced figures/tables exist and no referenced table is placeholder-flagged.

## Update (2026-02-13 04:05 EST)
- Added explicit ground-truth TC decomposition comparison benchmark:
  - Script: `drafts/scripts/compare_tc_decomposition_ground_truth.py`
  - Setup: synthetic Gaussian factor copula (`d=8`), analytic TC and analytic per-edge MI under D-vine natural ordering.
  - Fits \ours{} on sampled train data with canonical checkpoint and compares edge/tree/total TC on held-out data.
- Generated artifacts:
  - `drafts/figures/fig_tc_decomposition_ground_truth.pdf`
  - `drafts/paper_outputs/tc_decomposition_ground_truth_compare.json`
  - `drafts/tables/tab_tc_decomposition_ground_truth.tex`
- Current results (real run, no placeholders):
  - `TC_true = 1.7505`, `TC_est = 1.7916`, `|Delta TC| = 0.0411`
  - Edge-wise error: `MAE = 0.0041`, `RMSE = 0.0068` over `28` edges.
- Manuscript integration:
  - Added appendix subsection “Ground-Truth TC Decomposition Check” with figure/table includes in `drafts/vine_diffusion.tex`.

## Update (2026-02-13 02:35 EST)
- Added explicit in-repo canonical model pointers for camera-ready reproducibility:
  - `analysis/PAPER_CHECKPOINT.txt`
  - `analysis/PAPER_BEST_MODEL.json`
- Canonical checkpoint remains:
  - `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/checkpoints/model_step_190000.pt`
- Added dedicated documentation page:
  - `docs/PAPER_REPRODUCIBILITY.md` (resolution order + exact rerun commands for information benchmarks).
- Checkpoint resolver hardening:
  - `vdc/utils/paper.py` now resolves canonical checkpoint in this order:
    `PAPER_CHECKPOINT` env -> `analysis/PAPER_CHECKPOINT.txt` -> `analysis/PAPER_BEST_MODEL.json` -> run auto-discovery.
  - This prevents accidental drift to a different checkpoint when `--checkpoint` is omitted.

## Update (2026-02-11 02:12 EST)
- Execution pivot applied (results-first, no paper overwrite during sweeps):
  - Cancelled infeasible full-horizon daily-resimulation E3 pilot: `59792536` (`CANCELLED`, had reached `12/144` refits in ~`6.85h`).
  - Kept baseline E3 pilot running: `59792535` (`resimulate-daily=0`, currently near completion).
  - Submitted tractable daily-resimulation ablation:
    - `59843463` (`E3_RESIMULATE_DAILY=1`, `E3_MAX_DAYS=120`, `COPY_TO_PAPER=0`, `REGENERATE_ARTIFACTS=0`).
- Submitted focused multi-seed E4/E5 sweeps on `kempner_eng` using fixed checkpoint
  `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/checkpoints/model_step_190000.pt`:
  - E4 (`COPY_TO_PAPER=0`, `REGENERATE_ARTIFACTS=0`):
    - `59843465`–`59843467`: `kernel_h=0.05`, `candidate_pool=50000`, seeds `42/123/456`
    - `59843468`–`59843470`: `kernel_h=0.02`, `candidate_pool=100000`, seeds `42/123/456`
  - E5 (`COPY_TO_PAPER=0`, `REGENERATE_ARTIFACTS=0`):
    - `59843471`–`59843473`: `score_mode=best`, seeds `42/123/456`
    - `59843474`–`59843476`: `score_mode=neg_logpdf`, seeds `42/123/456`
- Added reproducible sweep tracking + summarization:
  - Manifest: `analysis/paper_sweep_jobs_20260211.json`
  - Summary script: `analysis/summarize_paper_sweeps.py`
  - Auto-output files:
    - `analysis/paper_sweep_results_latest.json`
    - `analysis/paper_sweep_results_latest.md`
  - Current summary snapshot: `12/13` sweep jobs finished with artifacts (only E3 `59843463` still running).
- Early sweep outcomes (seed-robust):
  - E4:
    - `h005_pool50k` (jobs `59843465`–`59843467`): mean delta vs best baseline `+23.93%` (still weak).
    - `h002_pool100k` (jobs `59843468`–`59843470`): mean delta `+32.14%` (worse than `h005_pool50k`).
    - Decision: keep `h005_pool50k` as less-bad E4 config; do not promote `h002_pool100k`.
  - E5:
    - `score_best` (jobs `59843471`–`59843473`): \ours{} mean AUROC `0.582`, gap vs best baseline `-0.260`.
    - `score_neglogpdf` (jobs `59843474`–`59843476`): \ours{} mean AUROC `0.457`, gap `-0.385`.
    - Decision: `score_best` remains the correct E5 default; performance gap to strongest baselines persists.
- Additional E4 robustness check launched (previous single-seed best setting):
  - `59844095`–`59844097`: `kernel_h=0.1`, `candidate_pool=50000`, seeds `42/123/456` (no paper overwrite).
  - Completed outcomes:
    - Mean delta vs best baseline: `+16.57%` (seed-level: `+15.09%`, `+17.16%`, `+17.46%`).
    - This is the least-weak E4 setting among tested sweeps, but still clearly behind baseline imputers.

## Update (2026-02-10 17:31 EST)
- Focused weak-spot reruns completed (E4/E5):
  - E4 sweep jobs (`59792454`–`59792458`) run without paper overwrite to test kernel bandwidth / candidate pool.
    - Best among tested configs: `kernel_h=0.1`, `candidate_pool=50000` (`59792457`), but \ours{} remains below simple imputers overall.
  - E5 sweep jobs (`59792459`–`59792461`) confirm score-orientation handling matters:
    - `score_mode=best` (`59792459`, `59792460`) gives \ours{} mean AUROC ≈ `0.582` (vs ~`0.457` with fixed `neg_logpdf`).
    - `score_mode=logpdf` (`59792461`) is weaker than `best` (mean AUROC ≈ `0.543`).
- Paper cache promotion jobs on H100 (`59794558`, `59794559`) completed:
  - Promoted E4 config: `kernel_h=0.1`, `candidate_pool=50000`, `n_eval=500`.
  - Promoted E5 config: `score_mode=best`.
- Current paper-facing tables after regeneration:
  - `drafts/tables/tab_imputation.tex`:
    - Power: `7.191` vs best baseline `5.743` (kNN), `-25.2%`.
    - Gas: `1.005` vs best baseline `0.993` (Mean), `-1.2%`.
    - Credit: `1.267` vs best baseline `1.013` (Mean), `-25.1%`.
  - `drafts/tables/tab_anomaly.tex`:
    - \ours{} now `AUROC=0.581`, `AP=0.269` (still below strongest classical methods).
- Artifacts and manuscript sync:
  - `python drafts/scripts/paper_artifacts.py all --force` completed.
  - `pdflatex` two-pass compile for `drafts/vine_diffusion.tex` completed (warnings only; no fatal errors).
  - Added two real appendix sensitivity tables (replacing placeholder intent with measured values):
    - `drafts/tables/tab_e4_sweep.tex`
    - `drafts/tables/tab_e5_scoremode_sweep.tex`
- E3 pilot jobs currently running for calibration decision:
  - `59792535`: baseline (`resimulate-daily=0`) on first 720 days.
  - `59792536`: daily-resimulation variant (`resimulate-daily=1`) on same slice.
  - These are diagnostic runs only (no paper cache overwrite) to decide whether full-horizon E3 rerun is warranted.

## Update (2026-02-10 16:58 EST)
- Paper benchmark queue check:
  - `COMPLETED`: `59756060` (MI), `59756061` (MI consistency), `59756062` (TC), `59756063` (E2 UCI), `59756075` (E4 imputation), `59756076` (E5 anomaly), `59756126` (E6 theory), `59756128` (E7 biomed).
  - Chunked E3 VaR queue all `COMPLETED`: `59756191`, `59756203`–`59756216`.
- E3 full-horizon merge finalized:
  - Merged 15 chunk outputs into `drafts/paper_outputs/e3_var_results.json` via `drafts/scripts/e3_var_merge_chunks.py`.
  - Aggregated coverage: `n=4326` evaluation days per method/alpha, `n_refits=867`, mean refit fit-time `4.8295s`.
  - Aggregated VaR summary (Kupiec p-values):
    - 1%: VDC `1.39%` (p=`0.0157`), Historical `1.46%` (p=`0.00474`), Gaussian `1.94%` (p=`3.53e-08`).
    - 5%: VDC `1.87%` (p=`0.0`), Historical `4.37%` (p=`0.0518`), Gaussian `4.72%` (p=`0.3865`).
- Paper artifacts synchronized after merge:
  - Regenerated tables/figures from cache (`drafts/scripts/paper_artifacts.py all --force`).
  - `drafts/tables/tab_var.tex` now correctly includes all 6 method/alpha rows from merged E3 results.
  - Updated figures include refreshed `drafts/figures/fig_var_calibration.pdf`.
- Manuscript build check:
  - `pdflatex` on `drafts/vine_diffusion.tex` passes (2-pass compile).
  - Warnings are non-fatal underfull boxes / float placement (`h -> ht`), no missing-file or compile errors.
- Note:
  - There is an unrelated SLURM array (`59598689_*`) still running on `kempner_eng`; it is not part of this paper benchmark bundle.

## Update (2026-02-10 12:35 EST)
- New checkpoint-pinned quick jobs completed (`model_step_190000.pt`):
  - `59756060` (MI benchmark) `COMPLETED`
  - `59756061` (MI consistency) `COMPLETED`
  - `59756075` (E4 imputation) `COMPLETED`
  - `59756076` (E5 anomaly) `COMPLETED`
- Key refreshed metrics:
  - MI benchmark (`results/mi_benchmark_summary.json`):
    - DCD MAE `0.0100` vs KSG `0.0205`, Gaussian `0.0640`, InfoNCE `0.0235`, MINE `0.0214`, NWJ `0.0238`.
  - MI consistency (`results/mi_self_consistency.json`):
    - DCD-Vine retains `0.0%` DPI violations and lower additivity error than KSG in current protocol.
  - E4 table (`drafts/tables/tab_imputation.tex`) refreshed from new artifact; \ours{} remains behind simple imputers.
  - E5 table (`drafts/tables/tab_anomaly.tex`) refreshed from new artifact; \ours{} remains below classical anomaly baselines.
- Currently running:
  - `59756062` (TC benchmark), `59756063` (E2 UCI),
  - `59756126` (E6 theory, `n_trials=5`),
  - `59756128` (E7 biomed, expanded setting),
  - plus chunked E3 queue (`59756191` + `59756203`–`59756216`).

## Update (2026-02-10 12:32 EST)
- Enforced **joint-best checkpoint selection** across paper benchmark pipeline:
  - `vdc/utils/paper.py`: `choose_best_checkpoint(..., prefer_joint=True)` now prefers `results/checkpoint_path_joint.txt`.
  - Updated callers in:
    - `drafts/scripts/e2_uci_benchmark.py`
    - `drafts/scripts/e3_var_backtest.py`
    - `drafts/scripts/e4_imputation_benchmark.py`
    - `drafts/scripts/e5_anomaly_benchmark.py`
    - `drafts/scripts/e6_theory_synthetic_benchmark.py`
    - `drafts/scripts/e7_biomed_benchmark.py`
    - `scripts/mi_estimation.py`
    - `slurm/paper_mi_consistency.sh`
    - `slurm/paper_tc_benchmark.sh`
    - `slurm/submit_all_paper_experiments.sh`
    - `slurm/submit_paper_expanded_suite.sh`
- Root-cause for weak/strong inconsistency confirmed: previous fresh E6/E7/E3 submissions were using `model_step_70000.pt` (standard-only selector), not joint-best `model_step_190000.pt`.
- Cancelled outdated runs:
  - `59741609` (E7 biomed with 70000)
  - `59729310`, `59729311`, `59729312` (long E3 runs with 70000)
- Submitted fresh checkpoint-pinned jobs (`model_step_190000.pt`) on `kempner_eng`:
  - `59756060` MI benchmark
  - `59756061` MI consistency
  - `59756062` TC benchmark
  - `59756063` E2 UCI
  - `59756075` E4 imputation
  - `59756076` E5 anomaly
  - `59756126` E6 theory (`n_trials=5`)
  - `59756128` E7 biomed (expanded datasets requested)
- Added chunked-parallel E3 infrastructure (exact same rolling protocol, no approximation):
  - `slurm/paper_e3_var.sh` now supports `E3_START_DAY`, `E3_COPY_TO_PAPER`, `E3_REGENERATE_ARTIFACTS`.
  - New submitter: `slurm/submit_e3_var_chunked.sh`
  - New merger: `drafts/scripts/e3_var_merge_chunks.py`
- Submitted 15 chunked E3 jobs (3 seeds × 5 chunks):
  - `59756191`, `59756203`, `59756204`, `59756205`, `59756206`,
    `59756207`, `59756208`, `59756209`, `59756210`, `59756211`,
    `59756212`, `59756213`, `59756214`, `59756215`, `59756216`
- Paper readability/compilation:
  - Introduction overview figure now has robust fallback to `figures/fig_method_pipeline_mi.pdf` when `fig1_pipeline_improved.pdf` is unavailable in the active compile path.

## Update (2026-02-10 11:36 EST)
- Comparison-strengthening change requested: use Table-3 information estimators beyond vine-only baselines.
- Implemented in benchmark code:
  - `drafts/scripts/e6_theory_synthetic_benchmark.py` now runs Table-3 MI methods on sampled bivariate pairs from E6 scenarios:
    - default methods: `dcd`, `ksg`, `gaussian`, `infonce`, `mine`, `nwj`
    - optional methods supported: `minde`, `mist`
    - writes `mi_records`, `mi_summary_records`, `mi_method_summary` into E6 JSON.
  - `drafts/scripts/e7_biomed_benchmark.py` now runs the same MI method family on selected real-data feature pairs (top-correlation pairs) and stores:
    - `mi_pair_records`, `mi_summary`, `mi_overall` (including agreement-vs-KSG fields).
- Artifact/paper integration:
  - New generated tables:
    - `drafts/tables/tab_theory_mi_methods.tex`
    - `drafts/tables/tab_biomed_mi_methods.tex`
  - Included in manuscript:
    - `drafts/vine_diffusion.tex` (main E6 MI comparison + appendix biomed MI comparison).
- Operational correction:
  - Previous E6/E7 jobs (`59739046`, `59739045`) were started before this code update and therefore did **not** include Table-3 MI suite.
  - Cancelled old jobs and resubmitted updated runs on `kempner_eng`:
    - `59741608` = `vdc_paper_theory` (updated E6 + MI suite)
    - `59741609` = `vdc_paper_biomed` (updated E7 + MI suite)
  - Both currently `RUNNING`.
- Interim cache caveat remains:
  - `drafts/paper_outputs/e6_theory_synthetic_results.json` and `drafts/paper_outputs/e7_biomed_results.json` are still smoke-validation artifacts until `59741608`/`59741609` finish.

## Update (2026-02-10 11:15 EST)
- Added new paper benchmark tracks requested for stronger positioning:
  - **E6 theory synthetic suite** (known-structure scenarios; includes TC-truth checks when available).
  - **E7 biomedical suite** (biology-facing anomaly + imputation benchmarks).
- New SLURM jobs submitted on `kempner_eng` with extended walltime (`48:00:00`):
  - `59739046` = `vdc_paper_theory` (E6)
  - `59739045` = `vdc_paper_biomed` (E7)
- Initial log checks:
  - both jobs started successfully;
  - expected conda env resolved to `/n/netscratch/kempner_dev/hsafaai/conda_envs/diffuse_vine_cop`;
  - expected staged output base resolved to `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula`.
- Paper integration updates:
  - Added main-text theory benchmark table include: `tables/tab_theory_synth.tex`.
  - Added appendix biomedical benchmark table includes:
    - `tables/tab_biomed_anomaly.tex`
    - `tables/tab_biomed_imputation.tex`
  - Removed duplicate abstract paragraph in `drafts/vine_diffusion.tex` to improve readability.
- Validation:
  - `drafts/scripts/paper_artifacts.py tables` now generates all three new tables.
  - `pdflatex` compile passes for `drafts/vine_diffusion.tex` (warnings only; no missing-file fallbacks for referenced tables/figures).
- Interim artifact caveat:
  - current `drafts/paper_outputs/e6_theory_synthetic_results.json` and `drafts/paper_outputs/e7_biomed_results.json` are smoke-run placeholders for pipeline validation and should be replaced by outputs from `59739046`/`59739045` before camera-ready claims.

## Update (2026-02-10 10:27 EST)
- E3 VaR reruns with `24h` walltime (`59591824`, `59591825`, `59591826`) all ended `TIMEOUT` and produced no final JSON.
- Root-cause clarification:
  - Earlier "fast/strong" VaR jobs (`59470884`, `59470885`) were submitted without exported `OUTPUT_BASE`, so Python defaulted to local debug dataset (`data/finance/sp100_returns.npy`, shape `(500, 10)`).
  - Expanded-suite jobs exported `OUTPUT_BASE`, so Python used staged paper dataset (`/n/holylfs06/.../datasets/finance/sp100_returns.npy`, shape `(1695, 100)`), which is substantially heavier.
- Mitigation applied:
  - Increased E3 default walltime to `48:00:00` in `slurm/paper_e3_var.sh`.
  - Set expanded-suite default `E3_SBATCH_TIME=48:00:00` in `slurm/submit_paper_expanded_suite.sh`.
  - Exported `OUTPUT_BASE` and enabled unbuffered Python output in `slurm/paper_e3_var.sh` for deterministic dataset resolution and visible logs.
  - Added progress logging/checkpointing to `drafts/scripts/e3_var_backtest.py` (`--progress-every`, per-refit progress file `<out>_progress.json`).
- New E3 submissions (48h, full dataset, seeds 42/123/456):
  - `59729310`, `59729311`, `59729312`.
  - At `2026-02-10 10:28 EST`: jobs started and logs confirm full data (`shape=(1695, 100)`) plus live progress output.
  - First observed refit times: `~6.18s` (seed 42) and `~7.29s` (seed 123); progress snapshots written to `results/e3_var_results_progress.json` under each run dir.

## Update (2026-02-08 21:52 EST)
- LaTeX readability pass completed:
  - Added width-constrained table inclusion macro (`\fitcoltable`) and applied it to all artifact-backed tables in `drafts/vine_diffusion.tex`.
  - Resolved prior table overflow issues (no `Overfull \hbox` table warnings in latest compile).
- Benchmark expansion code updates:
  - E2: added Gaussian-copula baseline (`gaussian_copula`).
  - E3: added VaR baselines (Historical and Gaussian) in the same rolling protocol.
  - E4: added IterativeImputer baseline (`rmse_iterative`).
  - E5: added EllipticEnvelope baseline; score mode now configurable (`neg_logpdf`/`logpdf`/`best`).
- New artifact-generated paper figures:
  - `drafts/figures/fig_var_calibration.pdf`
  - `drafts/figures/fig_imputation_methods.pdf`
  - `drafts/figures/fig_anomaly_methods.pdf`
- Expanded multi-seed suite submitted on `kempner_eng`:
  - Core info jobs: `59504504` (MI), `59504515` (MI consistency), `59504526` (TC)
  - E2/E3/E4/E5 seeds 42/123/456: `59504527`–`59504538`
  - Current queue snapshot: `59504504` running; others pending.

## Update (2026-02-08 21:30 EST)
- Integrity re-check completed after final VaR runs:
  - `59470884` (`vdc_paper_var_strong`) and `59470885` (`vdc_paper_var_fast`) are both `COMPLETED` with `ExitCode=0:0`.
  - Current paper cache uses strong-run VaR output (`drafts/paper_outputs/e3_var_results.json`, updated `20:57`).
- Artifact pipeline now regenerates two previously stale paper-facing figures from current artifacts:
  - `drafts/figures/fig2_e1_pareto_ise_vs_runtime.pdf` (now generated from selected checkpoint + baseline E1 model-selection JSON).
  - `drafts/figures/scaling_time_vs_d.pdf` (now generated from current E2 UCI timing records).
- `pdflatex` compile re-run passes (warnings only, no missing-file fallback boxes triggered).

## Update (2026-02-09 07:40 EST)
- Expanded-suite VaR jobs timed out at 12h:
  - `59504528` (seed=42), `59504532` (seed=123), `59504536` (seed=456) all `TIMEOUT`.
- Mitigation applied:
  - Increased default E3 walltime in `slurm/paper_e3_var.sh` from `12:00:00` to `24:00:00`.
  - Added `E3_SBATCH_TIME` to `slurm/submit_paper_expanded_suite.sh` (default `24:00:00`), and E3 submissions now pass explicit `--time`.
- Resubmitted timed-out E3 jobs on `kempner_eng` with `--time=24:00:00`:
  - `59591824` (seed=42), `59591825` (seed=123), `59591826` (seed=456).
  - Current state at submission check: all `PENDING`.

## Update (2026-02-08 21:00 EST)
- VaR GPU reruns on `kempner_eng` completed successfully:
  - `59470885` (`vdc_paper_var_fast`) completed in `01:21:57`
  - `59470884` (`vdc_paper_var_strong`) completed in `06:47:29`
- Fresh VaR artifact is now in paper cache:
  - `drafts/paper_outputs/e3_var_results.json` (updated `2026-02-08 20:57`)
  - `drafts/tables/tab_var.tex` regenerated from that artifact.
- Paper artifact generation now writes bivariate tables from the selected single checkpoint (`model_step_190000.pt`):
  - `drafts/tables/tab_bivariate.tex`
  - `drafts/tables/tab_bivariate_complex.tex`
- MI table now uses multi-seed summary when available:
  - `results/mi_benchmark_multiseed_summary.json`
  - `drafts/tables/tab_mi.tex` (mean±std; fallback to single-seed where multi-seed not yet available).
- `drafts/vine_diffusion.tex` appendix tables are now artifact-backed (no hardcoded numeric tables for imputation/TC/probit/IPFP sections).
- `pdflatex` compile passes for `drafts/vine_diffusion.tex` (warnings only: overfull/underfull boxes).

## Important Caveat (Current Data Quality)
- Current staged anomaly benchmark (`drafts/paper_outputs/e5_anomaly_results.json`) is saturated (`AUROC/AP ≈ 1.0` for all methods, including baselines), so it is not discriminative and should not be used as a headline comparative result without re-staging harder splits.
- Current imputation benchmark shows \ours{} underperforming best simple baselines (kNN/mean) on all tested datasets; claims have been revised to reflect this.

## Paper Model Decision
- Camera-ready paper will use a single model only: `denoiser_cond_enhanced`.
- Enhanced-CNN and diffusion runs are treated as internal diagnostics, not paper methods.

## Runs
- Denoiser enhanced run: `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687`
- Enhanced-CNN run: `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_enhanced_cnn_cond_20260207_160741_59393907`

## Best Checkpoints
- Denoiser standard-best: `model_step_70000.pt` (MI err=0.007865, Tau err=0.022406)
- Denoiser complex-best: `model_step_190000.pt` (MI err=0.004264, Tau err=0.024739)
- Denoiser joint best (std+complex): `model_step_190000.pt`
  - standard: MI err=0.008744, Tau err=0.026462, h-MAE=0.003015, ISE=5.129e-07
  - complex: MI err=0.004264, Tau err=0.024739, h-MAE=0.004010, ISE=9.929e-07
- Enhanced-CNN standard-best: `model_step_75000.pt` (MI err=0.010350, Tau err=0.022457)
- Enhanced-CNN complex-best: `model_step_5000.pt` (MI err=0.334728, Tau err=0.050060)

## Key Diagnosis
- Enhanced-CNN is competitive on standard bivariate suite but weak on complex suite across all checkpoints.
- Denoiser remains strong on both suites and dominates baselines on MI/h-function/ISE metrics.
- Joint checkpoint selection is necessary; selecting by standard-only can miss best overall checkpoint.

## Baseline Comparison (Denoiser joint-best checkpoint)
- Standard best baseline: `baseline:pyvine_param` with MI err=0.050250
- Complex best baseline: `baseline:histogram` with MI err=0.438129
- Denoiser joint-best standard MI err=0.008744
- Denoiser joint-best complex MI err=0.004264

## Produced Artifacts
- `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/results/model_selection_sweep.json`
- `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/results/model_selection_sweep_complex.json`
- `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/results/model_selection_joint_best.json`
- `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/results/model_selection_complex_joint_best.json`
- `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/results/model_selection_with_baselines.json`
- `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/results/model_selection_complex_with_baselines.json`
- `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/results/checkpoint_path_joint.txt`
- `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_enhanced_cnn_cond_20260207_160741_59393907/results/model_selection_sweep.json`
- `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_enhanced_cnn_cond_20260207_160741_59393907/results/model_selection_sweep_complex.json`

## Weak vs Strong (Same Model)
- Weak vs strong behavior is primarily **checkpoint selection**, not environment instability.
- In `vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687`:
  - Standard-suite best: `model_step_70000.pt` with MI err `0.007865`.
  - Complex-suite best: `model_step_190000.pt` with MI err `0.004264`.
  - Weak checkpoints exist in the same run:
    - Standard weak example: `model_step_80000.pt` MI err `0.029271`.
    - Complex weak example: `model_step_20000.pt` MI err `0.045016`.
- Joint ranking (`MI_std + MI_complex`) confirms `model_step_190000.pt` is best overall (`0.013008`), which explains why one recent run looked strong and another weak despite using the same architecture.

## Single-Model Benchmark Job Timeline
- Initial `kempner_eng` submissions (`59424869`, `59424870`, `59424883`, `59424891`, `59424892`, `59424893`, `59424894`) failed immediately:
  - root cause: `REPO_ROOT` resolved to `/var/slurmd/spool/slurmd` under SLURM.
- Patched scripts:
  - repo resolution via `SLURM_SUBMIT_DIR` fallback;
  - env activation via `VDC_CONDA_ENV_PATH` / `VDC_PYTHON_BIN`;
  - denoiser-only auto-checkpoint fallback for paper jobs.
- Resubmission (`59425571`, `59425572`, `59425573`, `59425585`, `59425592`, `59425607`, `59425608`) results:
  - completed: `59425573` (E4), `59425592` (TC), `59425607` (MI), `59425608` (MI consistency).
  - failed: `59425572` (E3), `59425585` (E5) due missing dataset files on holylfs.
- Staged missing datasets:
  - `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/datasets/finance/sp100_returns.npy`
  - `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/datasets/pyod/{arrhythmia,cardio,ionosphere,letter,pima}.npz`
- Reran failed jobs:
  - `59426029` (E3 VaR), `59426028` (E5 anomaly)

## Current Status (2026-02-07 22:05 EST)
- `COMPLETED`: `59425592` (TC), `59426028` (E5), `59425573` (E4), `59425607` (MI), `59425608` (MI consistency)
- `RUNNING`: `59425571` (E2 UCI), `59426029` (E3 VaR)

## Fresh Completed Metrics
- MI benchmark (`results/mi_benchmark_summary.json`):
  - DCD mean abs error: `0.01003` (better than KSG `0.02049`, InfoNCE `0.02347`, MINE `0.02145`, NWJ `0.02385`, Gaussian `0.06396`).
  - DCD runtime: `0.027 s` (faster than KSG `0.076 s`, much faster than neural MI baselines).
- MI self-consistency (`results/mi_self_consistency.json`):
  - DCD-Vine: DPI violation rate `0.0000`, additivity mean abs error `0.02124`, monotone invariance abs error `0.00000`, wall time `1.746 s`.
  - KSG: DPI violation rate `0.0333`, additivity mean abs error `0.02698`, monotone invariance abs error `9.34e-05`, wall time `23.528 s`.
- TC benchmark (`results/tc_benchmark.json`) with checkpoint `model_step_190000.pt`:
  - d=50: true `16.497`, DCD `16.244`, KSG `8.617` (large KSG underestimation; DCD close to truth).
  - d=20: true `6.397`, DCD `6.309`, KSG `5.010`.
  - d=10: true `3.030`, DCD `3.031`, KSG `2.839`.
- E5 anomaly (`drafts/paper_outputs/e5_anomaly_results.json`):
  - mean AUROC `0.5823`, mean AP `0.2692` over 5 datasets.
  - This is currently the weakest downstream result and likely needs either stronger feature handling or a better scoring protocol to be paper-competitive.
