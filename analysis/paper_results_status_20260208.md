# Paper Results Status (2026-02-08)

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
