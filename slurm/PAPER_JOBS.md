# Paper SLURM Jobs (ICML 2026) — Vine Diffusion Copula

This folder contains **paper-style SLURM jobs** that create a timestamped run directory with:

```
<OUTPUT_BASE>/
  vdc_paper_<method>_<timestamp>_<jobid>/
    results/      # JSON/CSV metrics (model_selection)
    logs/         # train/eval logs + env.txt
    figures/      # symlink to training visualizations (if available)
    checkpoints/  # model_step_*.pt + visualizations/
    analysis/     # exact config copy + git info + slurm script copy
```

## One-command submission (per method)

All scripts default to:

- `OUTPUT_BASE=/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula`

Override as needed:

```bash
OUTPUT_BASE=/n/holylfs06/LABS/kempner_project_b/Lab/<your_folder> sbatch slurm/paper_vdc_diffusion_cond.sh
OUTPUT_BASE=/n/holylfs06/LABS/kempner_project_b/Lab/<your_folder> sbatch slurm/paper_vdc_denoiser_cond.sh
OUTPUT_BASE=/n/holylfs06/LABS/kempner_project_b/Lab/<your_folder> sbatch slurm/paper_vdc_enhanced_cnn_cond.sh
```

Each training job will automatically run `scripts/model_selection.py` on the latest checkpoint and write:

- `results/model_selection.json`
- `results/model_selection.csv`

## Scaling job (E2)

To generate the real scaling plot used in the draft (`drafts/figures/scaling_time_vs_d.pdf`), submit:

```bash
OUTPUT_BASE=/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula \
sbatch slurm/paper_vdc_scaling.sh
```

This writes a scaling JSON under a new run directory and also updates:

- `drafts/paper_outputs/e2_scaling_results.json`

Then it runs `drafts/scripts/paper_artifacts.py all --force` so the paper figure is refreshed automatically.

## Evaluation-only job

If you already have checkpoints and just want the standardized benchmark:

```bash
sbatch slurm/paper_vdc_model_selection.sh \
  /path/to/model_step_100000.pt \
  /path/to/other_model_step_100000.pt
```

This writes one combined JSON/CSV under a fresh run directory.

## Notes

- These jobs run `scripts/train_unified.py` via `torchrun` (DDP) for training.
- The training configs in `configs/train/*.yaml` are copied into the run directory and augmented with a top-level `checkpoint_dir` so the run is self-contained.

## MI estimators (KSG / MINE / MINDE)

To run MI estimation baselines:

```bash
sbatch slurm/paper_vdc_mi_estimation.sh ksg
sbatch slurm/paper_vdc_mi_estimation.sh mine
sbatch slurm/paper_vdc_mi_estimation.sh minde
```
