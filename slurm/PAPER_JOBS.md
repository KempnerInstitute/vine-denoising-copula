# Paper SLURM Jobs (ICML 2026): Vine Diffusion Copula

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

## New Information Estimation Jobs (ICML 2026)

### MI Self-Consistency Tests

Tests Data Processing Inequality, Additivity, and Monotone Invariance:

```bash
sbatch slurm/paper_mi_consistency.sh
```

Output: `drafts/tables/tab_self_consistency.tex`

### MI Estimation Benchmark

Compares all MI estimators (KSG, MINE, InfoNCE, NWJ, MINDE):

```bash
sbatch slurm/paper_mi_benchmark.sh
```

Output: `results/mi_benchmark_summary.json`

### Total Correlation Benchmark

Evaluates TC estimation across dimensions:

```bash
sbatch slurm/paper_tc_benchmark.sh
```

Output: `results/tc_benchmark.json`

### Enhanced Training (Conditional + BB Families)

Train with conditional copulas and two-parameter BB families for better vine estimation:

```bash
sbatch slurm/paper_train_enhanced.sh
```

This includes:
- Conditional copulas (`conditional_gaussian`, `conditional_clayton`)
- BB1/BB7 two-parameter families
- Complex synthetic patterns

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

## Seed-variance array (UCI NLL, 3 seeds)

Generates per-seed UCI results used by the neural-pair and edge-scaling
tables. Runs seeds `7, 17, 42` in parallel on `kempner_priority`.

```bash
sbatch slurm/paper_seed_variance_array.sh
```

Outputs `e10_uci_flow_results_seed{7,17,42}.json` under
`drafts/paper_outputs/`. After the array completes, regenerate tables:

```bash
python drafts/scripts/paper_artifacts.py all
```

## IPFP iteration ablation

Sweeps the number of Sinkhorn/IPFP projection iterations `K` and reports
maximum marginal error and wall-clock cost. Produces the data behind
`tab_ipfp_iter_ablation.tex`.

```bash
sbatch slurm/paper_ipfp_iter_ablation.sh
```

Output: `drafts/paper_outputs/ipfp_iter_ablation.json` and
`drafts/tables/tab_ipfp_iter_ablation.tex`.
