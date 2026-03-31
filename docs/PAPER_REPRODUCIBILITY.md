# Paper Reproducibility (ICML 2026 Camera-Ready)

## Canonical Model (Single Source of Truth)

The paper uses one fixed model/checkpoint for all reported DCD/VDC information results.

- Canonical checkpoint path file: local private `analysis/PAPER_CHECKPOINT.txt` when available
- Canonical structured manifest: local private `analysis/PAPER_BEST_MODEL.json` when available
- Portable public fallback: the packaged pretrained release `vdc-denoiser-m64-v1`

Current frozen checkpoint:

`/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/checkpoints/model_step_190000.pt`

Packaged release model id:

`vdc-denoiser-m64-v1`

Model:

- Method tag: `denoiser_cond_enhanced`
- Model type: `denoiser`
- Selection rule: joint-best over standard + complex model-selection suites

## Checkpoint Resolution Order

Paper scripts resolve checkpoints in this order:

1. Explicit CLI `--checkpoint`
2. Environment variable `PAPER_CHECKPOINT`
3. local private `analysis/PAPER_CHECKPOINT.txt`
4. local private `analysis/PAPER_BEST_MODEL.json`
5. Hugging Face / packaged pretrained release via `vdc.pretrained.resolve_pretrained_checkpoint(...)`
6. Auto-discovery via `vdc.utils.paper.choose_best_checkpoint(...)`

The central resolver is in `vdc/utils/paper.py`.

Quick sanity command:

```bash
python scripts/show_paper_checkpoint.py
```

Release verification command:

```bash
python scripts/verify_pretrained_release.py \
  --model-id vdc-denoiser-m64-v1 \
  --device cpu \
  --out-dir docs/reports/pretrained_release
```

## Exact Reproduction Commands (Information Results)

Use these commands from repo root:

```bash
# Always pin the canonical checkpoint explicitly for strict reproducibility.
export PAPER_CHECKPOINT="$(python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1)"

# Table 3 MI benchmark (DCD row)
python scripts/mi_estimation.py \
  --estimator dcd \
  --checkpoint "${PAPER_CHECKPOINT}" \
  --n-samples 5000 \
  --seed 123 \
  --device cuda \
  --out-json results/mi_benchmark_dcd.json

# MI self-consistency (DPI/additivity)
MI_CONSISTENCY_CHECKPOINT="${PAPER_CHECKPOINT}" sbatch slurm/paper_mi_consistency.sh

# TC scaling benchmark
TC_CHECKPOINT="${PAPER_CHECKPOINT}" sbatch slurm/paper_tc_benchmark.sh

# Paper synthetic theory benchmark (includes MI-method comparisons)
python drafts/scripts/e6_theory_synthetic_benchmark.py \
  --checkpoint "${PAPER_CHECKPOINT}" \
  --device cuda \
  --out-json drafts/paper_outputs/e6_theory_synthetic_results.json

# Synthetic density-scaling figures used in Figure 3 / Appendix A1.
# These are separate from the main E6 artifact because they add the RealNVP flow
# baseline on fixed Gaussian and Clayton vine families.
python drafts/scripts/e6_theory_synthetic_benchmark.py \
  --checkpoint "${PAPER_CHECKPOINT}" \
  --device cuda \
  --scenarios gaussian_ar1 \
  --dims 5 10 20 50 \
  --n-train 5000 \
  --n-test 2000 \
  --n-trials 2 \
  --pyvine both \
  --gaussian-copula \
  --flow-realnvp \
  --flow-epochs 25 \
  --flow-batch-size 2048 \
  --flow-eval-batch-size 4096 \
  --mi-methods gaussian \
  --mi-pairs-per-case 1 \
  --mi-n-samples 1000 \
  --out-json drafts/paper_outputs/e6_gaussian_density_with_flow.json

python drafts/scripts/e6_theory_synthetic_benchmark.py \
  --checkpoint "${PAPER_CHECKPOINT}" \
  --device cuda \
  --scenarios clayton_vine \
  --dims 5 10 20 50 \
  --n-train 5000 \
  --n-test 2000 \
  --n-trials 2 \
  --pyvine both \
  --gaussian-copula \
  --flow-realnvp \
  --flow-epochs 25 \
  --flow-batch-size 2048 \
  --flow-eval-batch-size 4096 \
  --mi-methods gaussian \
  --mi-pairs-per-case 1 \
  --mi-n-samples 1000 \
  --out-json drafts/paper_outputs/e6_clayton_density_with_flow.json

python drafts/scripts/fig_density_scaling_e6.py \
  --gaussian-json drafts/paper_outputs/e6_gaussian_density_with_flow.json \
  --clayton-json drafts/paper_outputs/e6_clayton_density_with_flow.json
```

## Verification Checklist

- Confirm `analysis/PAPER_CHECKPOINT.txt` and `analysis/PAPER_BEST_MODEL.json` point to the same checkpoint when using the local private paper layout.
- Confirm generated result JSON files contain that same checkpoint path.
- Confirm `docs/reports/pretrained_release/PRETRAINED_RELEASE_VERIFICATION.md` and `docs/reports/pretrained_release/MI_BENCHMARK_DCD_RELEASE.md` were regenerated from the same checkpoint.
- Confirm the synthetic density-scaling artifacts exist when building the final paper figures:
  - `drafts/paper_outputs/e6_gaussian_density_with_flow.json`
  - `drafts/paper_outputs/e6_clayton_density_with_flow.json`
  - `drafts/figures/fig_density_scaling_clayton_main.pdf`
  - `drafts/figures/fig_density_scaling_family_compare.pdf`
- Keep this manifest frozen for camera-ready unless a deliberate model-change decision is made and documented.
- Publish the actual checkpoint outside git and update `vdc/resources/pretrained/vdc_denoiser_m64_v1.json` once the public Hugging Face repo exists.
