# v0.3.0 — Retrained estimators and one reproduction folder

**Linear-grid estimators.** The two estimators of the paper's exploratory-retraining appendix,
VDC-64 and VDC-128 (`vdc-denoiser-m64-v2`, `vdc-denoiser-m128-v2`), are now part of the
library: their continuous bilinear pair-copula with exact h-functions and inverses
(`vdc.models.linear_grid_copula`), the estimator and its evaluated settings
(`VineCopulaModel.from_pretrained`, `estimate_pair_copula_from_samples`), optional Newton
completion of the copula projection, configurable conditional clipping, and the pilot,
continuous-mixture data and fine-tuning recipes (`scripts/train_linear_grid_estimator.py`,
`configs/train/linear_grid_*.yaml`). The port reproduces the paper's saved evaluation arrays of
both checkpoints bitwise on the GPU type used for the paper, and 300-step training runs match the
original research code step by step. The checkpoints are registered with their SHA-256; until
their public upload, `load_pretrained_model(..., checkpoint_path=...)` loads a local copy.
See [linear-grid estimators](../IMPROVED_ESTIMATORS.md).

**Defaults unchanged.** All features are opt-in. For `vdc-denoiser-m64-v1`, pair densities, vine
log densities, Rosenblatt transforms, samples and the prepared sampler are bitwise identical to
v0.2.7 on CPU and GPU.

**One reproduction folder.** `paper/` holds the NeurIPS 2026 experiment runners, reference
outputs, the figure and table generators with their hash-checked saved inputs, and the ablation
training configurations, with one guide that maps every figure and table to its sources. The
figures regenerate pixel-identically and the 25 generated tables identically to the paper.

**Repository.** New README and documentation landing page with figures; maintained scripts in
`scripts/`, the GridUNet workflow scripts in `scripts/legacy/`; deprecated demo shims removed.
