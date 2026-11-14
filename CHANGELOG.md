# Changelog

Published versions are immutable. The NeurIPS 2026 paper pins v0.2.6;
its checkpoints and reproduction archives remain available.

## v0.4.0 — 2026-11-13

- Separate pretrained model construction from the training engine. The base
  install no longer requires pyvinecopulib, Matplotlib, HDF5, Hydra or
  scikit-learn; use `[train]`, `[baselines]`, `[viz]` and `[data]` as needed.
- Require Python 3.10 or later and test Python 3.10–3.13.
- Add installed-model checks and TestPyPI/PyPI Trusted Publishing. Model
  weights, constructor calculations and inference defaults are unchanged.


- Name the probability-mass histogram `vdc.data.hist.scatter_to_mass_hist`
  and retain `scatter_to_hist` as a compatibility alias. Document the distinct
  density normalization required by pretrained inference, and test that
  input contract through the public estimator. Histogram values and model
  outputs are unchanged.
- Add a portable reproduction and reference results for the histogram
  normalization failure reported in issue #8.

## v0.3.0 — 2026-11-12

- Rewrite the README and the documentation landing page with the method overview, results
  figures, pretrained-model table, training commands and reproduction guide.
- Keep maintained entry points in `scripts/`; move the GridUNet workflow scripts to
  `scripts/legacy/` and remove the two deprecated demo shims (use `examples/`).
- Gather all NeurIPS 2026 reproduction material in `paper/`: the experiment runners and
  reference outputs (formerly `projects/neurips2026/`), the figure and table generators with
  their saved inputs (formerly distributed only as a release archive) and the ablation training
  configurations (formerly in `configs/train/`). Saved outputs, runner defaults and numerical
  behavior are unchanged; [`paper/README.md`](paper/README.md) maps every figure and table to
  its script, inputs and runner.
- Add the opt-in linear-grid estimator of the paper's exploratory retraining
  appendix: `LinearGridCopula` (consistent PDF, h and inverse-h functions),
  `VineCopulaModel(pair_estimator="linear_grid", ...)`,
  `VineCopulaModel.from_pretrained` and `estimate_pair_copula_from_samples`.
  Defaults and the results of `vdc-denoiser-m64-v1` are unchanged.
- Add configurable conditional clipping (`conditional_clip_eps`, default 1e-6).
- Release `vdc-denoiser-m64-v2` and `vdc-denoiser-m128-v2` on Hugging Face (manifests pin the
  upload revisions) and their training pilot `vdc-linear-grid-pilot`; `checkpoint_path=` loads a
  hash-verified local copy of any registered model.
- Add their pretraining, continuous-mixture data and fine-tuning recipes
  (`scripts/train_linear_grid_estimator.py`, `configs/train/linear_grid_*.yaml`).
- Fix `stage_release_bundle` for manifests without local source files.

Details: [linear-grid estimators](docs/IMPROVED_ESTIMATORS.md).

## v0.2.7 — 2026-11-12

- Correct Joe/BB1/BB7 training samples and BB density targets; test against analytic CDFs.
- Correct Frank/Joe conditional CDFs, BB rotations and mixture counts; reject invalid conditional training pairs.
- Replace obsolete API examples with programs using the released model loader.
- Correct API signatures and document numerical limits and opt-in batching.
- Simplify installation, consolidate release documentation and expand CI coverage.
- Keep manuscript production and internal research records outside the library.

Details: [v0.2.7 release notes](docs/releases/RELEASE_NOTES_v0.2.7.md).

## Earlier releases

- [v0.2.6](docs/releases/RELEASE_NOTES_v0.2.6.md)
- [v0.2.5](docs/releases/RELEASE_NOTES_v0.2.5.md)
- [v0.2.4](docs/releases/RELEASE_NOTES_v0.2.4.md)
- [v0.2.3](docs/releases/RELEASE_NOTES_v0.2.3.md)
- [v0.2.2](docs/releases/RELEASE_NOTES_v0.2.2.md)
- [v0.2.1](docs/releases/RELEASE_NOTES_v0.2.1.md)
- [v0.2.0](docs/releases/RELEASE_NOTES_v0.2.0.md)
- [v0.1.1](docs/releases/RELEASE_NOTES_v0.1.1.md)
- [v0.1.0](docs/releases/RELEASE_NOTES_v0.1.0.md)
