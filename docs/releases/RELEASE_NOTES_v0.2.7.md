# v0.2.7 — Training-data corrections and documentation

This patch corrects synthetic data for future training and makes the public
repository easier to install, use and maintain. The released pretrained model,
its checksum, and the inference implementation are unchanged. The NeurIPS 2026
paper and figure archive remain pinned to v0.2.6.

## Training data

- Replace incorrect Joe, BB1 and BB7 samplers with pyvinecopulib sampling and
  replace finite-difference BB densities with its analytic evaluations.
- Apply BB rotations consistently to samples and targets; include BB families
  in mixtures and draw exact sample counts from the mixture weights.
- Correct the Frank and Joe conditional CDFs.
- Repair the public mixture generator import and midpoint targets; keep streaming
  augmentation coordinates aligned with their grids and return contiguous arrays.
- Reject the historical `conditional_*` training examples: their samples did
  not match the target density. Keep conditional diagnostics as separate APIs.
- Raise errors for invalid examples instead of substituting uniform targets.
- Add distribution-level checks against closed-form CDFs and their derivatives.

See [training data](../TRAINING_DATA.md) for compatibility with historical runs.

## Documentation and maintenance

- Replace obsolete demonstrations with public-loader examples and an exact
  Gaussian mutual-information reference. Duplicate demonstration entry points
  now forward to the maintained example and announce their deprecation.
- Correct API signatures, batching defaults and architecture configuration.
- Consolidate release notes under `docs/releases/` and provide contribution,
  configuration, dataset and model-loading guides.
- Use the package metadata as the dependency source, constrain the qualified
  pyvinecopulib API to 0.7.x, and run the full library test suite in CI.
- Retain portable paper experiments under `projects/neurips2026/`. Manuscript
  builds and authoritative figure generators remain in the paper repository
  and the existing versioned figure archive.
