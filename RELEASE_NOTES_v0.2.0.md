# Vine Denoising Copula v0.2.0

Release `v0.2.0` accompanies the NeurIPS 2026 paper *Amortized Estimation of Simplified Vine
Copulas for High-Dimensional Density and Information Analysis*. The released model is unchanged:
`hsafaai/vdc-denoiser-m64-v1` (checkpoint sha256
`8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f`).

## Changes since v0.1.1

Correctness

- Inverse h-functions now interpolate through the final grid interval instead of using the
  boundary curve there (`vdc/models/hfunc.py`). This changes sampling, conditional sampling and
  everything built on the inverse Rosenblatt map (imputation, VaR); density, h-function and
  likelihood evaluation are unaffected. Regression test: `tests/test_hfunc_dependent_boundary.py`.

Faster sampling (opt-in)

- `fitted_model.prepare_sampler(...)` packs the fitted h tables once and returns a sampler handle.
  With `backend="auto"` it uses a cached C++17/OpenMP D-vine kernel and falls back to NumPy when
  compilation, loading or the numerical self-check fails, the vine is unsupported, or the packed
  tables exceed the memory limit (default 512 MiB); `sampler.backend` and
  `sampler.fallback_reason` report the choice. Existing `simulate` and `inverse_rosenblatt` calls
  keep their NumPy behavior. See `docs/compiled_sampler.md`.

Faster fitting (opt-in)

- `VineCopulaModel(..., fast_fit=True)` reuses D-vine ordering correlations, builds monotone
  h-function tables with vectorized NumPy operations and copies predicted grids from GPU to CPU in
  bulk. The checkpoint, grid, projection, network precision and `edge_batch_size` are unchanged and
  the default remains `False`. See `docs/fast_fit.md`.

Documentation and metadata

- README: projection constraints are described as exact in the IPFP limit and met to a controlled
  numerical tolerance by the deployed finite-iteration projection; the self-consistency summary
  now quotes the paper's 480-trial violation rates; the paper citation is updated.
- `paper/neurips2026/`: scripts and compact saved outputs that reproduce the paper's figures and
  tables, with run instructions, and portable runners for the main experiments (including the
  comparison of the retained training-corruption variants).
- `scripts/download_uci_datasets.py`: POWER, GAS, HEPMASS and MINIBOONE are rebuilt from the MAF
  archive used by the paper (Zenodo record 1161203), with archive, member and output-array hash checks;
  the outputs are identical to the paper's files. The earlier downloader saved the Combined Cycle Power
  Plant (CCPP) data as `uci/power.npz` and could not parse GAS; CCPP is now prepared only under its own
  name (`--ccpp`, `uci/ccpp.npz`). The optional extra `data` installs pandas and openpyxl. See
  `docs/DATASETS.md`.

## Verification

- `pytest tests/test_hfunc_dependent_boundary.py tests/test_prepared_sampler.py
  tests/test_compiled_sampler.py tests/test_fast_fit_identity.py tests/test_uci_maf_preprocess.py`
- `python scripts/verify_pretrained_release.py --model-id vdc-denoiser-m64-v1 --device cpu
  --out-dir docs/reports/pretrained_release`

`v0.2.0` supersedes `v0.1.1` as the recommended release point.
