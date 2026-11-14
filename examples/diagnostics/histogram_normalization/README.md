# Histogram normalization in pretrained inference

This diagnostic reproduces the input-normalization failure reported in
[issue #8](https://github.com/KempnerInstitute/vine-denoising-copula/issues/8).
It uses the original `vdc-denoiser-m64-v1` checkpoint with either its intended
density histogram or the probability-mass histogram substituted by the
[historical integration](https://github.com/vinecopulib/pyvinecopulib/blob/d363f97ecb5f1b6cf08d78648df515b9bf9315c8/src/pyvinecopulib/torch/_fit_vdc.py).
No checkpoint is retrained.

## Cause and API contract

The v0.1.0 wheel omitted `vdc.inference` and `vdc.vine`. The integration's
workaround substituted `vdc.data.hist.scatter_to_hist` for
`vdc.inference.density.scatter_to_hist`. The former returns cell probabilities
that sum to one. The latter returns a density histogram whose entries average
one, as required by the pretrained network. For interior samples on a uniform
64-by-64 grid, the substituted input is 4,096 times too small.

The current wheel includes the missing modules. The probability-mass utility
is now named `vdc.data.hist.scatter_to_mass_hist`; the old name remains an alias
for compatibility. The pretrained API performs the density normalization
internally. No manual histogram construction is needed to use it.

## Reproduction

Run on a CPU with at least 8 GB of available memory. Loading all three models
downloads about 1.6 GB of checkpoint data unless it is already cached. Model
files are checked against their published SHA-256 hashes by the loader.

The archived run used VDC v0.3.0 (`ee4162b`), PyTorch 2.9.1, NumPy 2.2.6,
pyvinecopulib 0.7.5, and four CPU threads. To reproduce that environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.9.1' 'numpy==2.2.6' 'pyvinecopulib==0.7.5'
python -m pip install 'vine-denoising-copula @ git+https://github.com/KempnerInstitute/vine-denoising-copula@v0.3.0'
python reproduce.py --output results.json --reference reference_results.json
```

Run these commands from this directory, or give the path to `reproduce.py`.
The script also works with the current library. A shorter check of the original
checkpoint uses `--methods v1,v1_shim --seeds 42`. The script records input hashes,
checkpoint hashes, versions and individual results, and exits unsuccessfully if
a requested method or evaluation fails. Its CPU timings are diagnostics, not
evidence of a GPU speed advantage.

`reference_results.json` contains all 300 family-case evaluations and 50
independence evaluations from the original diagnosis, with private import paths
and CPU timings removed. Accuracy values are unchanged. A replay should match
them within `atol=5e-6, rtol=5e-4`; differences beyond that tolerance should be
investigated rather than silently accepted. Exact equality is not promised
across different numerical libraries or devices.

## Results and scope

The script follows the upstream precision benchmark's four family/parameter
cases, sample sizes 500/2,000/10,000, seed-42 draw order, and PDF evaluation
convention. Seeds 43–46 and the independent-data probes are additional checks.
The error is the mean absolute PDF difference on 10,000 points uniformly drawn
from `[0.02, 0.98]^2`. Although the upstream script calls it IAE, this is an
interior-domain average, not an integral over the whole unit square; it does not
measure the omitted boundary region. V1 grids are replicate-padded and evaluated
bilinearly, as in that integration. V2 uses its native continuous copula API.

Mean error over seeds 42–46 at 10,000 training samples:

| Case | V1 with mass substitution | V1 with intended density input |
|---|---:|---:|
| Gaussian, rho = 0.3 | 0.778 | 0.009 |
| Gaussian, rho = 0.7 | 0.751 | 0.011 |
| Clayton, theta = 2 | 0.767 | 0.024 |
| Clayton, theta = 5 | 0.943 | 0.019 |
| Independence | 0.770 | 0.008 |

With seed 42 alone, the substituted histogram gives 0.7736, 0.7462, 0.7625 and
0.9175 on the four family cases, reproducing the issue's displayed VDC numbers.
On independence at 10,000 samples, the mean per-run grid maximum is 56.43 with
the substitution and 1.23 with the intended input (the largest individual-run
maximum with the intended input is 1.32).

The optional TLL rows use **C++ pyvinecopulib 0.7.5, constant TLL, default
30-point grid**. They are not a reproduction of the issue's Torch TLL comparison
at grid size 64. The v2 rows are separate retrained estimators and are not needed
to resolve the v1 failure. These checks do not establish a general ranking of
estimators, validate GPU timings, or test complete-vine integration.
