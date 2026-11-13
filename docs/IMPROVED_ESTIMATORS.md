# Linear-grid estimators VDC-64 and VDC-128

The NeurIPS 2026 paper's appendix *Exploratory retraining on corrected data*
evaluates two estimators trained after the original submission. This library
version carries their inference, training and evaluated settings as opt-in
features. The paper's main results, and the library defaults, still use
`vdc-denoiser-m64-v1`.

| Model ID | Paper name | Grid | Training inputs | Checkpoint SHA-256 |
|---|---|---|---|---|
| `vdc-denoiser-m64-v2` | VDC-64 | 64 × 64 | latent uniforms with probability 0.5, otherwise ranks | `73097772a18eb4b02fb0608b4bfa87c0205b71667d97af6a39483fd44d0df0fc` |
| `vdc-denoiser-m128-v2` | VDC-128 | 128 × 128 | latent uniforms with probability 0.75, otherwise ranks | `f0a753d50b3cfb089e6d7dec999cfe741c6342c188454ec83183a9b3514d4653` |

**Released** on Hugging Face: [`hsafaai/vdc-denoiser-m64-v2`](https://huggingface.co/hsafaai/vdc-denoiser-m64-v2)
and [`hsafaai/vdc-denoiser-m128-v2`](https://huggingface.co/hsafaai/vdc-denoiser-m128-v2), with the
pilot they were fine-tuned from at [`hsafaai/vdc-linear-grid-pilot`](https://huggingface.co/hsafaai/vdc-linear-grid-pilot).
The packaged manifests ([m64](../vdc/resources/pretrained/vdc_denoiser_m64_v2.json),
[m128](../vdc/resources/pretrained/vdc_denoiser_m128_v2.json)) pin each upload's revision and record
the checkpoint hash and the evaluated settings.

## What differs from the original model

Both networks use the original U-Net (132,953,729 parameters; inputs:
histogram, log *n* and two probit coordinate channels). They were not
fine-tuned from `vdc-denoiser-m64-v1`:

1. A 60,000-step **pilot** was trained from random weights on exact cell-mass
   targets of corrected synthetic copulas, with histograms of rank
   pseudo-observations drawn from those targets as inputs
   (`vdc.train.linear_grid_pretraining`).
2. Each model was **fine-tuned** from the pilot for 8,000 steps on fresh
   continuous mixtures (`vdc.data.continuous_mixtures`): eight families
   (Gaussian, Student-t, Clayton, Gumbel, Frank, Joe, BB1, BB7) with signs or
   rotations, component Kendall's tau in five bins spanning 0.005–0.85,
   mixtures of 2–8 components (probability 0.70), independence (0.05), and
   sample sizes from 50 to 50,000. The objective is the negative
   log-likelihood of 512 independent points per population under the
   projected continuous density (`vdc.train.linear_grid_training`).
3. The output is a **continuous bilinear density** whose node values are
   projected to uniform margins with the integrated basis weights of that
   density. Its conditional CDFs (h-functions) and their inverses are exact
   for the same density (`vdc.models.linear_grid_copula.LinearGridCopula`).
   The original model instead interpolates a cell-average grid and its
   h-function tables approximately.

The cells are uniform; the probit coordinate features do not imply
probit-spaced cells.

## Use

```python
import numpy as np
from vdc import (VineCopulaModel, estimate_pair_copula_from_samples,
                 estimate_pair_density_from_samples, load_pretrained_model)

# Downloads once, verifies the SHA-256 and caches (checkpoint_path=... loads a verified local copy).
bundle = load_pretrained_model("vdc-denoiser-m128-v2", device="cuda")

# uv: (n, 2) pseudo-observations; u, v, q: arrays in [0, 1];
# U_train, U_test: (n, d) pseudo-observations.
copula = estimate_pair_copula_from_samples(bundle, uv)
density = copula.pdf(u, v)              # continuous density
cdf_u_given_v = copula.h_u_given_v(u, v)
u_from_q = copula.hinv_u_given_v(q, v)  # exact inverse of the h-function
grid = estimate_pair_density_from_samples(bundle, uv)  # (128, 128) node values, float32

# Vine with the evaluated settings (batched greedy-|tau| D-vine).
vine = VineCopulaModel.from_pretrained(bundle, vine_type="dvine")
vine.fit(U_train, diffusion_model=bundle.model, verbose=False)
log_density = vine.logpdf(U_test)
pit = vine.rosenblatt(U_test)
samples = vine.inverse_rosenblatt(np.random.default_rng(0).uniform(size=(1000, U_train.shape[1])))
```

The model downloads and caches like `vdc-denoiser-m64-v1`. A file placed at
`$VDC_PRETRAINED_CACHE/<model id>/<model id>.pt` (default cache `~/.cache/vdc`)
is also found and verified. `estimate_pair_copulas_from_samples` estimates many
pairs in one call. On GPUs, a pair's estimate can depend on the other pairs in
the same call at the level of float32 rounding: for the 64-cell model, pairs
estimated one at a time differed from the batched estimates by up to 1.4e-6
(relative) on both A100 and H100, while the 128-cell estimates were identical.
To reproduce a recorded evaluation exactly, pass the same pairs in the same
order (vine fits group the edges of each tree as in the evaluation). On the
command line, add `--checkpoint-path` to `vdc estimate-pair` or `vdc fit-vine`;
the latter applies the same settings as `from_pretrained`.

`VineCopulaModel.from_pretrained` sets the following options, which can also
be passed directly to `VineCopulaModel` (and overridden in `from_pretrained`):

| Option | Evaluated value | Meaning |
|---|---|---|
| `pair_estimator` | `"linear_grid"` | Inference and pair copulas described above (default `"legacy"`). |
| `m` | 64 or 128 | Grid size; must match the checkpoint. |
| `projection_iters` | 30 (15 for S&P100) | Float32 basis-weight scalings after the network. |
| `copula_projection_solver` | `"hybrid"` | Float64 re-projection of each pair copula to marginal error 1e-11; Newton completion if IPFP has not converged after 10,000 iterations. |
| `conditional_clip_eps` | `1e-12` | Clipping of pair inputs and conditional pseudo-observations (default `1e-6`). |
| `forward_batch_size` | 16 | Network batches of exactly 16 edges, padding the last one. |
| `edge_batch_size`, `batch_edges` | 64, `True` | Edges estimated together within a tree. |
| `hfunc_use_spline` | `False` | Not used by linear-grid pair copulas. |

The network runs in float32 with TF32 and cuDNN benchmarking disabled during
the forward pass; the previous global settings are restored afterwards.
Inputs outside [0, 1] are rejected. Iterative diffusion, the compiled sampler
(`prepare_sampler` falls back to NumPy) and `hfunc_vectorized_inverse` do not
apply to linear-grid vines. C-vines, R-vines and unbatched fitting run with
the same pair estimator but were not part of the evaluation below.

## Evaluation

All numbers are from the paper's appendix (one H100 GPU, four CPU threads,
float32, TF32 disabled). They are post-selection assessments of fixed
checkpoints; no checkpoint was chosen with these results.

**Bivariate likelihood** on the nine cases of the paper's Table 1, five sample
seeds each, 2,000 training and 20,000 test observations per fit (mean ±
standard deviation over 45 fits of held-out KL, nats):

| Estimator | Held-out KL | Lower KL than original |
|---|---|---|
| Original VDC (`vdc-denoiser-m64-v1`) | 0.00611 ± 0.00437 | – |
| VDC-64 | 0.00541 ± 0.00406 (−11.5%) | 37/45 |
| VDC-128 | 0.00311 ± 0.00213 (−49.1%) | 41/45 |
| pyvinecopulib TLL | 0.01348 ± 0.00760 | 4/45 |
| Parametric selection | 0.00036 ± 0.00082 | 42/45 |

**Full vines** on the paper's UCI inputs (MAF splits; NLL in bits/dim, lower
is better; mean marginal PIT-KS over all retained test rows):

| Dataset | Train | Original | TLL | VDC-64 | VDC-128 | KS-64 | KS-128 |
|---|---|---|---|---|---|---|---|
| POWER | 5k | −0.1413 | −0.1822 | −0.1544 | −0.1639 | 0.0170 | 0.0149 |
| GAS | 5k | −1.4773 | −1.7050 | −1.5461 | −1.5957 | 0.0448 | 0.0368 |
| HEPMASS | 5k | −0.2372 | −0.2009 | −0.2579 | −0.2643 | 0.0172 | 0.0161 |
| MINIBOONE | 5k | −0.9689 | −0.7444 | −0.9550 | −0.9661 | 0.0242 | 0.0232 |
| POWER | 50k | −0.1495 | −0.2169 | −0.1802 | −0.1924 | 0.0117 | 0.0103 |
| GAS | 50k | −1.4926 | −1.8113 | −1.6149 | −1.6800 | 0.0314 | 0.0245 |
| HEPMASS | 50k | −0.2468 | −0.2653 | −0.2845 | −0.2936 | 0.0150 | 0.0138 |
| MINIBOONE | 29,556 | −0.9784 | −0.8716 | −0.9791 | −0.9856 | 0.0229 | 0.0202 |

VDC-128 improves on the original model in seven of eight settings; no
estimator leads in every setting.

**S&P100, chronological split** (339 held-out days, `projection_iters=15`):
NLL −0.41833 (VDC-64) and −0.41876 (VDC-128) versus −0.41849 for the original
model and −0.29276 for TLL (bits/dim). Paired differences from the original
are small; 40-day moving-block bootstrap 95% intervals are
[−0.00277, 0.00317] (VDC-64) and [−0.00345, 0.00323] (VDC-128).

**Cost.** Single-run S&P100 fits took 16.0 s (VDC-64) and 81.0 s (VDC-128),
versus 15.6 s for four-thread TLL. A warmed batch of 45 bivariate edges took
4.92 and 18.87 ms per edge. The original model's speed ratios do not transfer
to these estimators.

**Replicates.** A second fine-tuning seed gave Table 1 reductions of 11.8% and
49.0% (38/45 and 41/45 fits). A pilot trained from a different random
initialization and fine-tuned with the same recipes gave 11.7% and 48.9%
(36/45 and 41/45), but its VDC-128 was worse than the original on
chronological S&P100 by 0.0053 bits/dim (interval [0.0021, 0.0087]). The
released checkpoints are the first fine-tuning seed of the first pilot.

## Limitations

- Exploratory estimators trained after the paper's review; they do not
  replace the paper's main accuracy or speed claims.
- Model development also used copula families from the paper's
  out-of-distribution suite. The bivariate results are in-distribution, and
  there is no clean out-of-distribution evaluation.
- The random S&P100 split is excluded: 272 of its 339 test days overlap
  model-development data, including 64 used to score candidates. Only the
  chronological split is held out.
- One pilot and one fine-tuning seed are released; the replicates above show
  the size of seed-to-seed variation.
- Parametric selection remains more accurate on in-library families.
- Results are bitwise reproducible only on the recorded device type and
  software (see below). Other GPUs give floating-point differences that grow
  with vine depth.
- Continuous margins and the simplifying assumption are required, as for
  every VDC vine.

## Retraining

The recipes are in `configs/train/linear_grid_*.yaml`; each stage runs with

```bash
python scripts/train_linear_grid_estimator.py configs/train/linear_grid_pilot.yaml
python scripts/train_linear_grid_estimator.py configs/train/linear_grid_cache.yaml
python scripts/train_linear_grid_estimator.py configs/train/linear_grid_vdc64_v2.yaml
python scripts/train_linear_grid_estimator.py configs/train/linear_grid_vdc128_v2.yaml
```

`key=value` arguments override entries, for example
`output_dir=runs/m64 initial_checkpoint=runs/pilot/pilot_ema_step60000.pt`.

| Stage | Output | Resources of the recorded run |
|---|---|---|
| Pilot (`stage: pretrain`) | 200,000 exact target grids (3.3 GB, built once from seed 42 in about 8 CPU minutes), then 60,000 steps of batch 128, EMA 0.9995 | about 1.7 h on one H200 |
| Data (`stage: cache`) | 8,000 batches of 128 populations as lossless uint16 counts for both grids, rank and latent inputs (about 11 GB) | about 45 min with 24 CPU processes |
| VDC-64 (`stage: finetune`) | `ema_step8000.pt` | about 23 min on one H100 |
| VDC-128 (`stage: finetune`) | `ema_step8000.pt` | about 63 min on one H100 |

A CUDA GPU is required for the recorded numerics (CPU runs are for tests).
Fine-tuning uses AdamW (peak learning rate 5e-5, 200 warmup steps, cosine
decay to 1%), gradient clipping at 1, EMA 0.999, batch 128 in microbatches of
32, bf16 autocast with float32 projection and loss, seed 202610091. Data
seeds depend only on the example index, not on worker scheduling. Load a
result with `vdc.load_checkpoint_bundle(path)`;
`VineCopulaModel.from_pretrained` recognizes its embedded linear-grid settings.

The fine-tuning configurations check the released pilot's hash
(`6e48f8c65012f4170ee9003c4aca28600abdb602ef624e47c3c3f92c0f5ace20`), published as
[`hsafaai/vdc-linear-grid-pilot`](https://huggingface.co/hsafaai/vdc-linear-grid-pilot); pass its
path as `initial_checkpoint=`. A new pilot run on other hardware is a new initialization (set
`initial_sha256=null`). The pilot bank
and the fine-tuning data are deterministic for the recorded software
(NumPy 2.2.6, SciPy 1.15.3, pyvinecopulib 0.7.5).

Only the two released recipes and their pilot are included; other latent
probabilities remain available through `latent_probability`. Research arms
that produced no released model are omitted: probit-spaced cells, the legacy
cell-mass objective and fixed-bank data for fine-tuning, distillation, partial
zero-corruption, sample-size-dependent latent inputs, conditional-copula input
banks, fine-tuning-stage training from scratch, compiled and bf16 inference
variants and adaptive shrinkage. Interrupted runs cannot be resumed; rerun the
stage.

## Port verification

The library implementation was compared with the research code that
produced the appendix: library commit `117651d` with the paper's evaluation
adapters for inference, and the paper's pilot and fine-tuning scripts for
training (PyTorch 2.9.1+cu128, NumPy 2.2.6, SciPy 1.15.3, pyvinecopulib 0.7.5).

- **Same device type as the appendix (one H100 80GB).** The library
  reproduced the saved appendix outputs bitwise: Table 1 node grids and test
  log densities of both models (mean held-out KL 0.0054080 and 0.0031109
  nats, 11.5% and 49.1% below the original model's 0.0061112), and, for both
  models on all eight UCI settings and chronological S&P100, the test log
  densities, PITs over all test rows, generated samples and their Rosenblatt
  round trips.
- **Another device type (one A100 40GB).** Library and research code agreed
  bitwise on all of the above. Relative to the H100 outputs, Table 1 grids
  differed by at most 1.6e-5 (relative) and test log densities by at most
  1.3e-5 nats; the KL reductions remain 11.5% and 49.1%. Vine log densities
  differed by up to 0.2 nats per row on UCI and 1 nat on the 100-dimensional
  S&P100 vine, because higher-tree inputs are histogrammed and rounding can
  move observations between cells; mean NLL differed by at most 3.5e-5 (UCI)
  and 2.3e-4 (S&P100) bits/dim.
- **Default path.** With `vdc-denoiser-m64-v1`, the previous `main` and this
  version gave bitwise identical pair densities, vine log densities, forward
  and inverse Rosenblatt transforms and samples (D-, C- and R-vines; default
  and batched options) on CPU and on an A100.
- **Training.** On one A100, 300 steps of each fine-tuning recipe gave
  bitwise identical losses, gradient norms and EMA weights (steps 1, 100, 200
  and 300) in the research script and the library; two research runs were
  also identical. The same holds for 300 pilot steps (losses, raw and EMA
  weights at steps 100, 200 and 300). The rebuilt 200,000-grid pilot bank has
  the original file hash, and 20 rebuilt fine-tuning batches (two per sample
  size) match the original data arrays. Compared with the logs of the
  original runs on other device types, the first 300 fine-tuning losses
  differed by at most 2.5e-4 (H100) and the pilot losses by up to 1.3% (H200,
  training from random weights); the learning-rate schedules were identical.
