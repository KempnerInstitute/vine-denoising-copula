# API reference

Start with the functions exported by `vdc`. Examples assume the package is
installed and inputs are finite arrays of continuous pseudo-observations in
`[0, 1]`. Raw-data transforms are described in the [user guide](USER_GUIDE.md).

## Pretrained models

```python
from vdc import load_pretrained_model, estimate_pair_density_from_samples

bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")
density = estimate_pair_density_from_samples(bundle, pair_data)
```

| Function | Arguments and result |
|---|---|
| `load_pretrained_model` | `model_id="vdc-denoiser-m64-v1"`, keyword arguments `device=None`, `cache_dir=None`, `force_download=False`, `prefer_local=True`, `repo_id=None`, `revision=None`; returns a `LoadedPretrainedModel`. |
| `load_checkpoint_bundle` | A trusted `checkpoint_path`, keyword `device=None`; loads a local unified-trainer checkpoint. |
| `resolve_pretrained_checkpoint` | The same model/download options, without `device`; returns a verified checkpoint `Path`. |
| `estimate_pair_density_from_samples` | `bundle`, an `(n, 2)` array, keywords `m=None`, `diffusion_steps=None`, `cfg_scale=1.0`, `projection_iters=50`; returns an `(m, m)` density grid. |
| `list_pretrained_models()` | Returns the packaged model manifests. |
| `load_pretrained_manifest(model_id)` | Returns one packaged manifest. |

A bundle contains `model`, `diffusion`, `config`, `device`, `model_id`,
`checkpoint_path` and `manifest`. `diffusion` is `None` for the released
single-pass denoiser. `device=None` selects CUDA when available; choose an
explicit device for reproducible resource use. The loader verifies the public
checkpoint's SHA-256. See [pretrained models](MODEL_RELEASES.md) for caching.

The grid uses cell centers `(i + 0.5) / m`. Values are **densities**, not cell
probabilities: on a uniform grid, the mass is `density.sum() / m**2`.
Projection corrects grid marginals to finite numerical tolerance.

## VineCopulaModel

```python
from vdc import VineCopulaModel

vine = VineCopulaModel(
    vine_type="dvine", m=64, device="cpu",
    batch_edges=True, edge_batch_size=16,
)
vine.fit(train_u, bundle.model, diffusion=bundle.diffusion, verbose=False)
log_density = vine.logpdf(test_u)
```

Constructor defaults in the library are:

```python
VineCopulaModel(
    vine_type="rvine", order=None, truncation_level=None, m=64,
    device="cuda", diffusion_steps=50, cfg_scale=1.0, projection_iters=50,
    pred_noise_clip=10.0, hfunc_use_spline=True, batch_edges=False,
    edge_batch_size=256, gpu_histogram=True,
    hfunc_vectorized_inverse=False, fast_fit=False,
)
```

Select `vine_type` and `device` explicitly. `order` gives the D-vine order or
C-vine root order; `None` selects it from the data. `truncation_level=None`
fits all `d - 1` trees. `m` must match the estimator's grid.

`batch_edges=True` enables batches within D/C-vine trees for single-pass models;
R-vine fitting evaluates edges individually. Trees remain sequential.
`fast_fit=True` enables the qualified execution optimizations described in
[optimized fitting](fast_fit.md). Neither option changes the network weights.

| Method | Behavior |
|---|---|
| `fit(U, diffusion_model, diffusion=None, verbose=True)` | Fits structure and pair densities to `(n, d)` inputs. |
| `logpdf(U)`, `pdf(U)` | Return one copula log density or density per row. |
| `logpdf_decomposed(U)` | Returns `per_sample`, `mean_total`, `per_tree_mean` and `per_edge_mean`. |
| `simulate(n, seed=None)` | Draws `(n, d)` samples for supported D/C-vines. A supplied seed resets NumPy's global random state. |
| `rosenblatt(U)`, `inverse_rosenblatt(W)` | Forward and inverse transforms for supported D/C-vines. |
| `prepare_sampler(...)` | Prepares a reusable sampler with automatic compiled/NumPy selection; see [sampling](compiled_sampler.md). |
| `summary()` | Returns the fitted dimension, tree/edge counts, type, order, truncation and grid size. |
| `get_structure_matrix()` | Returns the fitted vine matrix. |
| `save(path)`, `VineCopulaModel.load(path)` | Save/load a fitted model with pickle. Load only files you trust. |

General R-vine density fitting is supported; general R-vine sampling is not.
`fit_from_matrix(U, vine_matrix, diffusion_model, diffusion, verbose=True)` uses
the diagonal order for D/C-vines. The R-vine path selects its own structure;
it does not enforce an arbitrary supplied matrix.

## Grid transforms and information

| Function or object | Interface |
|---|---|
| `vdc.models.hfunc.HFuncLookup` | Construct from an `(m, m)` density grid; optional `use_spline=True`, `vectorized_inverse=False`, `vectorized_tables=False`. |
| `HFuncLookup.h_u_given_v(u, v)` | Conditional CDF of `u` given `v`. |
| `HFuncLookup.h_v_given_u(v, u)` | Conditional CDF of `v` given `u`; note the argument order. |
| `HFuncLookup.hinv_u_given_v(q, v)` | Inverse conditional CDF of `u` given `v`. |
| `HFuncLookup.hinv_v_given_u(q, u)` | Inverse conditional CDF of `v` given `u`. |
| `HFuncLookup.pdf(u, v)` | Interpolated pair density. |
| `vdc.models.projection.copula_project` | Projects a tensor density grid; `iters=30`, `eps=1e-8`, `method="sinkhorn"`, optional marginal targets. |
| `vdc.utils.metrics.mutual_information_from_density_grid` | Accepts a Torch density tensor and computes grid-integrated MI in nats. |
| `vdc.utils.information.ksg_mutual_information` | `x, y, k=5, seed=None` with keyword-only `k` and `seed`; KSG MI in nats. |
| `vdc.data.tabular.EmpiricalMarginals` | `fit(X_train)`, then `transform(X)` or `inverse_transform(U)`. |

The continuous density interpolation and cached conditional tables are
approximate counterparts. Grid projection does not imply exact normalization
or calibration of the complete interpolated vine.

## Command line

```bash
vdc list-models
vdc resolve-model --model-id vdc-denoiser-m64-v1
vdc estimate-pair pair.npy --output density.npy --device cpu
vdc fit-vine pseudo_obs.npy --output vine.pkl --vine-type dvine --device cpu
```

Array inputs may be `.npy`, comma-separated `.csv` or whitespace-separated
`.txt` files. Use `vdc COMMAND --help` for the complete options. These commands
expect pseudo-observations, not untransformed raw measurements.

## Advanced models

`vdc.GridUNet` and `vdc.CopulaAwareDiffusion` support iterative diffusion
checkpoints. They are not the released single-pass denoiser. Use
`load_checkpoint_bundle` to reconstruct an architecture from its checkpoint,
rather than guessing constructor arguments. Training options are in
[configuration](CONFIGURATION.md).
