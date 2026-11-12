# User guide

VDC separates continuous marginal distributions from their dependence. It
estimates pair-copula densities with a pretrained network, then combines them
in a simplified vine. The same fitted vine supports density evaluation,
sampling and an edge-level information decomposition.

## Transform raw data without leaking test observations

Split observations before fitting the marginal transform:

```python
import numpy as np
from vdc.data.tabular import EmpiricalMarginals, train_test_split

X = np.load("measurements.npy")  # finite array, shape (rows, variables)
X_train, X_test = train_test_split(X, test_frac=0.2, seed=42)
marginals = EmpiricalMarginals.fit(X_train)
train_u = marginals.transform(X_train)
test_u = marginals.transform(X_test)
```

Use a chronological split for time series. Handle missing values explicitly
before fitting. Ties and discrete variables need additional modeling choices;
the released estimator assumes continuous marginals.

`EmpiricalMarginals` uses right-sided training ranks and `(rank + 0.5)/(n + 1)`,
clipped away from the endpoints. Its inverse is a nearest-rank empirical
quantile, not a smooth parametric marginal model.

## Load the reusable pair estimator

```python
from vdc import load_pretrained_model, estimate_pair_density_from_samples

bundle = load_pretrained_model("vdc-denoiser-m64-v1", device="cpu")
pair_density = estimate_pair_density_from_samples(bundle, train_u[:, :2])
```

The grid is 64 × 64 for this checkpoint. Loading downloads the weights on the
first use, verifies their hash and caches them. The model has about 133 million
parameters; allow several GB of RAM, or GPU memory when using `device="cuda"`.

## Fit and evaluate a vine

```python
from vdc import VineCopulaModel

vine = VineCopulaModel(
    vine_type="dvine", m=64, device=str(bundle.device),
    batch_edges=True, edge_batch_size=16,
)
vine.fit(train_u, bundle.model, diffusion=bundle.diffusion, verbose=False)
mean_log_density = float(vine.logpdf(test_u).mean())
print(vine.summary())
```

The log density is in copula space. A raw-data likelihood also needs the
marginal log densities; empirical ranks alone do not provide those densities.

D-vines and C-vines support fitting and the implemented sampling paths.
R-vines support density fitting but not general inverse-Rosenblatt sampling.
For large D-vines, consult [optimized fitting](fast_fit.md); batching is opt-in,
and changing batch size or numerical precision can affect floating-point results.

## Generate observations

```python
samples_u = vine.simulate(1000, seed=42)
samples_x = marginals.inverse_transform(samples_u)
```

For repeated sampling, `vine.prepare_sampler(backend="auto")` can build a
compiled D-vine sampler, with an automatic NumPy fallback. Supported structures,
interpolation settings, thread controls and memory limits are described in
[compiled sampling](compiled_sampler.md).

## Inspect information contributions

```python
import torch
from vdc.utils.metrics import mutual_information_from_density_grid

mi = float(mutual_information_from_density_grid(torch.as_tensor(pair_density)))
parts = vine.logpdf_decomposed(test_u)
print("Pair MI (nats):", mi)
print("Held-out mean log copula density (nats):", parts["mean_total"])
print("Tree contributions:", parts["per_tree_mean"])
```

For a correctly specified population copula, the expected log density is total
correlation. A fitted held-out estimate includes sampling and model error;
individual fitted contributions may be negative. It is not causal attribution.

## Save a fitted model

```python
vine.save("vine.pkl")
restored = VineCopulaModel.load("vine.pkl")
```

Save the marginal transform and preprocessing choices separately. Pickles and
Torch checkpoints can execute code when loaded: use trusted artifacts. Compiled
sampler handles are not serialized; prepare a new handle after loading.

## Choosing settings

- Keep the grid size consistent with the checkpoint. A denser grid is not a
  substitute for a model trained and evaluated for that representation.
- Increase `projection_iters` when marginal residuals matter, and measure them.
- Start with small edge batches on a CPU or limited-memory GPU.
- Choose truncation from training/validation data, not the test set.
- Evaluate conditional predictions for the intended task; good joint likelihood
  alone does not establish imputation or tail-risk calibration.

For benchmark protocols, use the [NeurIPS project](../projects/neurips2026/README.md).
Its fixed settings and release pins should not be inferred from library defaults.
