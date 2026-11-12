# Estimator and numerical conventions

## Single-pass and diffusion models

The released `vdc-denoiser-m64-v1` is a single-pass denoiser. It maps an empirical
pair histogram, with its configured conditioning channels, to a density grid.
A marginal projection then adjusts the grid toward uniform marginals.

The package also supports iterative `diffusion_unet` checkpoints. Their reverse
diffusion settings do not apply to the released denoiser. Load architectures
from the configuration stored in the checkpoint, using the public loader.

## Density grids

A uniform `m × m` grid uses cell centers `(i + 0.5) / m`. The array contains
density values, so a cell's probability mass is its density divided by `m**2`.
The first array axis corresponds to the first variable.

Iterative proportional fitting alternates row and column normalizations. A
finite iteration count leaves a numerical residual; it does not guarantee
exactly uniform continuous marginals. The implementation supports both density-
and mass-space calculations and log-space projection utilities.

## Conditional transforms and vines

Each edge stores a density grid and cached conditional CDF tables. D/C-vine
recursion uses those transforms to construct the next tree's inputs. Edges
within a tree may be processed in batches, while the trees remain sequential.

`HFuncLookup.pdf` interpolates density values; conditional tables use their own
normalization and interpolation. These are approximate counterparts. Therefore
projection of every grid does not by itself imply exact continuous normalization
of the assembled vine or exact agreement between its sampling and likelihood.
The sampling implementation has boundary and inverse-round-trip regression tests.

## Information estimates

Bivariate MI is computed by grid quadrature of `c * log(c)` in nats. For the
fitted vine, `logpdf_decomposed` sums log-density contributions over its edges.
The mean on held-out observations estimates total correlation under the model's
assumptions. Fitted edge contributions can be negative; their interpretation
differs from nonnegative population conditional mutual information.

## Limits and reproducibility

- The released model assumes continuous marginals and simplified pair copulas.
- Fixed grid resolution limits near-deterministic dependence and fine tail detail.
- Conditional calibration and deep-tree recursion need task-specific evaluation.
- Changing grids, batch sizes, precision or interpolation is a numerical change.
- Fast fitting and compiled sampling have qualified settings, described in their
  [fitting](fast_fit.md) and [sampling](compiled_sampler.md) guides.
- Paper experiments use explicit model hashes and software pins in the
  [NeurIPS project](../projects/neurips2026/README.md), not unspecified current defaults.

Probit-coordinate and transformed-density training options are experimental.
They should not be interpreted as proof that a checkpoint supports nonuniform
cells throughout vine likelihood and sampling.
