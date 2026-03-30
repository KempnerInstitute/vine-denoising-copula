# Vine Copula Correctness Audit (2026-02-11)

This note documents what the repo currently implements for vine copulas, what is missing, and
what we changed/need to change to ensure theoretical correctness (density + sampling).

## 1) What A Vine Copula Must Do (minimal checklist)

Given pseudo-observations `U ∈ [0,1]^{n×d}` (empirical PIT per marginal):

1. **Density factorization**: represent the copula density as a product of bivariate (pair) copula
   densities on a vine graph:

   - For each tree level `k=1..d-1`, edges represent pair-copulas `c_{ij|D}` with a conditioning set
     `D` of size `k-1`.
   - The joint copula density is:
     `c(u_1..u_d) = Π_{edges} c_{ij|D}(u_{i|D}, u_{j|D})`

2. **Conditional pseudo-observations** via h-functions:

   - For each pair-copula `C_{ij|D}`, define:
     - `h_{i|j;D}(u_{i|D}, u_{j|D}) = ∂C_{ij|D}(u_{i|D}, u_{j|D}) / ∂u_{j|D}`
     - `h_{j|i;D}(u_{j|D}, u_{i|D}) = ∂C_{ij|D}(u_{i|D}, u_{j|D}) / ∂u_{i|D}`
   - These produce the next-level conditionals:
     - `u_{i|D∪{j}} = h_{i|j;D}(u_{i|D}, u_{j|D})`
     - `u_{j|D∪{i}} = h_{j|i;D}(u_{j|D}, u_{i|D})`

3. **Sampling** (inverse Rosenblatt): to sample `U ~ c(·)`, start from `W ~ Unif([0,1]^d)` and
   invert a sequence of conditional CDF evaluations using inverse h-functions. (The exact recursion
   depends on vine type and order.)

4. **Boundary conditions** are crucial for correctness/stability:
   - For any fixed `v`, `h_{U|V}(0|v)=0` and `h_{U|V}(1|v)=1`.
   - `h(·|v)` must be monotone and `h^{-1}` must invert it.


## 2) What This Repo Implements

### 2.1 High-level API

- `vdc/vine/api.py` defines `VineCopulaModel`.
  - `fit()` builds a structure and fits each pair-copula by calling a *bivariate density estimator*
    (diffusion UNet or single-pass denoiser/CNN) to produce a density grid, then projects it to a
    valid copula and builds h-functions (`HFuncLookup`).
  - For `vine_type="rvine"`, `fit()` uses a Dißmann-style sequential MST loop implemented in
    `VineCopulaModel._fit_rvine_dissmann()` (not the simplified `build_rvine_structure()` helper).

### 2.2 Structure selection / vine types

- D-vine / C-vine:
  - `vdc/vine/vine_types.py:build_dvine_structure()` creates the standard D-vine conditioning sets
    induced by an order (path).
  - `vdc/vine/vine_types.py:build_cvine_structure()` creates the standard C-vine conditioning sets
    induced by an order (root sequence).
  - If `order=None`, the repo uses greedy heuristics based on `|Kendall tau|` to choose an order.

- R-vine:
  - `vdc/vine/api.py:_fit_rvine_dissmann()` performs a Dißmann-style sequential maximum spanning tree
    per tree level using conditional pseudo-observations from previously fitted pair-copulas.
  - `vdc/vine/structure.py:build_rvine_structure()` is explicitly simplified and should not be used
    for correctness-sensitive runs.

### 2.3 Density evaluation (logpdf)

- `vdc/vine/recursion.py:VineRecursion.logpdf()` caches conditional pseudo-observations and for each
  edge `(i,j|D)` multiplies by `c_{ij|D}(u_{i|D},u_{j|D})` while propagating
  `u_{i|D∪{j}}`, `u_{j|D∪{i}}` via h-functions.

This matches the standard pair-copula construction recursion.

### 2.4 Sampling

- `vdc/vine/recursion.py:VineRecursion.inverse_rosenblatt()` is implemented for **D-vines and C-vines only**.
- `vine_type="rvine"` sampling is **not implemented** (will raise `NotImplementedError`).

Implication:
- Any experiment that requires sampling from the fitted copula must use `vine_type ∈ {dvine, cvine}`
  unless/until R-vine sampling is implemented.


## 3) Critical Numerical Issue Found (and fixed)

### 3.1 Root cause

`vdc/models/hfunc.py:HFuncLookup` builds h-functions from a density grid using a cumulative sum.

However, the repo’s default grids are **cell centers**:
`u_grid = (0.5/m, 1.5/m, ..., (m-0.5)/m)` (same for v).

If you compute h at centers using a naive right-Riemann cumsum:
`h(u_k|v) ≈ Σ_{r<=k} c(u_r,v) du`,
then even for the independence copula (`c=1`) you get:
`h(u_k|v) = u_k + du/2`,
and boundary behavior near 0/1 becomes distorted when interpolation clamps values to the grid.

This is *not just cosmetic*: vine recursion uses these h-values as conditional pseudo-observations,
so a systematic bias can affect higher-tree fits and inverse Rosenblatt sampling (tails especially).

### 3.2 Fix implemented

We changed `HFuncLookup` to:
- Use a midpoint-rule correction for h-grids so that, for `c=1`, `h(u|v)≈u` at the grid centers.
- Enforce correct boundary behavior via linear extension so that `h(0|v)=0` and `h(1|v)=1`.
- In `hinv_*`, invert using endpoints `(0,0)` and `(1,1)` so extreme quantiles do not collapse to 0/1.

Code changes:
- `vdc/models/hfunc.py` (midpoint integration + boundary extension + endpoint-aware inverses)
- Added strict sanity tests:
  - `tests/test_hfunc_independence_accuracy.py`

Validation:
- `python -m pytest -q` passes (`36 passed`).
- Added a ground-truth 3D Gaussian D-vine test and re-ran full suite:
  - `tests/test_vine_gaussian_ground_truth.py`
  - `python -m pytest -q` passes (`38 passed` after adding new tests).


## 4) What Is Still Missing / Needs Clarification

1. **R-vine sampling**:
   - If we want to claim R-vine support beyond likelihood evaluation, we need to implement
     forward/inverse Rosenblatt for general R-vines (or clearly scope claims to D/C-vines).

2. **Matrix parsing**:
   - `vdc/vine/vine_types.py:build_vine_from_matrix()` includes an R-vine matrix parser that is marked
     simplified and currently falls back to automatic R-vine construction.

3. **Structure-order sensitivity**:
   - D/C-vines depend on `order`. We should report how `order` is chosen and optionally include an
     ablation: fixed order vs heuristic order vs (if available) external baseline.


## 5) Recommended Next Actions (paper-facing)

1. Decide *one* primary vine type for sampling-based experiments:
   - Recommend defaulting to **D-vine** (simple, sampleable, scalable), and include **C-vine** as an ablation.
   - Use R-vine only for log-likelihood benchmarks unless we implement R-vine sampling.

2. Re-run E2/E3/E4/E5 after the h-function fix:
   - E3 (VaR) is especially sensitive to tails; this fix is expected to matter.

3. Add a small “known ground truth” correctness check (optional but high value):
   - Build a 3D Gaussian D-vine using analytic Gaussian copula densities on the grid, and verify:
     - logpdf matches the known Gaussian copula density (within grid resolution),
     - inverse Rosenblatt samples have the right rank correlations.
