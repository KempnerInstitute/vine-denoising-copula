# Vine Types: D-vine, C-vine, and R-vine Support

## Overview

The package now supports **three types of vine copulas**:

1. **R-Vine (Regular Vine)**: Most flexible, automatic structure selection via MST
2. **D-Vine (Drawable Vine)**: Sequential/path structure, requires variable ordering
3. **C-Vine (Canonical Vine)**: Star structure with root variables at each level

All three types can be fitted:
- **Automatically** (with optimized structure selection)
- **With custom ordering/structure**
- **From vine matrices** (compatible with `pyvinecopulib` format)

## Quick Reference

### R-Vine (Regular Vine)

**Use when**: Unknown/complex dependency structure
**Structure**: Flexible - uses MST on |Kendall's τ|

```python
from vdc.vine.api import fit_rvine

# Automatic structure selection
vine = fit_rvine(U, diffusion_model)
```

### D-Vine (Drawable Vine)

**Use when**: Sequential/ordered data (time series, spatial)
**Structure**: Path - (1,2), (2,3), (3,4), ...

```python
from vdc.vine.api import fit_dvine

# Automatic ordering (optimized)
vine = fit_dvine(U, diffusion_model)

# Custom ordering
vine = fit_dvine(U, diffusion_model, order=[0, 2, 4, 1, 3])
```

### C-Vine (Canonical Vine)

**Use when**: Key driver variables or hierarchical structure
**Structure**: Star - root connects to all others at each level

```python
from vdc.vine.api import fit_cvine

# Automatic root selection (optimized)
vine = fit_cvine(U, diffusion_model)

# Custom root ordering
vine = fit_cvine(U, diffusion_model, order=[2, 0, 1, 3, 4])
```

## Detailed Usage

### 1. Using the High-Level API

```python
from vdc.vine.api import VineCopulaModel
import torch
from vdc.models.unet_grid import GridUNet

# Load trained diffusion model
model = GridUNet(m=64)
model.load_state_dict(torch.load('checkpoints/best.pt')['model_state_dict'])
model.eval()

# Your data (pseudo-observations)
U = ...  # shape (n, d)

# ========== Option 1: R-Vine ==========
rvine = VineCopulaModel(vine_type='rvine')
rvine.fit(U, model)

# ========== Option 2: D-Vine ==========
# Automatic ordering
dvine_auto = VineCopulaModel(vine_type='dvine')
dvine_auto.fit(U, model)

# Custom ordering (e.g., time series: 0→1→2→3→4)
dvine_custom = VineCopulaModel(vine_type='dvine', order=[0, 1, 2, 3, 4])
dvine_custom.fit(U, model)

# ========== Option 3: C-Vine ==========
# Automatic root selection
cvine_auto = VineCopulaModel(vine_type='cvine')
cvine_auto.fit(U, model)

# Custom roots (e.g., variable 2 is main driver)
cvine_custom = VineCopulaModel(vine_type='cvine', order=[2, 0, 1, 3, 4])
cvine_custom.fit(U, model)
```

### 2. Fitting from Vine Matrix

```python
from vdc.vine.api import VineCopulaModel
import numpy as np

# Define vine matrix (d x d)
# Diagonal: variable order
# Off-diagonal: conditioning structure
vine_matrix = np.array([
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 0],
    [0, 0, 2, 3, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, 4],
])

# Fit from matrix
vine = VineCopulaModel(vine_type='dvine')  # or 'cvine', 'rvine'
vine.fit_from_matrix(U, vine_matrix, diffusion_model)
```

### 3. Inference and Sampling

All vine types support the same inference API:

```python
# Evaluate log-density
logpdf = vine.logpdf(U_test)
avg_loglik = np.mean(logpdf)

# Generate samples
U_samples = vine.simulate(n=1000, seed=42)

# Rosenblatt transform (copula → independent uniforms)
W = vine.rosenblatt(U)

# Inverse Rosenblatt (uniforms → copula)
U_recovered = vine.inverse_rosenblatt(W)

# Get vine structure as matrix
matrix = vine.get_structure_matrix()
```

### 4. Saving and Loading

```python
# Save fitted vine
vine.save('my_dvine.pkl')

# Load later
from vdc.vine.api import VineCopulaModel
vine = VineCopulaModel.load('my_dvine.pkl')

# Use immediately
logpdf = vine.logpdf(new_data)
```

## Vine Structure Details

### D-Vine Structure

Sequential coupling with neighbors:

```
Tree 1: (0,1)  (1,2)  (2,3)  (3,4)
Tree 2: (0,2|1)  (1,3|2)  (2,4|3)
Tree 3: (0,3|1,2)  (1,4|2,3)
Tree 4: (0,4|1,2,3)
```

**Ordering matters!** Different orders give different structures:
- `[0,1,2,3,4]`: Sequential
- `[0,2,4,1,3]`: Alternating
- Optimized: Chosen to maximize adjacent |τ|

### C-Vine Structure

Star structure with root at each level:

```
Tree 1 (root=0): (0,1)  (0,2)  (0,3)  (0,4)
Tree 2 (root=1): (1,2|0)  (1,3|0)  (1,4|0)
Tree 3 (root=2): (2,3|0,1)  (2,4|0,1)
Tree 4 (root=3): (3,4|0,1,2)
```

**Root order matters!** 
- First root should be the "main driver"
- Subsequent roots capture residual dependencies
- Optimized: Roots chosen by maximum sum of |τ|

### R-Vine Structure

Flexible structure selected by MST on |τ|:

```
Tree 1: MST on all pairwise |τ|
Tree 2: MST on conditional |τ| (proximity condition)
Tree 3: MST on 2nd-order conditional |τ|
...
```

**No user input needed** - structure is data-driven.

## When to Use Each Type

| Vine Type | Best For | Pros | Cons |
|-----------|----------|------|------|
| **R-Vine** | Unknown structure, complex dependencies | Flexible, data-driven | Slower to fit, harder to interpret |
| **D-Vine** | Sequential data (time series, spatial chains) | Fast, simple structure | Limited to sequential dependencies |
| **C-Vine** | Data with key drivers or hierarchy | Intuitive for factor models | Requires domain knowledge for root selection |

## Optimization Strategies

### Automatic D-Vine Ordering

The package uses a greedy algorithm:
1. Start with pair having maximum |τ|
2. Extend path by adding variable with strongest connection to endpoints
3. Repeat until all variables included

```python
# Automatic (recommended if no domain knowledge)
dvine = fit_dvine(U, model, order=None)
print(f"Optimized order: {dvine.structure.order}")
```

### Automatic C-Vine Root Selection

Greedy selection of roots:
1. Choose variable with maximum sum of |τ| as first root
2. For remaining levels, choose from unused variables
3. Select based on sum of |τ| with remaining variables

```python
# Automatic (recommended if no clear driver variable)
cvine = fit_cvine(U, model, order=None)
print(f"Root order: {cvine.structure.order}")
```

### Custom Ordering Tips

**For D-Vine:**
- Time series: Use temporal order `[0,1,2,3,...]`
- Spatial data: Use spatial proximity order
- Check pairwise correlations and arrange neighbors with high |τ|

**For C-Vine:**
- Choose main driver as first root (e.g., market index for stocks)
- Subsequent roots = secondary factors
- Use domain knowledge about causal relationships

## Performance Comparison

Run the comparison script:

```bash
python examples/compare_vines.py \
    --model checkpoints/best.pt \
    --n-train 1000 \
    --n-test 500 \
    --d 6
```

Expected results (depends on data):
- **R-Vine**: Highest log-likelihood, slowest fitting
- **D-Vine (optimized)**: Good balance of speed and accuracy
- **C-Vine (optimized)**: Fast, good if data has natural hierarchy

## Advanced: Truncation

All vine types support truncation (stop after k trees):

```python
# Full vine: d-1 trees
vine_full = fit_dvine(U, model, truncation_level=None)

# Truncated: only 3 trees
vine_truncated = fit_dvine(U, model, truncation_level=3)
```

**Benefits:**
- Faster fitting and inference
- Regularization (prevent overfitting)
- Sufficient for weak higher-order dependencies

**Rule of thumb:**
- Use first 3-5 trees for most applications
- Check if performance plateaus after k trees

## Integration with Other Libraries

### Export to pyvinecopulib format

```python
# Get vine matrix
matrix = vine.get_structure_matrix()

# Use with pyvinecopulib (if installed)
import pyvinecopulib as pv
# ... (interface code depends on pyvinecopulib API)
```

### Import from pyvinecopulib

```python
# If you have a vine matrix from pyvinecopulib
pv_matrix = ...  # from pyvinecopulib

# Build vine from it
vine = VineCopulaModel(vine_type='dvine')
vine.fit_from_matrix(U, pv_matrix, diffusion_model)
```

## Examples

See full examples in:
- `examples/compare_vines.py` - Compare all three types
- `examples/end_to_end.py` - Complete workflow with any vine type
- `examples/custom_structures.py` - Using custom orderings and matrices

## API Reference

### `VineCopulaModel`

Main class for all vine types.

**Constructor:**
```python
VineCopulaModel(
    vine_type='rvine',      # 'rvine', 'dvine', 'cvine'
    order=None,             # Variable ordering (list of ints)
    truncation_level=None,  # Stop after k trees
    m=64,                   # Grid resolution
    device='cuda'           # Device for inference
)
```

**Methods:**
- `fit(U, diffusion_model)` - Fit vine to data
- `fit_from_matrix(U, matrix, diffusion_model)` - Fit from vine matrix
- `logpdf(U)` - Evaluate log-density
- `pdf(U)` - Evaluate density
- `simulate(n, seed)` - Generate samples
- `rosenblatt(U)` - Forward transform
- `inverse_rosenblatt(W)` - Inverse transform
- `save(filepath)` - Save to file
- `load(filepath)` - Load from file (class method)
- `get_structure_matrix()` - Get vine matrix
- `summary()` - Get model info dict

### Convenience Functions

```python
from vdc.vine.api import fit_rvine, fit_dvine, fit_cvine

# Quick fitting
rvine = fit_rvine(U, model)
dvine = fit_dvine(U, model, order=[0,1,2,3,4])
cvine = fit_cvine(U, model, order=[2,0,1,3,4])
```

### Structure Building (Low-Level)

```python
from vdc.vine.vine_types import (
    build_dvine_structure,
    build_cvine_structure,
    build_vine_from_matrix
)

# Build structure only (no fitting)
structure = build_dvine_structure(U, order=None)
structure = build_cvine_structure(U, order=None)
structure = build_vine_from_matrix(U, matrix, vine_type='dvine')
```

## Troubleshooting

**Q: Which vine type should I use?**
A: Start with R-vine for exploration, then try D-vine/C-vine if you have domain knowledge or need speed.

**Q: My D-vine order doesn't match the data structure**
A: Try automatic ordering first, or visualize pairwise correlations to choose order.

**Q: C-vine performance is poor**
A: Root selection is critical. Try automatic selection or use domain knowledge.

**Q: Fitting is too slow**
A: Use truncation (`truncation_level=3`) or D-vine for speed.

**Q: How to choose truncation level?**
A: Fit full vine, evaluate test log-likelihood at each level, pick where it plateaus.

## References

- **D-vine & C-vine**: Kurowicka & Cooke (2006) - "Uncertainty Analysis with High Dimensional Dependence Modelling"
- **R-vine**: Bedford & Cooke (2002) - "Vines: A new graphical model for dependent random variables"
- **Structure selection**: Dißmann et al. (2013) - "Selecting and estimating regular vine copulae"

---

For more information, see the main README and API documentation.
