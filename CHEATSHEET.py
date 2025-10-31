"""
Vine Copula Quick Reference & Cheat Sheet
"""

# ============================================================================
# BASIC USAGE
# ============================================================================

from vdc.vine.api import fit_rvine, fit_dvine, fit_cvine, VineCopulaModel
from vdc.models.unet_grid import GridUNet
import torch
import numpy as np

# Load trained diffusion model
model = GridUNet(m=64)
model.load_state_dict(torch.load('checkpoints/best.pt')['model_state_dict'])
model.eval()

# Your data (convert to pseudo-observations)
U = ...  # shape (n, d), values in (0, 1)

# ============================================================================
# FITTING VINES
# ============================================================================

# R-VINE (automatic structure)
rvine = fit_rvine(U, model)

# D-VINE (optimized ordering)
dvine_auto = fit_dvine(U, model)

# D-VINE (custom ordering)
dvine_custom = fit_dvine(U, model, order=[0, 2, 4, 1, 3])

# C-VINE (optimized roots)
cvine_auto = fit_cvine(U, model)

# C-VINE (custom roots - first is main driver)
cvine_custom = fit_cvine(U, model, order=[2, 0, 1, 3, 4])

# TRUNCATED VINE (only first k trees)
vine_truncated = fit_dvine(U, model, truncation_level=3)

# ============================================================================
# USING VINE COPULAMODEL CLASS
# ============================================================================

# Initialize
vine = VineCopulaModel(
    vine_type='dvine',      # 'rvine', 'dvine', or 'cvine'
    order=[0, 1, 2, 3, 4],  # None for automatic
    truncation_level=None,  # None for full vine
    m=64,                   # Grid resolution
    device='cuda'           # 'cuda' or 'cpu'
)

# Fit to data
vine.fit(U, model, verbose=True)

# Or fit from vine matrix
vine_matrix = np.array([...])  # (d, d) matrix
vine.fit_from_matrix(U, vine_matrix, model)

# ============================================================================
# INFERENCE
# ============================================================================

# Log-density
logpdf = vine.logpdf(U_test)  # Returns (n,) array
avg_loglik = np.mean(logpdf)

# Density (not log)
pdf = vine.pdf(U_test)

# Generate samples
U_samples = vine.simulate(n=1000, seed=42)

# Rosenblatt transform (copula → independent uniforms)
W = vine.rosenblatt(U)

# Inverse Rosenblatt (uniforms → copula)
U_recovered = vine.inverse_rosenblatt(W)

# ============================================================================
# SAVE & LOAD
# ============================================================================

# Save fitted vine
vine.save('my_vine.pkl')

# Load later
loaded_vine = VineCopulaModel.load('my_vine.pkl')

# Get info
print(vine)
print(vine.summary())

# ============================================================================
# STRUCTURE INSPECTION
# ============================================================================

# Get structure
structure = vine.structure

# Number of trees and edges
n_trees = len(structure.trees)
n_edges = structure.num_edges()

# Variable order
order = structure.order

# Get as matrix
matrix = vine.get_structure_matrix()

# Print details
for tree_level, tree in enumerate(structure.trees):
    print(f"Tree {tree_level + 1}:")
    for edge, tau in zip(tree.edges, tree.tau_values):
        i, j, cond = edge
        print(f"  ({i},{j}|{cond}) τ={tau:.3f}")

# ============================================================================
# COMPARING VINE TYPES
# ============================================================================

# Fit all three types
rvine = fit_rvine(U_train, model)
dvine = fit_dvine(U_train, model)
cvine = fit_cvine(U_train, model)

# Evaluate on test data
rvine_loglik = np.mean(rvine.logpdf(U_test))
dvine_loglik = np.mean(dvine.logpdf(U_test))
cvine_loglik = np.mean(cvine.logpdf(U_test))

print(f"R-vine: {rvine_loglik:.4f}")
print(f"D-vine: {dvine_loglik:.4f}")
print(f"C-vine: {cvine_loglik:.4f}")

# ============================================================================
# COMMON PATTERNS
# ============================================================================

# TIME SERIES: Use D-vine with temporal ordering
U_timeseries = ...  # (n, T) where T = time steps
dvine = fit_dvine(U_timeseries, model, order=list(range(T)))

# FACTOR MODEL: Use C-vine with factor as root
# order = [factor_idx, asset1, asset2, ...]
U_factors = ...  # (n, d)
cvine = fit_cvine(U_factors, model, order=[0, 1, 2, 3, 4])

# UNKNOWN STRUCTURE: Use R-vine
U_unknown = ...
rvine = fit_rvine(U_unknown, model)

# FAST FITTING: Use truncation
vine_fast = fit_dvine(U, model, truncation_level=3)

# ============================================================================
# CONVERTING PSEUDO-OBSERVATIONS
# ============================================================================

# From raw data to pseudo-observations
X = ...  # (n, d) raw data
n, d = X.shape

U = np.zeros_like(X)
for j in range(d):
    # Rank transformation
    U[:, j] = (np.argsort(np.argsort(X[:, j])) + 1) / (n + 1)

# Now U is in (0, 1) and ready for vine fitting

# ============================================================================
# ERROR HANDLING
# ============================================================================

try:
    vine = fit_dvine(U, model)
    logpdf = vine.logpdf(U_test)
except RuntimeError as e:
    print(f"Fitting failed: {e}")

# Check if fitted
if vine.fitted:
    logpdf = vine.logpdf(U_test)
else:
    print("Vine not fitted yet")

# ============================================================================
# ADVANCED: LOW-LEVEL STRUCTURE BUILDING
# ============================================================================

from vdc.vine.vine_types import (
    build_dvine_structure,
    build_cvine_structure,
    build_vine_from_matrix
)

# Build structure without fitting
structure = build_dvine_structure(U, order=None)  # Auto-optimize

# Inspect structure before fitting
print(f"D-vine will have {len(structure.trees)} trees")
print(f"First tree: {len(structure.trees[0].edges)} edges")

# Then fit manually (advanced)
from vdc.vine.recursion import VineRecursion
vine_recursion = VineRecursion(structure)
# ... (manually add pair copulas)

# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

# 1. Use truncation for speed
vine = fit_dvine(U, model, truncation_level=3)  # 2x faster

# 2. Lower grid resolution for large d
vine = fit_dvine(U, model, m=32)  # Faster but less accurate

# 3. Use CPU if GPU memory is limited
vine = VineCopulaModel(vine_type='dvine', device='cpu')

# 4. Batch evaluation
U_test_batches = np.array_split(U_test, 10)
logpdf = np.concatenate([vine.logpdf(batch) for batch in U_test_batches])

# ============================================================================
# EXAMPLES
# ============================================================================

# Run comparison script
# python examples/compare_vines.py --model checkpoints/best.pt --d 5

# Run end-to-end demo
# python examples/end_to_end.py --n_train 1000 --d 5

# ============================================================================
# WHEN TO USE WHICH VINE
# ============================================================================

"""
R-VINE:
  ✓ Unknown dependency structure
  ✓ Complex dependencies
  ✓ Exploratory analysis
  ✗ Slower fitting
  ✗ Harder to interpret

D-VINE:
  ✓ Sequential/ordered data
  ✓ Time series
  ✓ Spatial chains
  ✓ Fast fitting
  ✗ Limited to path structure

C-VINE:
  ✓ Key driver variables
  ✓ Factor models
  ✓ Hierarchical data
  ✓ Fast fitting
  ✗ Requires domain knowledge
"""

# ============================================================================
# DOCUMENTATION
# ============================================================================

# Full guides:
# - README.md: Overview and quickstart
# - VINE_TYPES.md: Detailed vine type documentation
# - IMPLEMENTATION_COMPLETE.md: Implementation details
# - examples/compare_vines.py: Runnable comparison

# API documentation:
help(VineCopulaModel)
help(fit_dvine)
help(fit_cvine)
help(fit_rvine)
