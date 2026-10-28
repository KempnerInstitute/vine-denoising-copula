# Optional faster fitting

Pass `fast_fit=True` when constructing `VineCopulaModel` to reuse D-vine
ordering correlations, vectorize h-function table construction, and transfer
predicted grids in batches. The default remains `False`. This option changes
execution, not the trained estimator, projection count, or network batch size.

```python
from vdc import VineCopulaModel

vine = VineCopulaModel(
    vine_type="dvine", m=64, device="cuda", projection_iters=15,
    hfunc_use_spline=False, hfunc_vectorized_inverse=True,
    batch_edges=True, edge_batch_size=256, fast_fit=True,
)
vine.fit(training_uniforms, diffusion_model=pretrained_model, verbose=False)
```

Supply data already transformed to uniform marginals, and a loaded original
VDC checkpoint. The paper companion gives the checkpoint hash, loading recipe,
precision settings and a complete timing runner. Edges within a tree are
processed in batches of up to 256 in this configuration. Successive trees still
depend on the conditional inputs produced by earlier trees.

On the paper's Clayton benchmark at n=20,000 and d=100/200/500, the option
reduced VDC fit time by 34–35%. All nine dimension/data-seed fits preserved
the fitted state and evaluated likelihoods, PITs and samples bitwise against
the original execution. This qualification used float32, matrix TF32 disabled,
convolution TF32 enabled, one H100 and 16 CPU threads. It is not a guarantee
of identical floating-point results across hardware or numerical environments.

The resulting VDC/TLL deployment comparison is 3.3–4.0x faster than 16-thread
CPU TLL, or 6.3–6.7x against eight threads. VDC fits a greedy D-vine; TLL fits
a selected regular vine. Timings include structure selection and fitting,
exclude model loading, warmup and pretraining, and were measured sequentially
on one node. The speedup ratio decreases slightly with dimension while the
absolute time saved increases. This is a GPU-versus-CPU wall-time comparison.

From an unpacked source release, run on a compute node with four CPU cores:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
python -m pytest tests/test_fast_fit_identity.py \
  tests/test_compiled_sampler.py tests/test_prepared_sampler.py \
  tests/test_hfunc_vectorized_inverse.py \
  tests/test_vine_structure_robustness.py tests/test_vine_recursion.py
```

These tests cover exact structure/table/score comparisons, saved options,
legacy loading, and the compiled sampler and its NumPy fallback. They use
small CPU examples; the paper's full checkpoint and high-dimensional checks
are separate. See [compiled_sampler.md](compiled_sampler.md) for compilation
instructions, supported models and packing-memory limits.
