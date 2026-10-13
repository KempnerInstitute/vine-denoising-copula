# Optional compiled D-vine sampling prototype

`vdc.vine.compiled_sampler` executes an already-fitted legacy D-vine in C++
with OpenMP threads. It copies the existing h-function tables and leaves the
fitted model, likelihood, neural network and random-number generation alone.
The default library sampling path is unchanged.

Use a source checkout containing the prototype and a GCC-compatible C++17
compiler with OpenMP support. Build on a compute node, into a writable cache
outside the source checkout:

```python
from vdc.vine.compiled_sampler import build_kernel, CompiledDVineSampler
import numpy as np

kernel = "/path/to/compute/cache/vdc_sampler.so"
command = build_kernel(kernel)  # compile once; record the returned command
sampler = CompiledDVineSampler(fitted_model.vine, kernel)

# Match VineRecursion.simulate(n, seed)'s legacy NumPy random stream.
uniforms = np.random.RandomState(130).uniform(0, 1, (5000, sampler.d))
generated = sampler.inverse_rosenblatt(uniforms, threads=8, block_size=64)
```

The wrapper takes a snapshot of the fitted tables. Reconstruct it after
refitting or modifying a vine. Pass no more threads than the CPU allocation.
Constructing the snapshot and compiling are distinct costs from repeated
sampling. Record them when reporting latency for a first use.

The prototype supports the original `VineRecursion` D-vines with default
cell-center square grids and non-spline `HFuncLookup` objects that enable
`vectorized_inverse=True`. It supports reversed edge orientations and omitted
independence edges in truncated vines. It rejects experimental exact-density
classes, other vine types, splines and non-default grids. It does not speed
up the retrained exact-density estimators by implication.

The kernel uses double precision, preserves interpolation operation order,
and is compiled without fast math or multiply-add contraction. It generates
no random numbers and performs no reductions between samples. Exact equality
to the corrected NumPy sampler is nevertheless an explicit qualification
requirement on each supported build, rather than assumed from parallelism.

Tests (run on a compute node):

```bash
PYTHONPATH=/path/to/source python -m pytest tests/test_compiled_sampler.py tests/test_hfunc_vectorized_inverse.py
```

The camera-ready qualification script and evidence receipt belong to the paper
repository (`drafts/scripts/qualify_compiled_sampler.py` and
`drafts/rebuttal/compiled_sampler_qualification_20261013.md`). They compare the
original checkpoint on the publication sampling protocol, with seeded RNG
included in repeated timings and setup costs reported separately. This remains
an isolated prototype until it is integrated into a release. Packaging and
cross-platform compiler support have not been qualified by this benchmark.
