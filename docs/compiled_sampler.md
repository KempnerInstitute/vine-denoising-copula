# Optional compiled D-vine sampling

Prepare an accelerated sampler from an already fitted model:

```python
sampler = fitted_model.prepare_sampler(
    backend="auto", num_threads=8,
    cache_dir="/path/to/writable/kernel/cache",
)
print(sampler.backend, sampler.fallback_reason)
samples = sampler.simulate(5000, seed=130)
```

Use no more threads than your CPU allocation. Preparation compiles a cached
C++17/OpenMP kernel and packs the existing h tables. Repeated calls reuse those
tables. `backend="auto"` uses NumPy when the compiler is unavailable, a build
or load fails, the build fails its numerical check, the vine is unsupported,
or packed memory exceeds the configured limit. The selected `backend` and
`fallback_reason` make this visible. `backend="compiled"` raises on these
failures; `backend="numpy"` skips compilation and packing. Existing model
`simulate` and `inverse_rosenblatt` calls retain their NumPy behavior.

The handle requires an unchanged fitted vine. Reprepare after refitting or
manually editing its tables. Saved models do not serialize compiled handles;
prepare a fresh handle after loading. The sampler preserves the original
seeded draws and global NumPy random-stream behavior. Call
`sampler.inverse_rosenblatt(uniforms)` to supply independent uniforms yourself.

## Build and identity tests

A GCC-compatible C++17 compiler with OpenMP is required for acceleration.
Run compilation and tests on a compute node. Install from the source release
or wheel; both contain the C++ source. The source distribution also includes
this document and the preservation tests. Nothing is downloaded during kernel
compilation. To build explicitly:

```python
from vdc.vine.compiled_sampler import build_kernel
build_kernel("/path/to/cache/sampler.so", compiler="g++")
sampler = fitted_model.prepare_sampler(kernel="/path/to/cache/sampler.so",
                                      backend="auto", num_threads=8)
```

Automatic builds use `cache_dir`, then `VDC_KERNEL_CACHE`, then
`~/.cache/vdc/sampler`. Cache keys include source, compiler and platform;
completed builds are published atomically. Each kernel undergoes a small
NumPy-identity check once per process before automatic use. This check does
not replace full qualification on a new platform. GCC 8.5.0 on the recorded
Linux node is the measured camera-ready configuration.

From an unpacked source release, run:

```bash
python -m pytest tests/test_compiled_sampler.py tests/test_prepared_sampler.py tests/test_hfunc_vectorized_inverse.py
```

The kernel uses float64 and disables fast math and multiply-add contraction.
Tests cover generated coordinates, edge orientations, truncation, boundaries,
flat CDF sections, thread counts, RNG behavior, save/load, failed builds and
loads, memory fallback and cache reuse. No estimator weights, density grids,
likelihoods or h-function equations are changed by selecting this backend.

## Supported models and memory

Acceleration supports legacy D-vines with default square cell-center grids,
non-spline `HFuncLookup` objects, and `hfunc_vectorized_inverse=True` during
fitting. It supports reversed edges and omitted independence edges in truncated
vines. Other vine types, experimental exact-density models, splines and
non-default grids use the fallback in automatic mode.

Packed tables require `d * (d - 1) / 2 * 2 * m * m * 8` additional bytes.
They are a copy of the fitted h tables: 64 KiB per edge at `m=64`, 309.375 MiB
at `d=100`, and about 7.6 GiB at `d=500`. A 128-grid uses four times as much.
The default `max_table_bytes=512*1024**2` prevents those large allocations and
falls back before compiling or packing. Set a larger limit explicitly only
when the allocation permits it; `None` disables the cap. Validation can use
additional temporary memory, so the cap is on packed storage, not total RSS.

The camera-ready sampling comparison uses 5,000 S&P100 draws at `d=100`:
0.72 s including packing versus 0.92 s for TLL, both with eight CPU threads.
Kernel compilation was a separate one-time 0.37 s. VDC fits a D-vine and TLL
fits a regular vine. This does not establish speed at `d=200/500` or for the
retrained exact-density models. Paper scripts and qualification receipts live
in `drafts/scripts/` and `drafts/rebuttal/`.
