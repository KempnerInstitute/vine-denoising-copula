# Released Checkpoint MI Benchmark

This report records a direct run of `scripts/mi_estimation.py` using the official released checkpoint.

Command:

```bash
export PAPER_CHECKPOINT="$(cat analysis/PAPER_CHECKPOINT.txt)"
python scripts/mi_estimation.py \
  --estimator dcd \
  --checkpoint "${PAPER_CHECKPOINT}" \
  --device cpu \
  --n-samples 2000 \
  --seed 123 \
  --out-json docs/reports/pretrained_release/mi_benchmark_dcd_release.json
```

Summary:

- checkpoint: `vdc-denoiser-m64-v1`
- model type: `denoiser`
- grid size: `64`
- mean absolute MI error: `0.00731`
- mean runtime per case: `0.416 s`

Per-case results:

| Case | MI true | MI est | Absolute error | Time (s) |
|---|---:|---:|---:|---:|
| Independence | 0.0000 | 0.0002 | 0.0002 | 0.401 |
| Gaussian(ρ=0.7) | 0.3351 | 0.3394 | 0.0043 | 0.446 |
| Gaussian(ρ=-0.7) | 0.3351 | 0.3474 | 0.0123 | 0.394 |
| Student-t(ρ=0.7, df=5) | 0.3517 | 0.3582 | 0.0064 | 0.404 |
| Clayton(θ=3.0) | 0.6346 | 0.6258 | 0.0088 | 0.420 |
| Clayton(θ=3.0, rot=90) | 0.6346 | 0.6376 | 0.0031 | 0.472 |
| Gumbel(θ=2.5) | 0.5601 | 0.5586 | 0.0015 | 0.399 |
| Joe(θ=3.0) | 0.4752 | 0.4468 | 0.0285 | 0.406 |
| Frank(θ=5.0) | 0.2579 | 0.2572 | 0.0007 | 0.404 |

Source JSON:

- [mi_benchmark_dcd_release.json](mi_benchmark_dcd_release.json)
