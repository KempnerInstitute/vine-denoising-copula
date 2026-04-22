# Released Results and Checkpoint Verification

This repository contains the released checkpoint, the public benchmark scripts, and the verification reports used to validate the published VDC release.

What you can reproduce directly from this package repo:

- resolution and download of the released checkpoint
- checkpoint verification against analytic bivariate cases
- MI estimation with the released checkpoint
- MI self-consistency and TC benchmark reruns given an explicit checkpoint

The manuscript sources and paper-only artifact assembly are maintained in a separate paper workspace and are intentionally not part of this package release.

## Released Model

Current public model id:

`vdc-denoiser-m64-v1`

Resolve the checkpoint path:

```bash
python scripts/show_paper_checkpoint.py
```

Download the packaged release explicitly:

```bash
python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1
```

## Release Verification

Run the released-model verification suite:

```bash
python scripts/verify_pretrained_release.py \
  --model-id vdc-denoiser-m64-v1 \
  --device cpu \
  --out-dir docs/reports/pretrained_release
```

This regenerates:

- `PRETRAINED_RELEASE_VERIFICATION.md`
- `MI_BENCHMARK_DCD_RELEASE.md`
- the released qualitative figure assets and JSON summaries

## MI Estimation

To reproduce the DCD/VDC MI benchmark row with the released checkpoint:

```bash
export PAPER_CHECKPOINT="$(python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1)"

python scripts/mi_estimation.py \
  --estimator dcd \
  --checkpoint "${PAPER_CHECKPOINT}" \
  --n-samples 5000 \
  --seed 123 \
  --device cpu \
  --out-json results/mi_benchmark_dcd.json
```

## MI Self-Consistency

To rerun the DPI and additivity checks with the released checkpoint:

```bash
python scripts/mi_self_consistency_tests_v2.py \
  --checkpoint "${PAPER_CHECKPOINT}" \
  --device cpu \
  --json_output results/mi_self_consistency.json \
  --output results/tab_self_consistency.tex
```

## Total Correlation Benchmark

To rerun the TC benchmark with the released checkpoint:

```bash
python scripts/tc_benchmark.py \
  --checkpoint "${PAPER_CHECKPOINT}" \
  --device cpu \
  --dims 5 10 20 50 \
  --n 5000 \
  --n-trials 3 \
  --seed 42 \
  --out-json results/tc_benchmark.json
```

## Cluster Wrappers

Cluster launchers are documented in [slurm/PAPER_JOBS.md](../slurm/PAPER_JOBS.md). Those wrappers are convenience scripts for managed environments. For portable reproduction, prefer the Python commands above.
