# Cluster Job Wrappers for Released Benchmarks

This directory contains SLURM wrappers that were used during research and release validation. They are convenience scripts for managed cluster environments rather than part of the stable package API.

For portable public reproduction, prefer the Python commands in [docs/PAPER_REPRODUCIBILITY.md](../docs/PAPER_REPRODUCIBILITY.md).

## Output Layout

Most training and benchmark jobs write into an `OUTPUT_BASE` directory with run folders of the form:

```text
<OUTPUT_BASE>/
  vdc_paper_<method>_<timestamp>_<jobid>/
    results/
    logs/
    checkpoints/
    analysis/
```

Set `OUTPUT_BASE` explicitly before submission. If you do not set it, many scripts fall back to a repo-local `results/` directory or to the script's own defaults.

Example:

```bash
OUTPUT_BASE="$PWD/results" sbatch slurm/paper_train_enhanced.sh
```

## Recommended Public Jobs

These wrappers are the most useful ones to keep in sync with the public repo:

- `paper_train_enhanced.sh`
- `paper_mi_benchmark.sh`
- `paper_tc_benchmark.sh`
- `paper_mi_consistency.sh`

Typical usage:

```bash
OUTPUT_BASE="$PWD/results" sbatch slurm/paper_train_enhanced.sh

PAPER_CHECKPOINT=/path/to/checkpoint.pt \
OUTPUT_BASE="$PWD/results" \
sbatch slurm/paper_mi_benchmark.sh

PAPER_CHECKPOINT=/path/to/checkpoint.pt \
OUTPUT_BASE="$PWD/results" \
sbatch slurm/paper_tc_benchmark.sh
```

## Paper-Specific Jobs

Several `paper_*` scripts were written to support manuscript assembly in a separate paper workspace. They are retained for completeness, but they are not required for package use and should be treated as research automation rather than supported public interfaces.
