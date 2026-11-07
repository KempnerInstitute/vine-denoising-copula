# NeurIPS 2026: Vine Denoising Copula

**Amortized Estimation of Simplified Vine Copulas for High-Dimensional Density and Information Analysis**

This project reproduces selected experiments with the frozen paper estimator.
The reusable implementation is the `vdc` package in this repository. No access
to the authors' manuscript repository, home directory or cluster is required.

## Start here

Use the versioned `v0.2.6` checkout or its source archive, Python 3.10 and the
pinned numerical environment:

```bash
export VDC_LIBRARY_ROOT="$PWD"
python -m pip install -c projects/neurips2026/runners/constraints.txt -e ".[data]"
python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1
python projects/neurips2026/run.py runners/reproduce_table1_likelihood.py --help
```

The model is hosted at [Hugging Face](https://huggingface.co/hsafaai/vdc-denoiser-m64-v1).
The expected checkpoint SHA-256 is
`8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f`.
Use the downloaded checkpoint path explicitly in each command.

## Experiments and expected results

[Runner instructions](runners/README.md) give commands, settings, expected
values, hardware and limitations for Table 1 likelihoods, UCI, chronological
S&P100, fitting speed, MI/TC, self-consistency and projection timing.
[Reproducibility scope](REPRODUCIBILITY.md) distinguishes historical generation
from release replay and identifies unavailable inputs.

| Result | Entry point |
|---|---|
| Table 1 likelihoods, nine cases and five evaluation seeds | `runners/reproduce_table1_likelihood.py` |
| UCI at 5k/50k training rows | `runners/run_uci_maf.py` |
| Chronological S&P100 likelihood | `runners/reproduce_sp100_likelihood.py` |
| Fixed-thread fitting speed, d=100/200/500 | `runners/reproduce_scaling_timings.py` |
| Sampling quality on CCPP and S&P100 | `runners/sampling_quality.py` |
| Bivariate MI, TC and self-consistency | `experiments/` |
| Paired 50-dataset TC structural control | `runners/tc_replication_paired.py` |

Train the model architecture through `scripts/train_unified.py`; see
[`docs/CONFIGURATION.md`](../../docs/CONFIGURATION.md). The historical checkpoint's
exact training source is only partially pinned, so current training is not a
bitwise reconstruction of that run. Use the released checkpoint for evaluation.

## Figures and tables

Download the [figure reproduction archive](https://github.com/KempnerInstitute/vine-denoising-copula/releases/download/v0.2.6/vdc-neurips2026-figures-v0.2.6.zip) from the `v0.2.6` release.
It includes all ten figure generators, generators for 25 tables, compact saved
inputs with hashes, reference tables and an inventory of the remaining tables.
It formats saved experimental outputs; it does not rerun model training or the
full benchmark suite. The archive has its own README and environment file.

The archive is built from the private manuscript's authoritative figure sources.
No manuscript history, internal notes, personal storage paths, scheduler accounts
or historical cluster launchers are needed or included. The figure sources are
not installed in the VDC Python package.

## Data and license

The library's [dataset downloader](../../docs/DATASETS.md) obtains public UCI
benchmarks; the paper's sampling benchmark uses CCPP, not MAF POWER. S&P100
symbols and collection metadata are in `data/sp100_symbols.json`; the exact
historical returns matrix is unavailable for redistribution. See the runner
instructions before using a rebuilt matrix.

Code is covered by the repository's [MIT license](../../LICENSE). External data
and baseline software retain their own licenses. Source and checkpoint hashes
are recorded separately from runtime and hardware measurements.
