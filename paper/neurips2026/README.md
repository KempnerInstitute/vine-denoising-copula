# Reproducing the NeurIPS 2026 paper

This directory accompanies *Amortized Estimation of Simplified Vine Copulas for High-Dimensional
Density and Information Analysis* (Houman Safaai, NeurIPS 2026). It contains what is needed to check
the paper's figures and tables against saved outputs and to rerun its main experiments with the
released model.

| Folder | Contents |
|---|---|
| [`figures/`](figures/README.md) | Scripts and saved inputs that regenerate all nine figures (Figures 1-3 and A1-A6); the regenerated images are identical to the paper's. |
| [`tables/`](tables/README.md) | Generators that rebuild 23 of the paper's tables from saved outputs, byte for byte, and an inventory of how every other table was produced. |
| [`runners/`](runners/README.md) | Portable runners for the main experiments (Table 1 likelihoods, UCI benchmarks on the MAF splits, chronological S&P100, vine-fitting timings) and the library commands for the MI, TC and self-consistency benchmarks. |
| `data/sp100_symbols.json` | Symbols, column order, source and date range of the S&P100 returns matrix (prices are not redistributed). |
| `tools/sanitize_inputs.py` | How saved outputs were copied here: absolute paths and host names removed, numbers unchanged; each `MANIFEST.json` maps a copy to the sha256 of its original. |

## Requirements

- The `vdc` package of this repository (version 0.2.0): `pip install -e .` from the repository root;
  `runners/constraints.txt` pins the numerical packages of the environment used for the paper.
- The figure and table scripts need only NumPy, SciPy and Matplotlib; they read saved outputs and
  run no estimator. The figures were rendered with Python 3.10, NumPy 2.2.6, SciPy 1.15.3 and
  Matplotlib 3.10.6.
- The released model `hsafaai/vdc-denoiser-m64-v1` (checkpoint sha256
  `8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f`):
  `python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1`.

## Data

- UCI POWER, GAS, HEPMASS and MINIBOONE: `python scripts/download_uci_datasets.py` rebuilds the MAF
  train/validation/test splits from the authors' archive (Zenodo record 1161203) and checks every
  array against the paper's files. See [docs/DATASETS.md](../../docs/DATASETS.md).
- Combined Cycle Power Plant (CCPP), used by the sampling and imputation experiments:
  `python scripts/download_uci_datasets.py --ccpp`. Earlier versions of the downloader saved these
  data as `power.npz`; they are not the MAF POWER benchmark.
- S&P100: daily log-returns of Stooq close prices for the symbols in `data/sp100_symbols.json`. The
  returns matrix is not redistributed; the S&P100 runner checks the paper's matrix by hash.

## Quick checks

    bash paper/neurips2026/figures/make_figures.sh /path/to/new-figures
    bash paper/neurips2026/tables/make_tables.sh /path/to/new-tables

Each runs in a few minutes or less on a CPU and needs no GPU or checkpoint.

## Scope

- Results use the released model. The exploratory retrained estimators of the paper's Appendix G
  are not released, so their two tables are listed but not reproducible here.
- Timings depend on hardware; the runners document the paper's hardware and settings.
- Some appendix experiments are run by scripts that remain in the authors' paper workspace; the
  inventory in `tables/README.md` lists them, and their saved outputs are included where a table is
  rebuilt from them.
