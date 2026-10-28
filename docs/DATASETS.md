# Datasets

Paths follow `vdc.data.paths.data_root()`: `$DATA_ROOT` if set, else `$OUTPUT_BASE/datasets`, else
`<repo>/data`. Loaders such as `vdc.data.tabular.maybe_load_uci(name, data_root)` read
`<data root>/uci/<name>.npz` (arrays `train` and `test`).

Install the optional readers first:

```bash
pip install -e ".[data]"   # pandas (GAS is a pandas pickle) and openpyxl (CCPP is an Excel workbook)
```

## UCI density benchmarks (MAF splits)

The paper's POWER, GAS, HEPMASS and MINIBOONE results use the preprocessed train/validation/test splits
of Papamakarios et al. (2017), *Masked Autoregressive Flow for Density Estimation*. They are rebuilt
from the authors' Zenodo archive (record 1161203, `data.tar.gz`, 856,664,546 bytes, CC BY 4.0) with
the preprocessing of [gpapamak/maf](https://github.com/gpapamak/maf) (`datasets/*.py`):

```bash
python scripts/download_uci_datasets.py                          # download and prepare all four
python scripts/download_uci_datasets.py --archive /path/data.tar.gz  # or use a local copy
export DATA_ROOT=<data root>/uci_maf                             # point the loaders at the MAF splits
```

The script checks the archive (sha256 `e56bef4c...7d353d3`) and each member it reads, extracts only
those members, and writes `<out>/uci/{power,gas,hepmass,miniboone}.npz` (float64 arrays `train`,
`validate`, `test`) and `<out>/uci/manifest.json`. The default `<out>` is `<data root>/uci_maf`; the
paper's runners use `train` and `test`.

| Dataset | Train | Validate | Test | d |
|---|---:|---:|---:|---:|
| POWER | 1,659,917 | 184,435 | 204,928 | 6 |
| GAS | 852,174 | 94,685 | 105,206 | 8 |
| HEPMASS | 315,123 | 35,013 | 174,987 | 21 |
| MINIBOONE | 29,556 | 3,284 | 3,648 | 43 |

Every output array is compared with the corresponding array of the paper's runs (SHA-256 of its bytes);
the script stops on a difference unless `--allow-hash-mismatch` is given. The recorded hashes were
reproduced bit for bit with NumPy 2.2.6 and pandas 2.3.3; other library versions could change the last
bits of the standardized values. Preparing all four takes about one minute and 1.2 GB of memory.

The GAS file in the archive is a pandas pickle. Unpickling runs code from the file, so the script reads
it only after its hash has matched the recorded one.

## Combined Cycle Power Plant (CCPP)

The Combined Cycle Power Plant data (UCI dataset 294; 9,568 rows, d=5) are a different dataset from MAF
POWER. Earlier versions of `scripts/download_uci_datasets.py` saved them as `uci/power.npz`; results
labelled "Power" that were computed from that file used CCPP. The paper's sampling-quality and
imputation experiments use CCPP and label it as such. Prepare it with:

```bash
python scripts/download_uci_datasets.py --ccpp                   # writes <data root>/uci/ccpp.npz
python scripts/download_uci_datasets.py --ccpp --ccpp-archive /path/CCPP.zip
```

The processing is unchanged from the earlier downloader: the workbook's active sheet of
`Folds5x2_pp.xlsx`, float32, and an 80/20 split with the legacy NumPy seed 42. The script checks the
workbook (sha256 `ccd49098...396f4651`; the paper used `CCPP.zip` with sha256 `cc7b2a49...8c7a`) and
compares the arrays with those of the earlier file. It never writes `power.npz`, and it refuses to
replace an existing output whose arrays differ unless `--force` is given.

## S&P100 returns

The S&P100 experiments read `<data root>/finance/sp100_returns.npy` (daily returns of 100 stocks,
shape `(T, 100)`). The price data are not redistributed; `paper/neurips2026/data/sp100_symbols.json`
lists the symbols and date range used in the paper.
