# Reproducing the NeurIPS 2026 paper

Code and saved results for *Amortized Vine Copulas for High-Dimensional Density and
Information Estimation* (Houman Safaai, NeurIPS 2026,
[arXiv:2604.20568](https://arxiv.org/abs/2604.20568)).

This folder holds the paper's reproduction material. It imports the `vdc` package of this
repository, or another checkout selected with `run.py --library`, and is not installed with
the package. No access to the authors' manuscript repository, storage or cluster is needed.
Reproduction works at two levels:

- **Replay from saved outputs** (CPU, under a minute, no model or data download):
  `figures/make_figures.sh` and `tables/make_tables.sh` regenerate all ten figures and the
  25 generated tables from compact saved experiment outputs. Nothing is refitted.
- **Rerun of experiments** (GPU, public data): the runners in `runners/` and `experiments/`
  recompute the experiments with the released checkpoint and, where the protocol allows,
  check the result against the paper's records.

The [inventory](#inventory) maps every figure and table of the paper to its generator, saved
inputs and runner. [Scope and limitations](#scope-and-limitations) lists what cannot be rerun
from public material.

## Contents

| Path | Contents |
|---|---|
| `figures/` | `make_figures.sh`, figure scripts (`scripts/`), saved inputs with hashes (`inputs/`); [input provenance](figures/README.md) |
| `tables/` | `make_tables.sh`, 13 table generators (`scripts/`), saved inputs with hashes (`inputs/`); [input provenance and table notes](tables/README.md) |
| `runners/` | portable experiment runners; `saved_outputs/` of the TC control runs; [paired TC replication protocol](runners/tc_replication_PROTOCOL.md) |
| `experiments/` | MI, TC and MI self-consistency benchmarks |
| `reference/` | saved records that the runners compare a rerun with; `MANIFEST.json` gives their hashes |
| `configs/` | training configurations of the corruption and probit ablations |
| `data/sp100_symbols.json` | S&P100 tickers, column order, source and date range |
| `run.py` | runs a script of this folder with a chosen VDC checkout |
| `requirements-figures.txt` | pinned environment for figure and table replay |
| `constraints-experiments.txt` | pinned numerical environment for the runners |

## Setup

Use Python 3.10. Replay needs only the pinned plotting environment (NumPy 2.2.6, SciPy 1.15.3,
matplotlib 3.10.6, Pillow 11.3.0, pandas 2.3.3):

```bash
python -m pip install -r paper/requirements-figures.txt
```

Experiments need the library. From the repository root, install it with the numerical packages
in which the paper's replays were qualified (torch 2.9.1, NumPy 2.2.6, SciPy 1.15.3,
pyvinecopulib 0.7.5, ...); NumPy and BLAS versions can change the last digits of reported
summaries:

```bash
export VDC_LIBRARY_ROOT="$PWD"
python -m pip install -c paper/constraints-experiments.txt -e ".[data]"
python -m pip check
```

**Released model.** Apart from the checkpoint comparisons of the corruption, probit and
retrained-estimator appendices, the paper uses one frozen checkpoint, `vdc-denoiser-m64-v1`
(training checkpoint `model_step_190000.pt`), hosted on
[Hugging Face](https://huggingface.co/hsafaai/vdc-denoiser-m64-v1):

```bash
python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1   # prints the local path
export CKPT=/path/to/vdc-denoiser-m64-v1.pt
sha256sum "$CKPT"   # 8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f
```

Every runner in `runners/` that loads the model checks this hash (`run_uci_maf.py` accepts
another model only with `--expected-checkpoint-sha256 any`); the benchmarks in `experiments/`
load the checkpoint they are given, so check its hash first. The runners that take
`--expected-module-root` stop unless `vdc` is imported from that directory (the repository root
for an editable install, `site-packages` for a wheel).

**Public data** ([dataset guide](../docs/DATASETS.md)):

```bash
python scripts/download_uci_datasets.py --out /path/to/data              # MAF splits: POWER, GAS, HEPMASS, MINIBOONE
python scripts/download_uci_datasets.py --ccpp --ccpp-out /path/to/data  # Combined Cycle Power Plant (CCPP)
```

These write `/path/to/data/uci/{power,gas,hepmass,miniboone,ccpp}.npz` and check every array
against the arrays of the paper's runs. The MAF archive is about an 860 MB download; GAS and
HEPMASS need pandas and CCPP needs openpyxl (both in the `data` extra). The paper's sampling
benchmark uses CCPP, not MAF POWER. The S&P100 returns matrix is not redistributed.

**Launcher.** `run.py` puts the selected checkout first on the Python path, sets
`VDC_LIBRARY_ROOT`, checks that `vdc` is imported from that checkout and passes all other
arguments unchanged:

```bash
python paper/run.py [--library CHECKOUT] SCRIPT [ARGUMENTS ...]   # SCRIPT relative to paper/; default CHECKOUT: this repository
```

Each runner writes only to the output path it is given (use a directory that does not exist yet
where it asks for one) and records the checkpoint hash and software versions.

## Regenerate figures and tables

```bash
bash paper/figures/make_figures.sh output/figures   # ten figures, vector PDF and 400-dpi PNG at printed size
bash paper/tables/make_tables.sh output/tables      # 25 LaTeX tables
```

`PYTHON=/path/to/python` selects the interpreter. On eight CPU cores the figures take about 40 s
and the tables about 6 s. The generators format saved numbers; they run no estimator. Each
`inputs/MANIFEST.json` gives every input's sha256 and the sha256 of the experiment output it was
copied from, and the figure scripts refuse an input whose hash differs. Copies differ from the
original outputs only in metadata (storage paths cut to file names; host, job and reviewer
identifiers removed) unless the folder notes say otherwise ([figures](figures/README.md),
[tables](tables/README.md)).

With the pinned environment, all ten figures are pixel-identical to the paper's (PNG and 300-dpi
rasterizations of the PDFs) and all 25 tables are identical to the paper's apart from LaTeX
comment lines; other matplotlib or font versions may move text by a pixel.
`make_tables.sh OUT REFERENCE_DIR` also compares the output with a directory of reference tables,
ignoring comment lines. These generators and inputs are the contents of the figure archive
attached to release v0.2.6 (`vdc-neurips2026-figures-v0.2.6.zip`), with paths updated; that
archive also holds the paper's tables as `reference_tables/`.

## Inventory

Replay columns name the generator and its inputs, relative to the folder's `scripts/` and
`inputs/`. Rerun links to the runner that recomputes the experiment; its resources are in
[experiment runners](#experiment-runners). A dash means that no public runner exists.

### Figures

| Figure | Subject | Script | Saved inputs | Rerun |
|---|---|---|---|---|
| 1 | VDC overview | `fig1_overview_cr.py` | `fig1_overview/fig1_overview_data.npz` | model outputs: `scripts/fig1_overview_data.py` (CPU, [figures](figures/README.md)) |
| 2 | Density and vine-fitting cost | `fig2_density_runtime_cr.py` | (a) `tables/tab_bivariate.tex`; (b) `paper_outputs/e22_highdim_scaling.json`; (c) `fast_fit_scaling/timing_summary.json`; (d) `paper_outputs/e23_sp100_density.json`, `paper_outputs/e28_sp100_chrono.json` | (c) [fitting time](#fitting-time); (d) [S&P100](#chronological-sp100), chronological split, VDC and TLL |
| 3 | Information-estimation results | `fig3_information_cr.py` | (a) `tables/tab_mi.tex`; (b) `tc_scaling/tc_benchmark.json`; (c) `paper_outputs/mi_self_consistency_scaleup.json` | (a) [MI](#bivariate-mi); (b) [TC](#tc-versus-dimension); (c) [self-consistency](#mi-self-consistency) |
| 4 | TC decomposition | `fig4_tc_decomposition_cr.py` | `paper_outputs/tc_decomposition_ground_truth_compare.json`, `tc_knockout/*.json` | — |
| A1 | Training and projection diagnostics | `figA_training_diagnostics_cr.py` | `fig1_overview/fig1_overview_data.npz` | projection panel as Figure 1; the training log is not distributed |
| A2 | Non-simplified stress tests | `figA_nonsimplified_stress_cr.py` | `paper_outputs/e18_nonsimplified_stress.json`, checked against `tables/tab_nonsimplified_stress.tex` | — |
| A3 | MI and TC versus dimension | `figA_tc_story_cr.py` | (a) `paper_outputs/e6_theory_synthetic_results.json`; (b) `tc_scaling/tc_benchmark.json` | (b) [TC](#tc-versus-dimension) |
| A4 | Fixed-checkpoint MI sensitivity | `figA_mi_sample_size_cr.py` | `paper_outputs/e12_block_mi_sample_size_clayton_chain_results_with_neural_seed42.json` | — |
| A5 | VaR calibration on S&P100 | `figA_var_calibration_cr.py` | `var_backtest/e3_var_results.json` | — (returns matrix) |
| A6 | Rolling S&P100 dependence | `figA_rolling_dependence_cr.py` | `paper_outputs/e16_rolling_dependence_compact.json`, `sp100/sp100_returns_meta.json` | — (returns matrix) |

### Tables

"Full run" tables were written at the end of a complete experiment run in the authors'
workspace and have no generator here; [tables/README.md](tables/README.md) records the edits
made to them by hand.

| Table | Subject | Replay: generator and inputs | Rerun |
|---|---|---|---|
| 1 | Bivariate estimation, pretraining families | `table1.py`: `table1/table1_v2.json` | [Table 1](#table-1-likelihoods): held-out KL of VDC, TLL and parametric selection |
| A1 | Claims–evidence map | written in the manuscript; no data | — |
| A2 | Benchmark specification | written in the manuscript from the benchmark code defaults | — |
| A3 | Retained corruption variants | `corruption.py`: `corruption/corruption_variants_results.json` | [corruption variants](#training-corruption-variants); alternative checkpoints not released |
| A4 | Common real-data edge panel | `corruption.py`: as A3 | as A3 |
| A5 | UCI benchmark, 5,000 training rows | `uci.py`: `uci/e21_maf5k.json` | [UCI](#uci-benchmarks), `--max-train 5000` |
| A6 | UCI benchmark, 50,000 training rows | `uci.py`: `uci/e21_maf50k.json` | [UCI](#uci-benchmarks), `--max-train 50000` |
| A7 | Structure-matched UCI comparison | `uci.py`: `uci/matched_structure_5k.json` | — |
| A8 | UCI copula dependence diagnostic | full run | — |
| A9 | RealNVP capacity sensitivity | full run | — |
| A10 | Non-simplified conditional stress tests | full run; saved output in `figures/inputs/paper_outputs/e18_nonsimplified_stress.json` | — |
| A11 | Bivariate MI estimation | `mi.py`: `mi/results/` | [MI](#bivariate-mi) |
| A12 | MI self-consistency, 60 trials | `self_consistency.py`: `self_consistency/mi_self_consistency.json` | [self-consistency](#mi-self-consistency), `--n_trials 5` |
| A13 | Scaled-up self-consistency, 480 trials | `self_consistency.py`: `self_consistency/mi_self_consistency_scaleup.json` | [self-consistency](#mi-self-consistency), `--n_trials 40` |
| A14 | DPI tolerance sensitivity | `self_consistency.py`: as A13 | as A13 |
| A15 | Additivity, block versus learned structure | `self_consistency.py`: `self_consistency/e35_additivity_learned.json` | — |
| A16 | Paired replication study | `replicates.py`: `bivariate/e31_bivariate_replicates.json` | — |
| A17 | Bivariate benchmark dispersion | `table1.py`: `table1/table1_v2.json` | [Table 1](#table-1-likelihoods), KL columns |
| A18 | High-dimensional Clayton-vine sweep | `scaling.py`: `scaling/e22_highdim_scaling_seed_merged.json` | — |
| A19 | Fixed-thread deployment timings | `scaling.py`: `scaling/fast_fit_timing_rows.csv` | [fitting time](#fitting-time) |
| A20 | pyvine-TLL thread sweep | `threads.py`: `threads/*.json`, `scaling/e22_highdim_scaling_seed_merged.json` | — |
| A21 | Matched-resource timing | full run | — |
| A22 | Mixed-family D-vine scaling | full run | — |
| A23 | S&P100 density, random split | full run; saved output in `figures/inputs/paper_outputs/e23_sp100_density.json` | — (returns matrix) |
| A24 | S&P100 density, chronological split | full run; saved output in `figures/inputs/paper_outputs/e28_sp100_chrono.json` | [S&P100](#chronological-sp100): VDC and TLL rows (returns matrix) |
| A25 | Raw-space S&P100 composition | `raw_composition.py`: `sp100/e37_raw_composition.json` | — (returns matrix) |
| A26 | Hard synthetic bivariate benchmark | full run | — |
| A27 | Out-of-distribution families | `ood.py`: `ood/e26_rescored.json` | partial: the [corruption runner](#training-corruption-variants) re-scores the released model's NLL and ISE on these cases and checks them against this record |
| A28 | Representation floor of the grid | written in the manuscript; values in `grid_floor/grid_floor_table.json` | — |
| A29 | Neural pair-copula comparison | `neural_copula.py`: `bivariate/e8_neural_copula_results_seed{0,1,2}.json` | — (needs the authors' ACNet code) |
| A30 | MI methods on Gaussian AR(1) | full run | — |
| A31 | TC estimation by dimension | `scaling.py`: `scaling/tc_benchmark_seed_merged.json` | [TC](#tc-versus-dimension) |
| A32 | Deep-tree recovery stress test | full run | — |
| A33 | Sampling quality | `sampling.py`: `sampling/sampling.json`, `sampling/compiled_sampler_summary.json` | [sampling](#sampling-quality) |
| A34 | Inverse-Rosenblatt imputation | `sampling.py`: `sampling/imputation_exact.json` | — |
| A35 | Imputation RMSE | `sampling.py`: `sampling/imputation_approximate.json` | — |
| A36 | Probit versus non-probit grids | full run with four DDIM ablation checkpoints (not released; [configurations](#training-configurations)) | — |
| A37 | No-projection ablation | full run | — |
| A38 | UCI sensitivity to IPFP iterations | `uci.py`: `uci/ipfp_sensitivity_5k.json` | — |
| A39 | IPFP iteration ablation | `ipfp_iterations.py`: `ipfp/ipfp_iter_ablation.json` | [projection timing](#projection-iteration-timing) |
| A40 | Retrained estimators, bivariate suite | not from this folder: exploratory retrained checkpoints ([documentation](../docs/)) | — |
| A41 | Retrained estimators, UCI | as A40 | — |

## Experiment runners

| Runner | Paper results | Data | Resources |
|---|---|---|---|
| `runners/reproduce_table1_likelihood.py` | Tables 1 and A17, KL columns | simulated | one GPU |
| `runners/run_uci_maf.py` | Tables A5, A6 | MAF splits | one GPU, 4 CPU threads (paper: A100 MIG 3g.20gb slice) |
| `runners/reproduce_sp100_likelihood.py` | Table A24 (VDC, TLL); Figure 2(d) | S&P100 returns matrix | one GPU, 8 CPU threads; CPU replay possible |
| `runners/reproduce_scaling_timings.py` | Table A19; Figure 2(c) | simulated | one GPU, 16 CPU cores, 96 GiB at d = 500 (paper: H100, AMD EPYC 9454; 36 fits, 9.2 h of fitting) |
| `experiments/mi_estimation.py` | Table A11; Figure 3(a) | simulated | one GPU for VDC |
| `experiments/tc_benchmark.py` | Table A31; Figures 3(b), A3(b) | simulated | one GPU, 8 CPU cores per seed (paper: H200) |
| `runners/tc_tree1_control.py` | TC structural control (appendix) | simulated | CPU (KSG only) |
| `runners/tc_replication_paired.py` | paired TC replication (appendix) | simulated | one GPU, one CPU thread per method (paper: H100, AMD EPYC 9454) |
| `experiments/mi_self_consistency_tests_v2.py` | Tables A12–A14; Figure 3(c) | simulated | one GPU |
| `runners/corruption_variants.py` | Tables A3, A4 | simulated, MAF splits | one GPU (paper: about 7 min on one A100) |
| `runners/ipfp_iterations.py` | Table A39 | simulated | one CUDA GPU (paper: H100) |
| `runners/sampling_quality.py` | Table A33 | CCPP; S&P100 returns optional | one GPU, 8 CPU threads |

Run them on allocated compute resources, with a GPU for full-size runs. Hardware, thread
counts, seeds, precision, structures and truncation are explicit in each command and recorded in
its output. Timings are deployment measurements on the stated hardware, not portable targets.
Each runner's `--help` lists all options.

### Table 1 likelihoods

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" runners/reproduce_table1_likelihood.py \
  --checkpoint "$CKPT" --expected-module-root "$VDC_LIBRARY_ROOT" \
  --device cuda --output /path/to/new-table1-run
```

Nine fixed cases, five evaluation seeds, 2,000 training and 20,000 independent test observations
per fit, drawn with pyvinecopulib. The runner reproduces the three fitted methods' held-out
likelihoods (KL to the true copula), not the ISE, tail or Kendall-tau errors or the historical
timings; the saved scores of those columns are in `tables/inputs/table1/`. Expected mean KL
(nats per edge): VDC 0.00611124, TLL 0.01348007, parametric 0.00036485. The authors' replay
matched every one of the 45 fits to within 1e-5 nats; the optional `--reference` argument performs
that per-fit check against the authors' record, which is not distributed.

### UCI benchmarks

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" runners/run_uci_maf.py \
  --data-root /path/to/data --checkpoint "$CKPT" \
  --device cuda --num-threads 4 --max-train 5000 --output /path/to/new-uci-5k
python paper/run.py --library "$VDC_LIBRARY_ROOT" runners/run_uci_maf.py \
  --data-root /path/to/data --checkpoint "$CKPT" \
  --device cuda --num-threads 4 --max-train 50000 --output /path/to/new-uci-50k
```

Defaults match the paper: POWER, GAS, HEPMASS and MINIBOONE, at most 50,000 test rows, seed 42,
2,000 bootstrap resamples, and a RealNVP flow with 8 coupling layers of width 128 trained for at
most 25 epochs. The runner checks the train and test arrays against the paper's before fitting.
`--max-train` is required because the paper reports two settings. Paper values (held-out NLL,
bits/dim):

| Rows | Dataset | VDC | pyvine-TLL | pyvine-param | RealNVP | Gaussian copula |
|---|---|---|---|---|---|---|
| 5,000 | POWER | -0.14128 | -0.18217 | -0.10777 | -0.12211 | -0.06850 |
| 5,000 | GAS | -1.47734 | -1.70500 | -1.40396 | -1.09911 | -1.24719 |
| 5,000 | HEPMASS | -0.23719 | -0.20085 | -0.16285 | 0.25381 | -0.10393 |
| 5,000 | MINIBOONE | -0.96893 | -0.74437 | -0.93342 | -0.41449 | -0.88527 |
| 50,000 | POWER | -0.14950 | -0.21693 | -0.10641 | -0.22799 | -0.06785 |
| 50,000 | GAS | -1.49258 | -1.81131 | -1.43049 | -2.13427 | -1.24975 |
| 50,000 | HEPMASS | -0.24678 | -0.26527 | -0.16574 | -0.28984 | -0.10410 |
| 50,000 | MINIBOONE | -0.97843 | -0.87162 | -0.96184 | -1.08610 | -0.89011 |

MINIBOONE has 29,556 training and 3,648 test rows, so its 50,000-row setting uses all of them.
Other devices or software builds can change the last digits (RealNVP training in particular);
compare a rerun with the reported bootstrap intervals rather than digit by digit. The dependence
diagnostic of Table A8 has its own script in the authors' workspace and is not included.

Qualification: on a reduced CPU configuration (POWER and GAS, 1,000 training and 2,000 test
rows, 100 bootstrap resamples, 3 flow epochs, 4 threads), this runner and the paper's original
runner gave identical results. All 74 reported statistics (NLL, bootstrap bounds, PIT-KS, paired
differences and their p-values) and every per-row test log-density were bitwise equal, both with
the library version used for the paper's UCI runs and with release 0.2.0, whose library code
0.2.1 keeps. The full-size GPU runs were not repeated for the release.

### Chronological S&P100

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" runners/reproduce_sp100_likelihood.py \
  --checkpoint "$CKPT" --returns /path/to/sp100_returns.npy \
  --expected-module-root "$VDC_LIBRARY_ROOT" \
  --device cuda --threads 8 --output /path/to/new-sp100-run
```

The first 1,356 trading days train and the final 339 test. The runner requires the paper's
returns matrix (float32, 1695 x 100, sha256
`5a2accb5e518b498464ea11732fa6bead1b1c34a22cc28ae95d719964ca1e476`), which is not
redistributed; [`data/sp100_symbols.json`](data/sp100_symbols.json) lists its symbols, column
order, source (Stooq daily closes) and date range. A matrix rebuilt from that list will generally
not match the hash, because the raw-price collection was not recorded in enough detail to
reproduce it, and the runner then stops; `--rebuilt-returns` runs the same protocol on such a
matrix without the checks against the paper's hash and values. With the paper's matrix the
predefined tolerances are 0.001 bits/dim for VDC and 1e-8 for TLL, relative to the historical
-0.41848794 and -0.29275967 bits/dim. Historical VDC refits were not bitwise stable, and this
replay does not replace the reported confidence intervals. A CPU replay (`--device cpu --threads 8`)
gave -0.41857 bits/dim for VDC, within 1e-4 of the reported GPU value, and TLL's value exactly.
Apart from `--rebuilt-returns`, this runner is the authors' qualified replay unchanged.

### Fitting time

Allocate one GPU, 16 CPU cores and enough host memory (the paper's d = 500 runs used 96 GiB), and
run the methods sequentially in one allocation:

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
python paper/run.py --library "$VDC_LIBRARY_ROOT" runners/reproduce_scaling_timings.py \
  --checkpoint "$CKPT" --expected-module-root "$VDC_LIBRARY_ROOT" \
  --device cuda --dimensions 100 200 500 --seeds 7 17 42 \
  --cpu-threads 16 --tll-threads 8 16 --output /path/to/new-scaling-run
```

Defaults use 20,000 training and 5,000 test rows of a Clayton vine. Each method runs in a separate
worker, in randomized order within each dimension and seed. Both VDC paths use the same library,
with `fast_fit=False` and `fast_fit=True`; the optimized path must reproduce the fitted state,
every evaluated log-likelihood and up to 128 PIT and generated samples exactly. A TLL likelihood
overflow is recorded as missing while its fit time is kept. The paper's mean seconds at
d = 100 / 200 / 500:

| Method | d = 100 | d = 200 | d = 500 |
|---|---|---|---|
| Optimized VDC | 30.05 | 118.68 | 748.99 |
| TLL, 8 threads | 202.19 | 779.61 | 4746.16 |
| TLL, 16 threads | 118.96 | 427.13 | 2488.91 |

These are H100-versus-CPU deployment times on one AMD EPYC 9454 node. VDC uses a greedy D-vine and
TLL a selected regular vine. The protocol keeps the 64-cell grid, 15 projection iterations, edge
batches of 256, float32, disabled matrix TF32 and enabled convolution TF32. Timers include
structure selection and fitting and exclude loading, warm-up, evaluation and the one-time
pretraining. A short CPU workflow check (not a timing replay): `--device cpu --dimensions 3
--seeds 7 --n-train 256 --n-test 128 --cpu-threads 4 --tll-threads 1 4`.

### Information estimation

#### Bivariate MI

Run each of seeds 42, 123 and 456 for VDC, KSG and the Gaussian-copula baseline, and seed 42 for
InfoNCE, MINE, NWJ and MINDE (MINDE needs a clone of
[MustaphaBounoua/minde](https://github.com/MustaphaBounoua/minde), passed with `--minde-repo`):

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" experiments/mi_estimation.py --estimator dcd --checkpoint "$CKPT" \
  --n-samples 5000 --seed 42 --dcd-seed-base 5 --dcd-pred-noise-clip 1.0 --dcd-diffusion-steps 8 \
  --out-json results/mi_multiseed/mi_dcd_seed42.json
python paper/run.py --library "$VDC_LIBRARY_ROOT" experiments/mi_estimation.py --estimator ksg --n-samples 5000 --seed 42 \
  --out-json results/mi_multiseed/mi_ksg_seed42.json
python paper/run.py --library "$VDC_LIBRARY_ROOT" experiments/mi_estimation.py --estimator gaussian --n-samples 5000 --seed 42 \
  --out-json results/mi_multiseed/mi_gaussian_seed42.json
python paper/run.py --library "$VDC_LIBRARY_ROOT" experiments/mi_estimation.py --estimator infonce --n-samples 5000 --seed 42 \
  --out-json results/mi_infonce.json
```

The three `--dcd-*` values are those recorded in the paper's output files (the diffusion settings
do not affect the one-pass denoiser). The suite also contains a Joe(theta = 3) case whose samples
came from a defective sampler; Table A11 and Figure 3(a) exclude it, and `tables/scripts/mi.py`
drops it from the saved records.

#### TC versus dimension

Run each of seeds 7, 17 and 42. Figure 3(b) shows the median runtime over the nine trials per
dimension, Table A31 and Figure A3(b) the mean over the three seeds; each paper run used one H200
GPU and 8 CPU cores.

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" experiments/tc_benchmark.py --checkpoint "$CKPT" --device cuda \
  --rho 0.7 --dims 2 5 10 20 50 --n 5000 --n-trials 3 --seed 7 --ksg-k 5 \
  --diffusion-steps 4 --diffusion-cfg-scale 1.0 --diffusion-pred-noise-clip 1.0 \
  --truncation-level 1 --batch-edges --edge-batch-size 256 --include-mine --include-infonce \
  --out-json results/tc_benchmark_seed7.json
```

The paper's runs fit the VDC vine truncated after its first tree (`--truncation-level 1`); for
this first-order autoregressive process the true TC lies entirely in the first tree of the D-vine
that follows the chain order.

**Structural control** (appendix, "Structural control for the TC benchmark"): KSG summed over the
same adjacent pairs, on the seed-42 samples of the Figure 3(b) sweep, with every trial and its
provenance recorded. It reads an earlier seed-42 run (`reference/tc_benchmark_neural_full.json`,
`--dims 5 10 20 50`) whose datasets and VDC and KSG estimates equal those of the sweep's seed-42
run:

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" runners/tc_tree1_control.py "$VDC_LIBRARY_ROOT" \
  paper/reference/tc_benchmark_neural_full.json /path/to/new-control.json
```

The paper's run is saved as `runners/saved_outputs/tc_tree1_control_results.json` (host and job
identifier removed). Its per-trial estimates should be reproduced exactly; its runtimes are not
compared with VDC's in the paper, because the host of the reference run was not recorded.

**Paired replication.** The paper's accuracy intervals and same-node timing come from 50 further
datasets at d = 50, with the [protocol](runners/tc_replication_PROTOCOL.md) fixed before the run.
The runner first replays the three datasets above (KSG must match exactly, VDC within 0.001 nats)
and then times both methods sequentially on one node (one GPU for VDC, one CPU thread each):

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" runners/tc_replication_paired.py --library "$VDC_LIBRARY_ROOT" \
  --checkpoint "$CKPT" --reference paper/runners/saved_outputs/tc_tree1_control_results.json \
  --device cuda --output /path/to/new-replication
```

`runners/saved_outputs/tc_replication_{summary,qualification,manifest}.json` hold the paper's run
(one H100 and an AMD EPYC 9454; host and job identifier removed): mean absolute errors 0.187 (VDC)
and 0.201 (KSG) nats, paired difference -0.013 [-0.070, 0.045], median times 0.76 s and 3.58 s.
This copy differs from the authors' runner only in taking `--reference` and in checking the
library pin (commit 67c5b1b) only with `--require-pins`. `runners/saved_outputs/MANIFEST.json`
gives the hashes of these files and of the outputs they were sanitized from.

#### MI self-consistency

Data-processing inequality, additivity and monotone invariance; the original 60-trial protocol and
the 480-trial scale-up (160 base datasets at three noise levels):

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" experiments/mi_self_consistency_tests_v2.py --checkpoint "$CKPT" \
  --n_samples 10000 --n_trials 5 --seed 42 --ksg-k 5 --device cuda \
  --json_output results/mi_self_consistency.json
python paper/run.py --library "$VDC_LIBRARY_ROOT" experiments/mi_self_consistency_tests_v2.py --checkpoint "$CKPT" \
  --n_samples 10000 --n_trials 40 --seed 42 --ksg-k 5 --device cuda \
  --json_output results/mi_self_consistency_scaleup.json
```

The DPI violation counts, the Fisher exact test over base datasets and the additivity errors of
the paper are computed from these JSON files by the figure and table scripts.

### Training-corruption variants

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" runners/corruption_variants.py --data-root /path/to/data \
  --output /path/to/new-run --expected-module-root "$VDC_LIBRARY_ROOT" --device cuda \
  --checkpoint uniform="$CKPT" --checkpoint direct=PATH \
  --checkpoint gaussian=PATH --checkpoint multinomial=PATH --checkpoint multinomial_190k=PATH
```

Every checkpoint scores identical inputs: the Table 1 cases, the out-of-distribution cases, a
common panel of real-data vine edges (one D-vine order and reference TLL h-functions give all
checkpoints the same edge inputs) and its own D-vine on the MAF splits. The first checkpoint must
be the released model; it is checked against the saved Table 1, out-of-distribution and UCI
records in `reference/` first, and the run stops if it differs. The three alternative checkpoints
(direct histograms, Gaussian grid noise and multinomial count resampling, with otherwise identical
saved training settings; the appendix lists the differences in training code and retained steps)
are not released. The paper's run, with the alternatives' sha256 values, is saved as
`reference/corruption/corruption_variants_results.json`. It took about seven minutes on one A100.

### Projection-iteration timing

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" runners/ipfp_iterations.py --checkpoint "$CKPT" --output ipfp.json
```

Requires a CUDA GPU and checks the deployed checkpoint hash. It repeats the synchronized timing
protocol of Table A39: ten warmups per iteration count K and 100 randomized-order repetitions with
synchronization before and after each projection. Another GPU or other numerical libraries can
change timings and last-bit errors.

### Sampling quality

```bash
python paper/run.py --library "$VDC_LIBRARY_ROOT" runners/sampling_quality.py \
  --checkpoint "$CKPT" --data-root /path/to/data \
  --datasets ccpp --seeds 7 17 42 --device cuda --num-threads 8 \
  --vectorized-inverse --out-json /path/to/new-sampling.json
```

For S&P100 add `--datasets ccpp sp100 --sp100-data /path/to/returns.npy`. The historical `power`
sampling rows refer to CCPP, not MAF POWER. The runner preserves the original sampling, PIT and
metric calculations and the fixed pair-selection seed. The original S&P100 matrix and the
historical GAS preprocessing are not redistributed; a rebuilt dataset gives a protocol replay, not
the exact historical scores. The separately reported compiled-sampler timings require the stated
hardware and include table packing; see [the sampler guide](../docs/compiled_sampler.md) and its
identity tests.

## Training configurations

The released checkpoint carries its own training configuration; evaluation never needs a YAML
file. [`configs/`](configs/) keeps, as source records, the configurations of two appendix
ablations whose checkpoints are not released:

- Tables A3–A4, corruption variants: `denoiser_cond_enhanced_direct.yaml`,
  `denoiser_cond_enhanced_gaussian.yaml`, `denoiser_cond_enhanced_multinomial.yaml`. Their
  `conditional_*` training examples are rejected by the corrected training-data generator of
  release 0.2.7 ([training data](../docs/TRAINING_DATA.md)).
- Table A36, probit comparison (DDIM checkpoints, not the one-pass released model):
  `diffusion_cond_m64_tuned_bilinear.yaml`, `diffusion_cond_m64_tuned_probit_bilinear.yaml`,
  `diffusion_cond_m128_tuned_bilinear.yaml`, `diffusion_cond_m128_tuned_probit_bilinear.yaml`.

Train with `scripts/train_unified.py` ([configuration guide](../docs/CONFIGURATION.md)). The
released model was trained on one H100 GPU for 240,000 steps (9.6 h); the deployed step-190,000
checkpoint was written after 7.6 h. Its training source is only partially pinned, so retraining
is not a bitwise reconstruction of that run; use the released checkpoint for evaluation.

## Versions and pins

The runners were qualified with library release v0.2.6, whose `vdc` runtime is identical to
v0.2.3 (commit 9b2d33ca610b63282b9bdc493dcad607522d313b) except for its version string; that
release was qualified against the earlier camera-ready runtime. Generation pins recorded inside
saved outputs identify the original runs and are not overwritten. No new checkpoint is used.
Release 0.2.7 changed only training-data generation in `vdc.data` and the legacy grid trainer;
among the paper's runners this affects only the Joe case of the MI benchmark (now sampled
correctly), which the paper excludes. Release 0.3.0 adds the opt-in linear-grid estimators and reorganizes
files; for `vdc-denoiser-m64-v1` its outputs are bitwise identical to 0.2.7. Check the release
notes of later versions for numerical changes. For the qualified runtime, check out tag `v0.2.6` and pass that checkout as `--library`
and `$VDC_LIBRARY_ROOT`.

The runners were relocated from the qualified paper workflows with their numerical parameters and
estimators preserved: the MI runner requires an explicit checkpoint, and the sampling runner takes
its data paths as arguments and names the historical CCPP data explicitly. `reference/` holds
selected immutable inputs, not a second maintained experiment; `reference/MANIFEST.json` records
their released and source hashes.

## Scope and limitations

The paper is not completely reproducible from a fresh download:

- **S&P100.** The returns matrix is not redistributed, and its collection metadata cannot
  reconstruct it exactly. Figures 2(d), A5 and A6 and Tables A25 and A33 regenerate from saved
  outputs, and the saved outputs behind Tables A23–A24 are included, but every S&P100 rerun needs
  the paper's matrix; a rebuilt matrix gives a protocol replay, not the reported values.
- **Checkpoints.** The alternative corruption checkpoints (Tables A3–A4) and the DDIM ablation
  checkpoints (Table A36) are not released, and the exploratory retrained estimators of
  Tables A40–A41 are separate models (see the [documentation](../docs/)). None of these results
  may be attributed to the released checkpoint.
- **Training.** The deployed model's training source is only partially pinned, so the public
  checkpoint permits evaluation without claiming an exact retraining replay.
- **Partial runners.** The Table 1 runner covers the held-out likelihoods of VDC, TLL and
  parametric selection, not every ISE, tail or timing column; their saved scores and the table
  generator are included. Twelve appendix tables were written by full experiment runs in the
  authors' workspace and have no generator here; of these, only the VDC and TLL rows of
  Table A24 have a public runner (see the inventory).
- **External code.** MINDE needs its upstream implementation, and the ACNet runs of Table A29 need
  the authors' code.
- **Defective Joe case.** Historical MI files include a defective Joe case; the published MI
  summary and the table generator exclude it.
- **Qualification.** Figure and table regeneration uses saved outputs. The release qualification
  checked runtime identity, rendering, imports and reduced numerical workflows; it did not repeat
  the complete training and full-size experimental campaign.

## Data and licenses

Code is covered by the repository's [MIT license](../LICENSE). External data and baseline
software keep their own licenses; the MAF archive is CC BY 4.0. Source and checkpoint hashes are
recorded separately from runtime and hardware measurements.
