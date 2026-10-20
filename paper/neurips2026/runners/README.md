# Experiment runners for the NeurIPS 2026 paper

These runners reproduce the paper's main experiments with the released model and the `vdc` package
of this repository (version 0.2.0). Each one writes into a new output directory, records the
checkpoint hash and software versions, and never edits files outside that directory. Figures and
tables are rebuilt from saved outputs by the scripts in `../figures` and `../tables`.

| Paper result | Runner | Data |
|---|---|---|
| Table 1, likelihood (KL) columns of VDC, pyvine-TLL and parametric selection | `reproduce_table1_likelihood.py` | simulated |
| UCI held-out NLL with bootstrap CIs, 5,000 and 50,000 training rows (Appendix UCI tables) | `run_uci_maf.py` | MAF splits, prepared by `scripts/download_uci_datasets.py` |
| Chronological S&P100 likelihood of VDC and pyvine-TLL | `reproduce_sp100_likelihood.py` | the paper's returns matrix (not distributed; see below) |
| Fixed-thread vine-fitting timings, d = 100, 200, 500 (Figure 2(c); fast-fit timing table) | `reproduce_scaling_timings.py` | simulated |
| Bivariate MI benchmark (Figure 3(a); MI table) | `scripts/mi_estimation.py` (library) | simulated |
| TC versus dimension (Figure 3(c); appendix TC tables and figure) | `scripts/tc_benchmark.py` (library) | simulated |
| MI self-consistency, 60- and 480-trial protocols | `scripts/mi_self_consistency_tests_v2.py` (library) | simulated |

Run everything on allocated compute resources (a GPU for the full-size runs), with output
directories that do not exist yet.

## Setup

From the repository root, on Python 3.10:

    python -m pip install -c paper/neurips2026/runners/constraints.txt -e .
    python -m pip check

`constraints.txt` pins the numerical packages of the environment in which the paper's replays were
qualified (torch 2.9.1, NumPy 2.2.6, SciPy 1.15.3, pyvinecopulib 0.7.5, ...). NumPy and BLAS
versions can change the last digits of reported summaries.

Download the released model and check its hash:

    python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1    # prints the local path
    sha256sum /path/to/vdc-denoiser-m64-v1.pt
    # 8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f

Every runner checks this hash (`run_uci_maf.py` accepts another model only with
`--expected-checkpoint-sha256 any`). The retrained 64- and 128-cell models of the appendix are
different estimators and are not released.

`reproduce_table1_likelihood.py`, `reproduce_sp100_likelihood.py` and
`reproduce_scaling_timings.py` take `--expected-module-root`, the directory that contains the `vdc`
package being imported (the repository root for an editable install, or `site-packages` for a
wheel); they stop if Python imports `vdc` from anywhere else.

## Table 1 likelihoods

    python paper/neurips2026/runners/reproduce_table1_likelihood.py \
      --checkpoint /path/to/vdc-denoiser-m64-v1.pt \
      --expected-module-root /path/to/repository \
      --device cuda --output /path/to/new-table1-run

Nine fixed cases, five evaluation seeds, 2,000 training and 20,000 independent test observations
per fit, drawn with pyvinecopulib. This reproduces the three fitted methods' held-out likelihoods
(KL to the true copula), not ISE, tail or Kendall-tau errors or the historical timings. Expected
mean KL (nats per edge): VDC 0.00611124, TLL 0.01348007, parametric 0.00036485. The authors'
replay matched every one of the 45 fits to within 1e-5 nats; the optional `--reference` argument
performs that per-fit check against the authors' record, which is not distributed.

## UCI benchmarks on the MAF splits

Prepare the data once (about 860 MB download; GAS and HEPMASS need pandas,
`pip install "vine-denoising-copula[data]"`):

    python scripts/download_uci_datasets.py --out /path/to/uci_maf

This writes `/path/to/uci_maf/uci/{power,gas,hepmass,miniboone}.npz`. Then, for the two training
sizes of the paper:

    python paper/neurips2026/runners/run_uci_maf.py \
      --data-root /path/to/uci_maf --checkpoint /path/to/vdc-denoiser-m64-v1.pt \
      --device cuda --num-threads 4 --max-train 5000 --output /path/to/new-uci-5k

    python paper/neurips2026/runners/run_uci_maf.py \
      --data-root /path/to/uci_maf --checkpoint /path/to/vdc-denoiser-m64-v1.pt \
      --device cuda --num-threads 4 --max-train 50000 --output /path/to/new-uci-50k

Defaults match the paper: POWER, GAS, HEPMASS and MINIBOONE, at most 50,000 test rows, seed 42,
2,000 bootstrap resamples, a RealNVP flow with 8 coupling layers of width 128 trained for at most
25 epochs. The runner checks the train and test arrays against the paper's before fitting.
`--max-train` is required because the paper reports two settings.

Paper values (held-out NLL, bits/dim; one A100 MIG 3g.20gb slice and 4 CPU threads):

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
Results on other devices or software builds can differ in the last digits (RealNVP training in
particular); compare a rerun with the reported bootstrap intervals rather than digit by digit.
The appendix's dependence diagnostic of the UCI data has its own script in the paper workspace and
is not ported here.

Qualification: on a reduced CPU configuration (POWER and GAS, 1,000 training and 2,000 test rows,
100 bootstrap resamples, 3 flow epochs, 4 threads), this runner and the paper's original runner gave
identical results. All 74 reported statistics (NLL, bootstrap bounds, PIT-KS, paired differences
and their p-values) and every per-row test log-density were bitwise equal, both with the library
version used for the paper's UCI runs and with this release (0.2.0). The full-size GPU runs were
not repeated for the release.

## Chronological S&P100

    python paper/neurips2026/runners/reproduce_sp100_likelihood.py \
      --checkpoint /path/to/vdc-denoiser-m64-v1.pt --returns /path/to/sp100_returns.npy \
      --expected-module-root /path/to/repository \
      --device cuda --threads 8 --output /path/to/new-sp100-run

The first 1,356 trading days train and the final 339 test. The runner requires the paper's
returns matrix (float32, 1695 x 100, sha256
`5a2accb5e518b498464ea11732fa6bead1b1c34a22cc28ae95d719964ca1e476`), which is not redistributed:
`../data/sp100_symbols.json` lists its symbols, column order, source (Stooq daily closes) and date
range. A matrix rebuilt from that list will generally not match the hash, because the raw-price
collection was not recorded in enough detail to reproduce it exactly, and the runner then stops;
add `--rebuilt-returns` to run the same protocol on such a matrix without the checks against the
paper's hash and values.
With the paper's matrix the predefined tolerances are 0.001 bits/dim for VDC and 1e-8 for TLL,
relative to the historical -0.41848794 and -0.29275967 bits/dim. Historical VDC refits were not
bitwise stable, and this replay does not replace the reported confidence intervals. A CPU replay
with this release (`--device cpu --threads 8`) gave -0.41857 bits/dim for VDC, within 1e-4 of the
reported GPU value, and TLL's value exactly. Apart from `--rebuilt-returns`, this runner is the
authors' qualified replay unchanged.

## Fitting-time protocol

Allocate one GPU, 16 CPU cores and enough host memory (the paper's d = 500 runs used 96 GiB), and
run the methods sequentially in one allocation:

    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
    python paper/neurips2026/runners/reproduce_scaling_timings.py \
      --checkpoint /path/to/vdc-denoiser-m64-v1.pt \
      --expected-module-root /path/to/repository \
      --device cuda --dimensions 100 200 500 --seeds 7 17 42 \
      --cpu-threads 16 --tll-threads 8 16 --output /path/to/new-scaling-run

Defaults use 20,000 training and 5,000 test rows of a Clayton vine. Each method runs in a separate
worker, with randomized method order within each dimension and seed. Both VDC paths use the same
library with `fast_fit=False` and `fast_fit=True`; the optimized path is required to reproduce the
fitted state, every evaluated log-likelihood and up to 128 PIT and generated samples exactly. TLL
likelihood overflow is recorded as missing while its fit time is kept.

The paper's mean seconds at d = 100 / 200 / 500 are:

    Optimized VDC:   30.05 / 118.68 / 748.99
    TLL, 8 threads: 202.19 / 779.61 / 4746.16
    TLL, 16 threads: 118.96 / 427.13 / 2488.91

These are H100-versus-CPU deployment times on one AMD EPYC 9454 node, not portable timing targets.
VDC uses a greedy D-vine and TLL a selected regular vine. The protocol keeps the 64-cell grid, 15
projection iterations, edge batches of 256, float32, disabled matrix TF32 and enabled convolution
TF32. Timers include structure selection and fitting and exclude loading, warm-up, evaluation and
the one-time pretraining. A short CPU workflow check: `--device cpu --dimensions 3 --seeds 7
--n-train 256 --n-test 128 --cpu-threads 4 --tll-threads 1 4` (not a timing replay).

## Information estimation (library runners)

Bivariate MI benchmark, run from the repository root for each seed 42, 123 and 456 (VDC, KSG and
the Gaussian-copula baseline) and for seed 42 (InfoNCE, MINE, NWJ, and MINDE, which needs a clone
of github.com/MustaphaBounoua/minde passed with `--minde-repo`):

    python scripts/mi_estimation.py --estimator dcd --checkpoint /path/to/vdc-denoiser-m64-v1.pt \
      --n-samples 5000 --seed 42 --dcd-seed-base 5 --dcd-pred-noise-clip 1.0 --dcd-diffusion-steps 8 \
      --out-json results/mi_multiseed/mi_dcd_seed42.json
    python scripts/mi_estimation.py --estimator ksg --n-samples 5000 --seed 42 \
      --out-json results/mi_multiseed/mi_ksg_seed42.json
    python scripts/mi_estimation.py --estimator gaussian --n-samples 5000 --seed 42 \
      --out-json results/mi_multiseed/mi_gaussian_seed42.json
    python scripts/mi_estimation.py --estimator infonce --n-samples 5000 --seed 42 \
      --out-json results/mi_infonce.json

The three `--dcd-*` values are those recorded in the paper's output files (the diffusion settings
do not affect the one-pass denoiser). The suite also contains a Joe(theta = 3) case whose samples
come from a defective sampler (see the paper's training-data appendix); the paper's MI table and
Figure 3(a) exclude it, and the table generator in `../tables` drops it from the saved records.

TC versus dimension, for each seed 7, 17 and 42 (Figure 3(c) shows the median runtime over the nine
trials per dimension, the appendix TC tables and figure the mean over the three seeds; each paper run
used one H200 GPU and 8 CPU cores):

    python scripts/tc_benchmark.py --checkpoint /path/to/vdc-denoiser-m64-v1.pt --device cuda \
      --rho 0.7 --dims 2 5 10 20 50 --n 5000 --n-trials 3 --seed 7 --ksg-k 5 \
      --diffusion-steps 4 --diffusion-cfg-scale 1.0 --diffusion-pred-noise-clip 1.0 \
      --truncation-level 1 --batch-edges --edge-batch-size 256 --include-mine --include-infonce \
      --out-json results/tc_benchmark_seed7.json

The paper's runs fit the VDC vine truncated after its first tree (`--truncation-level 1`); for this
first-order autoregressive process the true TC lies entirely in the first tree of the D-vine that
follows the chain order.

Structural control (paper appendix, "Structural control for the TC benchmark"): KSG summed over the same
adjacent pairs, on the seed-42 samples of the Figure 3(c) sweep, with every trial and its provenance
recorded. It reads an earlier seed-42 run (`tc_benchmark_neural_full.json`, `--dims 5 10 20 50`) whose
datasets and VDC and KSG estimates equal those of the sweep's seed-42 run:

    python paper/neurips2026/runners/tc_tree1_control.py . \
      paper/neurips2026/figures/inputs/paper_outputs/tc_benchmark_neural_full.json /path/to/new-control.json

The saved output of the paper's run is `saved_outputs/tc_tree1_control_results.json` (host and job id
removed). Its per-trial estimates should be reproduced exactly; its runtimes are not compared with VDC's
in the paper, because the host of the reference run was not recorded.

The paper's accuracy intervals and same-node timing come from a paired replication on 50 further datasets
at d = 50, with the protocol in `tc_replication_PROTOCOL.md` fixed before the run. It first replays the
three datasets above (KSG must match exactly, VDC within 0.001 nats) and then times both methods
sequentially on one node (one GPU for VDC, one CPU thread each):

    python paper/neurips2026/runners/tc_replication_paired.py --library . \
      --checkpoint /path/to/vdc-denoiser-m64-v1.pt \
      --reference paper/neurips2026/runners/saved_outputs/tc_tree1_control_results.json \
      --device cuda --output /path/to/new-replication

`saved_outputs/tc_replication_{summary,qualification,manifest}.json` hold the paper's run (one H100 and
an AMD EPYC 9454; host and job id removed): mean absolute errors 0.187 (VDC) and 0.201 (KSG) nats, paired
difference -0.013 [-0.070, 0.045], median times 0.76 s and 3.58 s. This copy differs from the authors'
runner only in taking `--reference` and in checking the library pin only with `--require-pins`.

MI self-consistency (data-processing inequality, additivity and monotone invariance), original
60-trial protocol and the 480-trial scale-up (160 base datasets at three noise levels):

    python scripts/mi_self_consistency_tests_v2.py --checkpoint /path/to/vdc-denoiser-m64-v1.pt \
      --n_samples 10000 --n_trials 5 --seed 42 --ksg-k 5 --device cuda \
      --json_output results/mi_self_consistency.json
    python scripts/mi_self_consistency_tests_v2.py --checkpoint /path/to/vdc-denoiser-m64-v1.pt \
      --n_samples 10000 --n_trials 40 --seed 42 --ksg-k 5 --device cuda \
      --json_output results/mi_self_consistency_scaleup.json

The DPI violation counts, the Fisher exact test over base datasets and the additivity errors in the
paper are computed from these JSON files by the figure and table scripts.
