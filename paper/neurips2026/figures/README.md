# Paper figures

The scripts in `scripts/` draw every figure of the NeurIPS 2026 paper from the saved experiment outputs in
`inputs/`; nothing is refitted.

```bash
bash make_figures.sh [OUTPUT_DIR]        # default ./output; PYTHON=/path/to/python to choose the interpreter
```

Each figure is written as a vector PDF (embedded TrueType fonts) and a 400-dpi PNG at its printed size
(5.5 in, the NeurIPS text width; 3.9 in for Figure A5).

| Figure | Script (in `scripts/`) | Inputs (in `inputs/`) |
|---|---|---|
| 1 | `fig1_overview_cr.py` | `fig1_overview/fig1_overview_data.npz` |
| 2 | `fig2_density_runtime_cr.py` (loaders in `fig4_density_runtime_composite_v2.py`, `paper_style.py`) | `tables/tab_bivariate.tex`, `paper_outputs/e22_highdim_scaling.json`, `fast_fit_scaling/timing_summary.json`, `paper_outputs/e23_sp100_density.json`, `paper_outputs/e28_sp100_chrono.json` |
| 3 | `fig3_information_cr.py` | `tables/tab_mi.tex`, `paper_outputs/e6_theory_synthetic_results.json`, `paper_outputs/tc_benchmark_neural_full.json` |
| A1 | `figA_training_diagnostics_cr.py` | `fig1_overview/fig1_overview_data.npz` |
| A2 | `figA_nonsimplified_stress_cr.py` | `paper_outputs/e18_nonsimplified_stress.json`; values are checked against `tables/tab_nonsimplified_stress.tex` |
| A3 | `figA_tc_story_cr.py` (tree-share helpers in `fig_tc_story_composite.py`) | `paper_outputs/tc_decomposition_ground_truth_compare.json`, `tc_scaling/tc_benchmark.json`, `tc_knockout/tc_decomposition_uci_maf_{power,gas,hepmass}.json`, `tc_knockout/e17_tree_knockout_tc_maf.json`, `paper_outputs/mi_self_consistency_scaleup.json` |
| A4 | `figA_mi_sample_size_cr.py` | `paper_outputs/e12_block_mi_sample_size_clayton_chain_results_with_neural_seed42.json` |
| A5 | `figA_var_calibration_cr.py` | `var_backtest/e3_var_results.json` |
| A6 | `figA_rolling_dependence_cr.py` | `paper_outputs/e16_rolling_dependence_compact.json`, `sp100/sp100_returns_meta.json` |

`vdc_figstyle.py` holds the shared fonts, sizes and per-method colours, markers, line styles and hatches.

## Inputs and their provenance

`inputs/MANIFEST.json` lists every input with its sha256; the scripts refuse a file whose hash does not
match. Its `source_sha256` is the hash of the original experiment output. The released copies differ from
the originals only in metadata: absolute storage paths are cut to their file names, a cluster host name is
replaced by `<host>`, and LaTeX comment lines are removed. Every number is unchanged. Two inputs are
reduced:

- `paper_outputs/e16_rolling_dependence_compact.json` keeps the fields of the full rolling-dependence output
  that Figure A6 reads (`scripts/extract_rolling_dependence_inputs.py`).
- `sp100/sp100_returns_meta.json` gives the 100 S&P100 tickers (Stooq symbols), the first and last dates and
  the shape of the returns matrix; the returns themselves are not distributed.

The returns matrix stores no per-row dates, so Figure A6 assigns window end dates by mapping its 1,695 rows
to consecutive NYSE trading days between the recorded first and last dates, after checking that the count
matches. With `--returns PATH` the script also checks that known extreme market days are among the largest
moves; the figure is identical with or without this check.

## Figure 1 data (optional)

`fig1_overview_data.npz` holds outputs of the released checkpoint (densities, histograms and samples) and the
downsampled training-loss history. To recompute the model outputs, install this repository and download the
checkpoint (`python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1`, sha256
`8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f`), then run

```bash
python scripts/fig1_overview_data.py --lib <repository root> --checkpoint /path/to/model.pt \
    --loss-npz inputs/fig1_overview/fig1_overview_data.npz --out /path/to/new/fig1_overview_data.npz
```

The training log is not distributed; `--loss-npz` copies the loss arrays from the saved file.
`inputs/fig1_overview/fig1_overview_data.json` records the versions used (CPU, torch 2.9.1, pyvinecopulib
0.7.5, numpy 2.2.6).

## Verification

With Python 3.10.20, numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.6 and Pillow 11.3.0, `make_figures.sh`
reproduces the figures of the camera-ready paper exactly: identical PNG pixels and identical 300-dpi
rasterizations of the PDFs, for all nine figures. With the same library version and checkpoint, the Figure 1
data step on CPU reproduces every array of `fig1_overview_data.npz` exactly. Other versions of matplotlib or
its bundled STIX fonts may move text by a pixel.
