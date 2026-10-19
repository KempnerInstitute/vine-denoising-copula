# Paper tables

This folder rebuilds tables of the NeurIPS 2026 paper from saved experiment outputs. The generators
only format saved numbers; none of them runs an estimator, so they need Python 3.9+ with numpy and
scipy and finish in seconds.

```bash
bash make_tables.sh /tmp/vdc_tables                    # write the tables
bash make_tables.sh /tmp/vdc_tables path/to/tables     # ... and compare them with a reference copy
```

The comparison ignores full-line LaTeX comments (each generated table starts with one naming its
generator) and is otherwise byte for byte. All 21 generated tables are identical to the paper's tables.

## Inputs

`inputs/` holds the saved outputs the generators read, about 0.7 MB in total. Every input that
records its estimator was produced with the released model `vdc-denoiser-m64-v1` (the deployed
checkpoint, `model_step_190000.pt`, sha256 `8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f`).

- JSON files were rewritten without absolute paths (only the final path component is kept),
  hostnames, scheduler job identifiers or reviewer identifiers. Numbers are unchanged.
  `MANIFEST.json` gives, for every file, its sha256 and the sha256 of the original output it was
  copied from.
- `ood/e26_rescored.json` is a compact extract of the out-of-distribution re-scoring output
  (`e31_e26_check.json`, 1,153,279 bytes, sha256
  `9bf91f5dcd83e9f81e3803c2fad071f771e15bcf4dc09672cb35c6bc46720d98`): the nine per-case summaries
  and their means. `scripts/compact_inputs.py` made it.
- `threads/d500_single_thread.json` was written by hand. It records when the single-thread
  d = 500 TLL fit of the thread sweep started and when the sweep job reached its time limit with the
  fit still running, taken from the sweep's file times and the scheduler's accounting.
- The data set called `power` in `sampling/` is the Combined Cycle Power Plant (CCPP) data, not the
  MAF POWER benchmark.

## Generators

| Script | Tables | Inputs |
|---|---|---|
| `table1.py` | `tab_bivariate` (Table 1), `tab_bivariate_dispersion` | `table1/table1_v2.json` |
| `uci.py` | `tab_vine_nll_ci`, `tab_vine_nll_ci_50k`, `tab_uci_matched_structure`, `tab_ipfp_uci_sensitivity` | `uci/*.json` |
| `ood.py` | `tab_ood_families` | `ood/e26_rescored.json` |
| `mi.py` | `tab_mi` | `mi/results/**` |
| `scaling.py` | `tab_highdim_scaling`, `tab_tc_scaling`, `tab_fast_fit_timing` | `scaling/*` |
| `sampling.py` | `tab_sampling_quality`, `tab_imputation_exact`, `tab_imputation` | `sampling/*.json` |
| `self_consistency.py` | `tab_self_consistency`, `tab_self_consistency_scaleup`, `tab_dpi_tolerance`, `tab_additivity_learned` | `self_consistency/*.json` |
| `ipfp_iterations.py` | `tab_ipfp_iter_ablation` | `ipfp/ipfp_iter_ablation.json` |
| `raw_composition.py` | `tab_raw_composition` | `sp100/e37_raw_composition.json` |
| `threads.py` | `tab_pyvine_threads` | `threads/*.json`, `scaling/e22_highdim_scaling_seed_merged.json` |

Notes:
- The time column of Table 1 is not measured by the benchmark the table summarizes. It holds the
  per-fit times of the same five procedures from the original timing run at n = 2000 (VDC on one
  H100 GPU, baselines on one CPU thread); `table1.py` states these constants.
- `tab_mi` drops the Joe case of the saved suite, whose samples came from a defective sampler.
- `tab_dpi_tolerance` recounts the DPI violations of the scaled protocol at each tolerance; its
  Fisher tests are two-sided and pool the 480 trials of each estimator.
- `tab_pyvine_threads`: the paper's table was checked against these saved outputs and corrected in
  two places. At d = 100 with 8 threads the three saved fits (202.87, 203.05 and 204.12 s) give a
  population standard deviation of 0.55 s, printed as 0.6 (earlier versions printed 0.5). The
  single-thread d = 500
  fit started at 21:20:53 and was still running when the sweep job reached its 16-hour limit at
  05:55:42 the next morning, so it took more than 30,889 s: the table prints > 3.0 x 10^4 s and a
  > 28x speedup over VDC's 1,081.6 s (earlier versions gave > 2.2 x 10^4 s, a 6.3-hour budget and
  > 20x).

## Every table in the paper

Released here (generator and inputs in this folder): the 21 tables listed above.

Written at the end of a full experiment run by a runner from the paper workspace; the runner and
its raw outputs are not in this folder:

| Table | Experiment | Notes |
|---|---|---|
| `tab_uci_dependence_diag` | dependence-signal diagnostic on the MAF splits | dataset labels upper-cased by hand |
| `tab_noipfp_ablation` | no-projection ablation on the MAF splits | method labels edited by hand |
| `tab_flow_sensitivity` | RealNVP capacity sweep | rows from withdrawn UCI files removed by hand; Clayton rows unchanged |
| `tab_nonsimplified_stress` | non-simplified conditional stress test | |
| `tab_bivariate_replicates` | paired bivariate replicates | ISE column rescaled in place (an extra 1/m^2 factor removed) |
| `tab_bivariate_complex` | complex-pattern bivariate suite | ISE rescaled in place |
| `tab_neural_copula` | neural-copula comparison (ACNet) | needs the authors' ACNet code; ISE rescaled in place |
| `tab_theory_mi_methods` | synthetic MI comparison | |
| `tab_probit_comparison` | probit-grid ablation | separately trained ablation models, not the released model; ISE rescaled in place |
| `tab_cpu_timing` | matched-hardware CPU timing | |
| `tab_heterogeneous_scaling` | mixed-family (Gaussian + Clayton) vine scaling | |
| `tab_deep_tree_recovery` | deep-tree stress test | |
| `tab_sp100_density` | S&P100, random split | needs the returns matrix, which is not redistributed |
| `tab_sp100_chrono` | S&P100, chronological split | as above |

Depend on experimental checkpoints that are not released: `tab_retrained_bivariate` and
`tab_retrained_uci` (the exploratory VDC-64 and VDC-128 estimators).

Not reproducible from saved outputs:
- `tab_ipfp_validity` has neither a generator nor a saved input. Its values were entered before
  the deployed checkpoint was trained, and one value was later edited by hand, so it does not
  describe the released model.
- `tab_vine_nll` is a fallback used only when `tab_vine_nll_ci.tex` is missing. It still holds
  withdrawn rows and is not printed in the paper.

Tables written directly in the manuscript source:
- `tab:claims_evidence` maps claims to the figures and tables that support them; it holds no data.
- `tab:benchmark_spec` gives the training-zoo and evaluation-suite specification, written by hand
  from the benchmark code defaults.
- `tab:grid_floor` gives the representation floor of the copula grid. Its values were typed in
  from the output of the grid-floor analysis, which is not in this folder.
