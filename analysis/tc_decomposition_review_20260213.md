# TC Decomposition Review (2026-02-13)

## Goal
Review the new TC edge-decomposition analysis and regenerate it from **real data** using the canonical paper checkpoint.

Canonical checkpoint used:

`/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/checkpoints/model_step_190000.pt`

## Real-data runs completed

Script:

`drafts/scripts/generate_tc_decomposition_figure.py`

Mode:

`--mode uci` (fit D-vine on empirical-copula-transformed train set; decompose TC on held-out test set).

### 1) UCI Power (`d=5`)
- Artifact JSON: `drafts/paper_outputs/tc_decomposition_uci_power.json`
- Figure: `drafts/figures/fig_tc_decomposition.pdf`
- Table: `drafts/tables/tab_tc_decomposition_uci_power.tex`
- Results:
  - TC = `2.3566` nats
  - Tree-1 abs share = `89.0%`
  - Tree-(1+2) abs share = `97.1%`
- Interpretation quality: **strong** (clear concentration in early trees, readable edge ranking).

### 2) UCI Gas (`d=8`)
- Artifact JSON: `drafts/paper_outputs/tc_decomposition_uci_gas.json`
- Figure: `drafts/figures/fig_tc_decomposition_gas.pdf`
- Table: `drafts/tables/tab_tc_decomposition_uci_gas.tex`
- Results:
  - TC = `-0.0010` nats (near-zero, with sign cancellations across edges)
  - Tree-1 abs share = `21.4%`
  - Tree-(1+2) abs share = `46.0%`
- Interpretation quality: **weak** (near-independence regime, not compelling as main story).

### 3) UCI Hepmass (`d=21`, subsampled train/test)
- Artifact JSON: `drafts/paper_outputs/tc_decomposition_uci_hepmass.json`
- Figure: `drafts/figures/fig_tc_decomposition_hepmass.pdf`
- Table: `drafts/tables/tab_tc_decomposition_uci_hepmass.tex`
- Results:
  - TC = `-0.0278` nats (again near-zero signed TC under this estimator setting)
  - Tree-1 abs share = `14.1%`
  - Tree-(1+2) abs share = `19.8%`
- Interpretation quality: **weak** for main-paper interpretability messaging.

## Decision

Use the **Power** real-data decomposition in the main paper.  
Do **not** use Gas as the flagship panel despite matching `d=8`, because it does not support the interpretability claim strongly in this estimator regime.

## Paper integration status

- Main-text figure inserted in Information section:
  - `drafts/vine_diffusion.tex` (`Figure~\ref{fig:tc_decomposition}`)
  - Caption currently uses real Power values (89.0%, 97.1%).
- Figure generation is now wired into artifact pipeline:
  - `drafts/scripts/paper_artifacts.py` calls
    `generate_tc_decomposition_figure.py --mode uci --dataset power ...`

## Repro command

```bash
python drafts/scripts/generate_tc_decomposition_figure.py \
  --mode uci \
  --dataset power \
  --device cpu \
  --out-prefix fig_tc_decomposition \
  --out-json drafts/paper_outputs/tc_decomposition_uci_power.json \
  --out-tex drafts/tables/tab_tc_decomposition_uci_power.tex
```
