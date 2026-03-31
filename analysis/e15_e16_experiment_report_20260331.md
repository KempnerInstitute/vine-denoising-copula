# E15 and E16 Experiment Report

Date: 2026-03-31

This note records the implementation, results, and paper decisions for the two late-cycle experiment tracks:

- E15: time-budgeted D-vine structure search
- E16: rolling dependence dashboard on S&P100 returns

The goal is to preserve both the positive results and the negative or mixed findings, so later revisions do not need to reconstruct what was tried and why some additions were kept out of the paper.

## E15: Structure Search

### Goal

Test whether VDC's edge-fitting speed can be converted into a stronger claim than "fits a fixed vine faster":

- evaluate many D-vine orderings cheaply with VDC
- compare best validation NLL found so far versus cumulative wall-clock
- check whether faster search can compensate for weaker per-order fitting quality

### Implementation

Files added or updated:

- `drafts/scripts/e15_structure_search_benchmark.py`
- `drafts/scripts/fig_structure_search_frontier.py`
- `slurm/paper_e15_structure_search.sh`
- `slurm/submit_paper_e15_structure_search.sh`
- `vdc/baselines/pyvinecopulib.py`

Important implementation detail:

- `pyvinecopulib` fitting had to be extended to support explicit fixed D-vine orders so both methods could be evaluated on the same candidate structures.
- `pyvinecopulib` expects 1-indexed `DVineStructure` orders; using 0-indexed orders produced misleading errors.

### Jobs Run

- `2781755`: Power + Gas pilot
- `2781820`: Hepmass

### Artifacts

- `drafts/paper_outputs/e15_structure_search_pilot_power_gas.json`
- `drafts/figures/fig_structure_search_frontier_pilot.png`
- `drafts/paper_outputs/e15_structure_search_hepmass.json`
- `drafts/figures/fig_structure_search_frontier_hepmass.png`

### Main Results

Power (`d=5`):

- VDC random search improves over VDC greedy.
- VDC has a better quality-time frontier at very small budgets.
- pyvine-TLL still wins once it is given enough time.

Gas (`d=8`):

- VDC is much faster.
- VDC random search improves only slightly over VDC greedy.
- pyvine-TLL is already better in validation NLL once it enters the budget range.

Hepmass (`d=21`):

- VDC random search improves over VDC greedy.
- VDC evaluates many candidates in the time pyvine fits one or a few.
- pyvine-TLL greedy is still better in validation NLL than VDC best-of-12.

### Most Important Caveat

The higher-dimensional pilots were candidate-matched, not fully time-matched.

That means:

- Gas and Hepmass establish a feasibility gap and a per-order quality gap
- they do not yet establish the stronger claim that VDC finds a better final model under the same wall-clock budget

This matters because on Hepmass, for example, VDC could have evaluated far more candidate orders than pyvine in the same total time, but that exact experiment was not run.

### Surrogate-Screening Check

The existing E15 JSONs were mined for a screening-style interpretation:

- rank correlation between VDC and pyvine validation NLLs
- top-k overlap between the best candidate orders under each method

Results:

Power:

- Spearman correlation: `0.763`
- Kendall tau-b: `0.567`
- top-5 overlap: `1 / 5`
- top-10 overlap: `2 / 10`
- top-20 overlap: `8 / 20`

Interpretation:

- VDC has some broad ranking signal on Power
- but the top of the ranking is not reliable enough to support a strong paper claim about screening or hybrid search

Gas and Hepmass:

- the stored pyvine validation NLL values were tied at `0.0` in the current artifacts
- therefore the surrogate-screening idea could not be evaluated honestly from these runs

### Decision

E15 was not added to the main paper.

Reason:

- the experiment is valid and informative
- VDC search clearly helps relative to VDC greedy
- but the current evidence does not support a strong claim that faster search makes VDC the better final density model

Best current use:

- internal diagnostic
- possible future appendix note
- future direction: true time-matched structure search, possibly beyond D-vines

## E16: Rolling Dependence Dashboard

### Goal

Use the rolling S&P100 pipeline to demonstrate a capability that global flow baselines do not naturally provide:

- rolling dependence monitoring
- tree-level total-correlation decomposition
- named edge-level drivers of dependence over time

This is intended as a descriptive, interpretability-oriented figure, not a predictive benchmark win.

### Implementation

Files added or updated:

- `vdc/vine/recursion.py`
- `vdc/vine/api.py`
- `drafts/scripts/e16_rolling_dependence_dashboard.py`
- `drafts/scripts/fig_rolling_dependence_dashboard.py`
- `slurm/paper_e16_rolling_dependence.sh`

Core method change:

- added a decomposed log-density path that returns
  - total mean log-copula contribution
  - per-tree mean contributions
  - per-edge mean contributions

This reuses the existing `logpdf` recursion but records the contributions instead of only summing them.

### Semantic Choice

The figure and text use "contribution to total dependence" or "tree-level contribution" language.

This is deliberate:

- unconditional tree-1 edges align naturally with pairwise dependence summaries
- deeper tree terms are conditional contributions to total correlation
- calling all of them "edge MI" would be imprecise

### Jobs Run

- `2783595`: pilot
- `2783861`: full run

### Runtime

Full job `2783861`:

- state: completed
- elapsed time: `00:32:51`
- windows: `289`
- mean fit time: `5.49 s`

### Artifacts

Pilot:

- `drafts/paper_outputs/e16_rolling_dependence_pilot.json`
- `drafts/figures/fig_rolling_dependence_dashboard_pilot.png`

Full:

- `drafts/paper_outputs/e16_rolling_dependence_full.json`
- `drafts/figures/fig_rolling_dependence_dashboard_full_abs.png`
- `drafts/figures/fig_rolling_dependence_dashboard_full_frac.png`

Cleaned paper version:

- `drafts/figures/fig_rolling_dependence_dashboard_full_frac_cross.png`
- `drafts/figures/fig_rolling_dependence_dashboard_full_frac_cross.pdf`

### Figure Refinements

Several rounds of cleanup were applied:

1. Ticker labels

- numeric asset indices were replaced by S&P100 ticker labels from the staged finance metadata

2. Tree-band view

- the middle panel can be rendered either in absolute contribution units or as a share of total dependence
- the share-of-total version is cleaner and more interpretable for the final figure

3. Persistent edge selection

- the bottom panel now favors edges that persist across many windows, not one-off spikes

4. Dual-share exclusion

- the trivial same-issuer dual-share pair `GOOGL-GOOG` was excluded from the paper version of Panel C
- this makes the cross-name structure much easier to read

### Main Results

Top panel:

- rolling total dependence varies substantially over time
- observed range: approximately `48.0` to `86.0` in mean log-copula contribution

Middle panel:

- average tree-band shares of total dependence:
  - Trees 1-3: `60.3%`
  - Trees 4-10: `10.7%`
  - Trees 11-99: `29.0%`
- deeper trees remain materially important and their share shifts over time

Bottom panel:

- after removing the dual-share pair, the persistent cross-name leaders are:
  - `MA-V`
  - `XOM-CVX`
  - `HD-LOW`
  - `BAC-JPM`
  - `DUK-SO`

### Decision

E16 was added to the paper as an appendix figure, with a light pointer from the main TC section.

Files updated in the draft:

- `drafts/vine_diffusion.tex`
- `drafts/vine_diffusion.pdf`

Placement:

- appendix figure with descriptive framing
- not elevated to a main-text centerpiece

Reason:

- the figure is useful and clean
- it demonstrates a distinct structured-monitoring capability
- but it is still more naturally a supporting interpretability figure than a core benchmark

## Additional Paper Changes Triggered During This Work

Two low-cost, reviewer-aligned improvements were added to the main text while E15 and E16 were being evaluated:

1. Depth-stability promotion

- the depth-wise real-edge stability result was promoted into the main real-edge subsection
- exact mean edge NLL values now appear in the main narrative:
  - shallow: `-5.451e-05`
  - mid: `2.278e-04`
  - deep: `6.575e-04`

2. MI-vs-`n` callout

- the fixed-checkpoint Clayton-chain varying-`n` result received a stronger one-sentence main-text callout
- the text now states that on the non-Gaussian `d=50` Clayton benchmark:
  - VDC relative MI error is `2.2%` at `n=30000`
  - VDC relative MI error is `3.3%` at `n=100000`
  - KSG remains above `56%` over the same range

These additions answer reviewer concerns more directly without forcing new main-text figures.

## Bottom Line

E15:

- useful diagnostic
- not paper-worthy in its current form
- keep out of the main paper

E16:

- successful and worthwhile
- now included as a cleaned appendix figure
- demonstrates a structured rolling dependence analysis that aligns with the paper's interpretability story

Net effect on the paper:

- no overclaiming
- stronger reviewer alignment
- better support for the claims about stability, interpretability, and fixed-checkpoint information estimation
