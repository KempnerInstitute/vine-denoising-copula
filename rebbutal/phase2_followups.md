# Phase-2 follow-up snippets (post Jul 27 – Aug 3)

Ready-to-post continuations for the discussion phase. Each was promised in the
Phase-1 response as "running; will post in discussion." Paste under the
corresponding thread. Character counts are small; they fit comfortably in the
remaining headroom of each thread.

**UPDATE (Jul 24, later):** per the author's decision to use each response's own
10k budget, ALL completed results (E31, E32, E34, E35 finals; E33 calibration)
are now folded directly into the initial responses — snippets 1, 2, 3, 5 below
are superseded (kept for reference / in case a reviewer asks follow-ups on
them). ALL experiments are complete and folded into the initial responses; the snippets below are retained purely as reference material for Phase-2 follow-up conversations.

Status of the supporting jobs:
- E31 100-replicate bivariate suite — **COMPLETE, folded into responses**
- E35 learned-structure additivity — **COMPLETE, folded into responses**
- E34 sampling quality — **COMPLETE, folded into responses** (finals: Gas
  0.016/0.017/floor 0.018; Power 0.020/0.036/0.011; S&P100 0.047/0.192/0.039)
- E32 deep-tree recovery — **COMPLETE, folded into responses**
- E33 VDC-on-CPU timing — **COMPLETE, folded into responses** (VDC CPU-8-threads beats TLL-8-threads on matched hardware: 181.8 vs 203.3 s at d=100, 729.8 vs 775.6 s at d=200; RTX 35/146 s)

---

## 1. For Reviewer DzsK (also cite in LCT8 thread if useful) — replication scale

> As promised, the large-replication bivariate benchmark is complete: 8
> in-distribution families × **100 paired replications each** (fresh n=2000
> sample per replication; VDC and pyvine-TLL fit on the identical sample).
> Pooled over 800 paired replications, VDC wins on **87.9% [85.4, 90.0]**
> (Wilson 95% CI) of replications for density ISE and **88% [85, 90]** for MI
> error; suite-mean ISE 3.3e-5 vs 5.2e-5, MI error 0.026 vs 0.057 nats.
> Per-family win rates are 97–100% for ISE on 6/8 families and 69–100% for MI
> error on 8/8, each with its own Wilson CI; the one exception is BB7 density
> ISE, where TLL wins 76% of replications (VDC still wins BB7 MI error 92%) —
> we report it as-is. Within-case replication stds are now separated from
> across-case spread, resolving the Table A7 ambiguity: e.g., Clayton ISE
> across 100 replications is 1.1e-4 ± 2.2e-6 (2% relative), i.e., the
> replication noise is small; the large A7 stds were across-case
> heterogeneity, as stated in our response.

## 2. For Reviewer LCT8 (Q4 thread) — learned-structure additivity

> The learned-structure additivity rerun is complete (120 paired trials, same
> generator and n=4000 as the paper, all three estimators on the same data
> per trial): additivity error is **0.024 ± 0.020 nats with oracle block
> structure and 0.024 ± 0.020 with fully learned Dißmann R-vine structure**
> (statistically indistinguishable), versus **0.039 ± 0.029 for KSG**. So the
> oracle-structure asymmetry we disclosed does not drive the additivity gap:
> with structure learned from data, symmetric to what KSG receives, the
> contrast is unchanged. One further transparency note: in inspecting this we
> found the paper's "oracle" arm already learned the D-vine order by a greedy
> |τ| heuristic — the hand-specified part was only the D-vine topology class —
> and the fully learned R-vine matches it anyway. We agree KSG's dimensional
> bias remains a *contributing* factor to the 4-D LHS and will keep that
> caveat in the revision.

## 3. For Reviewers VcZn and gb3Z — sampling quality (UPDATE WITH FINALS)

> The unconditional sampling-quality benchmark [is complete / preliminary]:
> we simulate n=5000 from each fitted vine and compare pairwise dependence
> against the held-out test set, with a bootstrap of the training data as the
> attainable noise floor. On Gas (d=8): pairwise Kendall-τ matrix RMSE is
> **0.027 for VDC samples vs 0.027 for pyvine-TLL samples vs 0.022 for the
> real-data bootstrap floor**; lower/upper tail-concentration MAE 0.042/0.034
> (VDC) vs 0.034/0.031 (TLL) vs 0.028/0.042 (floor). [FINAL: add power +
> S&P100 rows, 3 seeds ± sd.] I.e., data simulated from the fitted vine
> reproduces held-out dependence essentially as well as resampled real data —
> sampling is a first-class operation (inverse h-functions, Algorithm in the
> appendix), not an afterthought.

## 4. For Reviewer LCT8 (Q2 thread) — VDC on CPU (PLACEHOLDER, fill when E33 lands)

> Completing the matched-compute picture in the other direction: VDC fit on
> **CPU only (8 threads)** takes [X] s at d=100 and [Y] s at d=200 on the
> same benchmark (held-out NLL identical to the GPU fits within tolerance),
> versus multithreaded pyvine-TLL at 203/776 s (8 threads) and 120/426 s (16
> threads). On a mid-range GPU ([name]), VDC takes [Z] s at d=100/200. So
> [conclusion depending on numbers: even without any GPU, amortized fitting
> is within ~[k]× of multithreaded vinecopulib / the advantage requires a
> GPU, which we state plainly].
> (Smoke calibration: 75 ms/edge on CPU at n=5000 → ~373 s at d=100.)

## 5. For Reviewer LCT8 (Q5 thread) — deep-tree recovery (PLACEHOLDER, fill when E32 lands)

> The controlled deep-vine experiment is complete. Important note first: we
> found the paper's synthetic Clayton benchmark generator is first-order
> Markov, i.e., its true deeper trees are exactly independent — so it cannot
> test deep-tree recovery (this does not affect any submitted claim, which
> concerned density and timing, but we state it for full transparency). We
> therefore built a simplified D-vine with **Clayton(θ=2) at all 29 trees**
> (d=30; true per-edge dependence 0.432 nats at every depth, verified by
> quadrature, Monte Carlo, and grid integration) and measured per-tree
> recovered dependence at n_train=20,000:
> [TABLE: depth bucket | true | VDC | TLL | VDC/true]
> [Honest conclusion template: (a) if recovery holds: VDC recovers ≥[x]% of
> true dependence through tree [t], vs TLL's [y]% — deep near-zero NLL on UCI
> therefore reflects the data, not estimator default-to-independence; (b) if
> recovery decays: both estimators attenuate deep dependence (VDC less than
> TLL); we will state this as a shared finite-sample limitation of simplified
> vine estimation at depth and scope the UCI depth claim accordingly.]

---

**Process notes:** all snippets are anonymity-safe (no links/names). If a
reviewer replies before a job lands, respond to their message and note the
pending experiment will follow within the phase. After posting, record the
posted text verbatim in this directory for the camera-ready audit trail.
