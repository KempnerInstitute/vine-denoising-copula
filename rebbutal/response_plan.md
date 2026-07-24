# NeurIPS 2026 rebuttal plan — Submission 33494 (VDC)

Prepared 2026-07-24. Phase 1 (author response only) runs Jul 23–27; per-review
limit 10,000 chars, markdown, **no links, no paper revisions, no identity
leaks** (do not mention arXiv, GitHub, or the public checkpoint).

## 1. Overall read

Scores: DzsK **2 (reject, conf 4)** · VcZn **4 (borderline accept, conf 4)** ·
gb3Z **4 (borderline accept, conf 3)** · LCT8 **4 (borderline accept, conf 3)**.
Meta (AC Htv7): "mostly borderline"; the AC's single named lever is
**articulating what is gained versus diffusion/flow models**.

This is a very winnable position. Three of four reviewers are at
borderline-accept and explicitly say what would move them (VcZn: "I would like
to see suggestions on practically most of the comments on clarity before I can
confidently advocate"; LCT8 asks five concrete questions). DzsK is the outlier
reject, but their own scores say Originality 4/4 — the objections are
clarity, definitions, run counts, and missing theory, i.e., addressable by
response + camera-ready commitments, not by disputing the contribution.

**The decisive fact: we already ran almost every experiment the reviewers ask
for** (between submission and now, E25–E30). LCT8's five concerns match our
internal red-team almost verbatim, and each has a completed experiment with
numbers we can quote. The response strategy is: lead with new numbers, not
promises.

## 2. Coverage matrix — reviewer point → what we have

| # | Reviewer point | Status | Evidence to quote |
|---|----------------|--------|-------------------|
| 1 | LCT8 Q1 / gb3Z / DzsK: headline bivariate accuracy is in-distribution; test on families excluded from training | **DONE (E26)** | 9 OOD cases (BB6, BB8, Tawn, Plackett, 90° rotations — none in the zoo): ISE parity with per-case-refit TLL (9.61e-5 vs 9.58e-5), better |Δτ| in 7/9 (0.013 vs 0.018), ~half the MI error (0.025 vs 0.054 nats) |
| 2 | LCT8 Q2 / meta: GPU vs single-thread CPU confound; matched-compute | **DONE (E25) + correction** | The submitted footnote *misstates* the setup: run records show pyvine-TLL was fit with 8–16 threads, so reported speedups are already GPU-vs-multithreaded-C++. Full sweep: at d=100/200/500, TLL takes 1466/5897/>22000 s (1 thread), 203/776/4726 s (8), 120/426/3102 s (16) vs VDC 44/173/1082 s → 2.5–4.6× vs multithreaded, 21–34× vs single-thread. Break-even ≈478/125/59 vines (multithreaded baseline), ≈46/11 (single-thread) |
| 3 | LCT8 Q3: UCI near independence floor; "is VDC's +NLL mildly harmful structure?" | **DONE (E27)** | Gaussian-copula TC after PIT: Gas 0.001, Hepmass 0.002, Credit 0.006, Miniboone 0.017 nats (vs Power 1.881; mean |τ|≈0.008 vs 0.409). Concede the reading: these four sets cannot discriminate copula methods; VDC's ≤2e-3 bits/dim overhead is the cost of fitting structure where there is ~none. Discriminative benchmarks: Clayton vines, S&P100 |
| 4 | LCT8 Q4 / DzsK: DPI 0/60 vs 2/60 not significant; only 5 trials; τ unjustified; 4-D additivity may disadvantage KSG | **DONE (480-trial scale-up + τ-sweep)** | 10/480 (2.1%) vs 25/480 (5.2%), Fisher p=0.015; τ-sensitivity (new, computed from the same runs): VDC fewer violations at *every* τ, significant for τ∈{0,1e-4,3e-4,1e-3,3e-3} (p=.005/.017/.035/.015/.025); at τ=0: 13 vs 32. Concede the 4-D point for additivity: KSG's dimensional bias contributes; offer learned-structure additivity rerun; also disclose the D-vine block-structure asymmetry ourselves |
| 5 | LCT8 Q5 / gb3Z: conditional-task diagnosis; can it impute/predict? | **DONE (E30)** | Exact inverse-Rosenblatt conditional sampling (per-pattern observed-first D-vine refits): Gas RMSE 1.151→1.001 (best simple baseline 0.993), Credit 1.415→1.018 (best 1.013) → the reported deficits were ~entirely the approximate importance-weighting *procedure*; Power −20% vs kNN persists (genuine model-side gap). This *is* the requested error decomposition: procedure ≫ model on 2/3 datasets |
| 6 | LCT8 Q5b: deep-tree near-zero NLL = weak dependence or default-to-independence? wants controlled deep vine | **PARTLY DONE + small job** | Fig 3(a) is exactly this: d=8 Clayton vine with known nonzero dependence at *all* trees; per-edge estimated-vs-true MI across trees (MAE 0.0041 nats), knockout corr 0.932. Offer: per-tree held-out edge-NLL on the d=100 Clayton vine (cheap job) to extend to depth |
| 7 | Meta/gb3Z: what is gained vs flows / vs per-edge nonparametric | **DONE (E28, E29 + capacity study)** | (i) modularity → exact MI/TC decomposition, h-functions, conditional sampling — a monolithic flow exposes none; (ii) small-n prior regularization: S&P100 chrono (n_train=1356, d=100): VDC −0.419 vs TLL −0.293 (overfits below the Gaussian copula −0.337) vs RealNVP −0.047 (collapses), paired p<0.001; capacity study shows bigger flows don't close the Clayton-vine gap; (iii) E29: disabling the IPFP projection destroys validity (Clayton "NLL" exceeds the log m information ceiling ~10×; UCI degrades 0.43–0.85 bits/dim) — the gain over a generic amortized regressor is the *valid-copula composition* |
| 8 | VcZn: sampling experiments? | **MACHINERY EXISTS + quick job** | Algorithm A2 (inverse-h sampling) is implemented; the rolling dashboard and E30 already sample from fitted vines. Quick job: sample-quality metrics (τ-matrix / tail-dependence of simulated vs held-out data) on 1–2 datasets |
| 9 | VcZn: d>2 pretraining? | **ANSWER ONLY** | Bottleneck: histogram grid scales as m^d and IPFP marginal projection generalizes but validity constraints multiply; the vine factorization is precisely the mechanism that composes 2-D blocks into d dims; amortizing 3-D blocks (for non-simplified vines) is natural future work |
| 10 | VcZn ~25 line-level clarity items; DzsK definitions/cross-refs/zoo spec | **MOSTLY ALREADY FIXED in our revision** | Item-by-item acceptance list; nearly all were independently fixed in our working revision (citations for U-Net/FiLM, clip notation, corruption/one-shot definitions, IPFP forward-definition, Fig 2 caption + error-bar defs, appendix roadmap + claims–evidence map, zoo parameter ranges); commit for camera-ready. L218 RealNVP claim: agree, scoped to Hepmass/Credit/Miniboone (Power favors RealNVP) — exact fix already drafted |
| 11 | DzsK: 3–5 runs too few; A7 stds large | **PARTLY DONE + job** | DPI now 480 trials; A7's ±std is dispersion *across heterogeneous copula cases*, not run noise (explain); NEW JOB: 100-replicate bivariate suite with paired per-case win-rates + bootstrap CIs (each fit costs ms — trivially feasible) |
| 12 | DzsK: no consistency/rates theory | **ANSWER + commitment** | Honest scope: same open-theory status as the amortized-inference/PFN class; what *is* guaranteed: strict positivity → Prop 1 uniqueness, exact copula validity after projection, exact vine recursion given edges, MI ceiling log m. Commit to a "what is and is not guaranteed" subsection; do not promise a consistency theorem |
| 13 | gb3Z: name "Vine Denoising Copula" misleading | **CONCEDE** | Agree it is an estimation procedure; offer to foreground "amortized pair-copula estimation for simplified vines" and keep VDC as the estimator name (or rename — AC's preference) |
| 14 | gb3Z: failure diagnostics for OOD edges | **PARTIAL + answer** | The histogram→prediction residual and the pre-projection marginal deviation are computable per edge at inference (both observable without ground truth); OOD table shows graceful behavior; commit to a diagnostics paragraph |
| 15 | VcZn: high sample sizes for 2-D MI; TLL scaling with n | **ANSWER** | Fig A2 sweeps n=1e3→1e5 at fixed checkpoint (VDC best from n≥1e4); TLL fit time does grow with n (kernel evaluation); VDC histogramming is O(n) with constant grid |

## 3. New analyses to run (priority order, all feasible in Phase 1–2 window)

1. **DONE — DPI τ-sweep** (computed from existing 480-trial records; table above).
2. **Bivariate 100-replicate suite** (GPU, ~hours): per-family 100 fresh
   replications; report mean±CI and paired win-rate vs TLL. Kills DzsK's
   run-count objection at its strongest point.
3. **Per-tree edge-NLL on deep synthetic Clayton vine** (d=100; data/fits
   exist): direct answer to LCT8's "controlled deep vine with known nonzero
   deep-tree dependence".
4. **VDC-on-CPU timing** (one job): completes LCT8's matched-compute request
   in the second direction.
5. **Sampling quality** (quick): simulated-vs-held-out τ matrices + tail stats
   on S&P100/Gas (answers VcZn/gb3Z simulation question with numbers).
6. **Additivity with learned structure** (moderate): removes both the oracle
   D-vine asymmetry we disclosed and LCT8's KSG-dimensional-bias concern.

## 4. Draft responses (v1 — trim to <10k chars each when posting)

### 4.0 To AC Htv7 (official comment, after reviewer responses are posted)

> We thank the AC. Per the meta-review, our responses focus on (i) what is
> gained relative to diffusion/flow models, with new matched-compute and
> small-sample evidence, and (ii) the presentation concerns. Briefly: a
> monolithic flow gives one d-dimensional density; it exposes no pair-copulas,
> no h-functions, and no edge/tree information decomposition, so quantities
> central to dependence analysis (edge-wise MI, tree-wise TC shares,
> conditional simulation) require separate machinery. VDC keeps the classical
> vine object — every downstream vine operation remains available — while
> removing its per-edge fitting bottleneck. Empirically this matters in two
> regimes: (a) non-Gaussian dependence: on the Clayton-vine benchmark larger
> RealNVP variants do not close the gap (App. C.3); (b) small-sample real
> data: on S&P100 (n_train=1356, d=100) under a new, stricter chronological
> split, VDC reaches −0.419 bits/dim vs −0.293 (pyvine-TLL, which overfits
> below the trivial Gaussian copula at −0.337) and −0.047 (RealNVP), all
> paired-bootstrap p<0.001 — the pretrained synthetic prior acts as a
> regularizer exactly where flexible per-dataset estimators overfit. A new
> ablation also shows the copula projection is load-bearing rather than
> cosmetic: disabling it produces a non-normalized object whose apparent
> likelihood exceeds the information-theoretic ceiling of the representation
> ~10×, and degrades UCI NLL by 0.43–0.85 bits/dim. We view VDC as
> complementary to flows (it can also serve as the dependence module under
> separately-modeled marginals) and have added this articulation to the
> revision.

### 4.1 To Reviewer LCT8 (highest-quality review — answer all 5 with numbers)

Structure (each ≈1–1.5k chars):
1. **In-distribution concern — new OOD experiment.** Quote E26 numbers (9
   excluded families; parity ISE, better τ/MI). State plainly: Table 1
   measures in-distribution amortization quality; the revision says so.
2. **Matched compute — correction + full thread sweep.** Own the footnote
   error: "on re-examining run records, the baselines were fit with
   vinecopulib's multithreading enabled (8–16 threads), not 1 thread as the
   footnote stated; the submitted speedups are therefore already
   GPU-vs-multithreaded-C++." Then the sweep table + both break-evens. Offer
   VDC-on-CPU numbers in Phase 2.
3. **UCI near-independence — concede + quantify.** E27 TC table; agree with
   their reading including "slightly worse than independence"; the table's
   role is "does no harm at the floor"; discriminative sets are Clayton +
   S&P100 (add chrono result).
4. **Self-consistency thinness — 480 trials + τ-sweep + concession.** New
   significant result incl. τ=0; concede the 4-D additivity bias point and
   the oracle-structure asymmetry (disclose proactively); offer
   learned-structure rerun.
5. **Conditional diagnosis — E30 decomposition + deep-vine evidence.** Exact
   sampling isolates procedure vs model (numbers); Fig 3(a) is the controlled
   deep vine (per-tree edge-MI vs truth, MAE 0.004); offer per-tree edge-NLL
   at d=100. On IPFP distortion: cite the IPFP-sensitivity table (K=15→100
   leaves NLL/PIT unchanged) + tail-loss training.
Close: list the four limitation-gaps they enumerate; all four are now stated
explicitly in the revision.

### 4.2 To Reviewer DzsK (reject → aim for 3–4; tone: everything concrete)

1. **Runs/statistics.** DPI: 5→40 trials/config (480 total), significant, plus
   full τ-sensitivity (their exact question) — include both tables. A7: the
   ±std is across-case heterogeneity, not run noise (cases span weak→strong
   dependence); new 100-replicate suite with paired win-rates underway —
   will post in discussion phase.
2. **Definitions/spec.** Zoo parameter ranges, held-out composition, counts:
   give the actual numbers in the response (from App. C.2 + generator code)
   and commit to a consolidated "benchmark specification" appendix table.
3. **Cross-referencing.** Accept both examples (Fig 5 ↔ A5/A14/A15; A7 ↔
   Table 1); note the revision adds an appendix roadmap + a claims→evidence
   map table; captions now name their appendix sources.
4. **Theory.** Honest scoping (see matrix row 12); enumerate what IS exact;
   note this open-theory status is shared by the amortized-inference class;
   commit to a formal "guarantees and non-guarantees" subsection. Push back
   gently on "technical flaws": no reviewer identified an incorrect claim;
   LCT8 verified the formal content where checked.
5. **Readability.** Point to the concrete restructuring done (roadmap,
   glossary of acronyms at first use, algorithm boxes) and VcZn item list.

### 4.3 To Reviewer VcZn (borderline accept, wants clarity; likely to raise)

- Open: thank; all ~25 line-item points accepted; give a compact itemized
  table: item → fix (one line each; most already implemented in the working
  revision — citations, math notation for clip/coordinates/output head,
  corruption + one-shot definitions at first use, IPFP forward-reference
  fixed, Fig 2 caption rewritten with error-bar definitions, algorithm
  referenced from 3.4, seeds explained, VaR protocol described where first
  mentioned, "suite" defined).
- L218: agree — claim now scoped ("on Hepmass, Credit, Miniboone"; Power
  favors RealNVP).
- Sampling: yes — inverse-h sampling is implemented (Alg. A2); used by the
  rolling dashboard and by a new exact-conditional-imputation experiment
  whose numbers we quote (E30); offer sample-quality stats in Phase 2.
- Sample sizes for 2-D MI: point to fixed-checkpoint n-sweep (A2 figure)
  down to n=1e3; TLL fit time indeed grows with n, VDC histogramming is O(n).
- d>2: answer per matrix row 9.
- Tagasovska + deep-Archimedean citations: accept, will recategorize/add.

### 4.4 To Reviewer gb3Z (borderline accept; positioning)

1. **What is gained.** Not just faster approximation: (i) the per-edge
   estimator is *strictly more accurate* than the standard nonparametric
   estimator on OOD families too (E26 numbers); (ii) speed changes what is
   *practical*: repeated refits (rolling windows), structure sweeps, d≥200;
   (iii) small-n regularization (S&P100 story, numbers); (iv) everything a
   classical vine supports remains available because the object IS a
   classical vine.
2. **What is lost.** Candid list: grid resolution (log m MI ceiling, stated),
   no per-edge asymptotic theory (vs kernel estimators), pretraining cost
   (break-even numbers), simplified-vine scope.
3. **Probabilistic completeness.** Simulation: yes (Alg. A2; dashboard).
   Conditional prediction/imputation: new E30 exact-sampling result —
   procedure-vs-model decomposition with parity on 2/3 datasets. Tail
   dependence: |Δλ_U| in Table 1 + OOD table; tail-switch stress test.
   Higher-tree dependence: Fig 3(a) per-tree recovery vs ground truth.
4. **Failure diagnostics.** Observable per-edge signals (pre-projection
   marginal deviation, histogram-prediction residual); commit to a
   diagnostics paragraph.
5. **Name.** Agree; we now foreground "amortized pair-copula estimation for
   simplified vines" and are happy to adjust the method name if preferred.

## 5. Process notes

- **Do not** reference arXiv/GitHub/HuggingFace or "our public revision" —
  anonymity. Present all new numbers as "experiments run during the response
  period" and revision text as "the revision".
- The camera-ready-ready revision already exists locally (all promised fixes
  are implemented) — commitments in the responses are pre-satisfied, which
  lets us phrase them as done ("the revision now states/defines/…").
- Post responses early in Phase 1 window; keep ~1k chars of headroom per
  response for Phase 2 follow-ups.
- The paper/supplement must NOT be revised on OpenReview during this period.
