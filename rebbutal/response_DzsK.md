We thank the reviewer for the detailed criticism. We note the review rates Originality 4/4 and identifies no incorrect technical claim; the objections concern statistical rigor, definitions, organization, and theory scope. We address each with new experiments run during the response period and concrete revision commitments (all already implemented in our working revision).

**1. "3–5 runs is too few; substantially more Monte Carlo runs."**

We agree where it matters and have acted on it:

- **Self-consistency (App. D.7, the reviewer's example):** scaled from 5 trials/config (60 DPI trials) to 40 trials/config (**480 DPI trials per estimator**). Result: VDC 10/480 (2.1%) DPI violations vs KSG 25/480 (5.2%), Fisher exact p=0.015 — the contrast is now statistically significant, not merely suggestive. Additivity error at the same scale: 0.027 vs 0.044 nats.
- **Tolerance τ (the reviewer's specific question):** we computed the violation rates over the full tolerance range rather than the single τ=1e-3:

| τ (nats) | VDC | KSG | Fisher p |
|---|---|---|---|
| 0 | 13/480 (2.7%) | 32/480 (6.7%) | 0.005 |
| 1e-4 | 13/480 | 29/480 | 0.017 |
| 3e-4 | 13/480 | 27/480 | 0.035 |
| 1e-3 | 10/480 | 25/480 | 0.015 |
| 3e-3 | 5/480 | 16/480 | 0.025 |
| 1e-2 | 0/480 | 4/480 | 0.12 |

The ordering (VDC fewer violations) holds at every τ including the strictest τ=0, and is significant for all τ≤3e-3. The revision reports this sensitivity table instead of a single threshold.
- **Bivariate benchmark at the requested replication scale — complete:** 8 in-distribution families × **100 paired replications** each (fresh n=2000 sample per replication; VDC and pyvine-TLL fit on the identical sample; amortized fits cost milliseconds, so large replication counts are exactly the regime our method makes cheap). Pooled over 800 paired replications: VDC wins **87.9% [85.4, 90.0]** (Wilson 95% CI) of replications on density ISE and **88% [85, 90]** on MI error; suite means ISE 3.3e-5 vs 5.2e-5, MI error 0.026 vs 0.057 nats. Per-family: ISE win rates 85–100% on 7/8 families with the one exception reported as-is (BB7 density ISE: TLL wins 76% of replications, though VDC still wins BB7 MI error 92% [85, 96]).
- **On Table A7's large stds:** the replicate suite now separates the two variance sources. Within-case replication noise is small (e.g., Clayton ISE over 100 replications: 1.1e-4 ± 2.2e-6, ~2% relative); the large A7 stds are dispersion **across heterogeneous held-out cases** (weak to strong dependence, light to heavy tails), where std exceeding the mean is legitimate for a heavy-tailed case distribution. Captions will state the dispersion source everywhere.

Where 3 seeds remain (e.g., the d=500 vine sweep at ~1100–2200 s per fit), the across-seed std is 0.02–0.4% of the mean (reported in the tables); we will note this justification explicitly.

**2. "Missing mathematical definitions; benchmark specifications."**

We accept this and can give the requested specifics here (values taken directly from the released streaming generator's defaults). Copula zoo: Gaussian ρ ~ U(−0.95, 0.95); Student-t ρ ~ U(−0.95, 0.95), ν ~ U(2.5, 30); Clayton θ ~ U(0.2, 12); Gumbel θ ~ U(1, 10); Frank θ ~ U(−12, 12); Joe θ ~ U(1, 10); BB1 (θ, δ) ~ U(0.1, 5)×U(1, 5); BB7 (θ, δ) ~ U(1, 5)×U(0.1, 5); rotations {90°,180°,270°}; independence; conditional pair-copula examples; structured non-monotone patterns; 2–4-component mixtures; per-example sample size n ~ LogUniform(200, 8000); training streams fresh examples (≈6.1M histogram–target pairs at the deployed checkpoint). Held-out synthetic suite: fixed benchmark cases per family with parameters disjoint from training draws and fresh samples; the exact case list, counts, and parameter values are enumerated in a new consolidated "benchmark specification" appendix table in the revision, together with formal definitions of every symbol at first use (including the histogram H, corruption operator, one-shot inference map, and each loss term with its weights). We are also happy to answer any specific undefined symbol here — the review's variable names did not render in OpenReview, so we could not identify which specific symbols were meant; please let us know and we will give the definitions directly.

**3. "Organization and cross-referencing."**

Both examples are correct and both are fixed: figure captions now name their supporting appendix tables (Fig. 5 ↔ the MI, method-comparison, and TC-scaling tables), and the Table 1 ↔ Table A7 relationship is stated in both captions. More broadly the revision adds (i) an appendix roadmap paragraph, (ii) a claims→evidence map table listing every main-text claim (including the negative results) with its supporting artifact, and (iii) a four-theme organization of the experimental appendix (edge-level diagnostics / full-vine density / information estimation / downstream stress tests) with cross-references. We believe this addresses the navigability concern structurally rather than cosmetically.

**4. "Almost no theoretical analysis (consistency, rates, sample complexity)."**

This is a fair description of scope, and we want to be precise about what is and is not guaranteed. Guaranteed: (i) strict positivity of the network output (clipped exponential), hence Proposition 1 applies and the projected grid is the unique KL-projection with exactly uniform marginals — every deployed edge is a valid pair-copula by construction; (ii) given the edges, vine likelihood evaluation, h-recursion, sampling, and the TC decomposition are exact for the fitted model; (iii) the representation's information ceiling (log m) is stated, so MI estimates are interpretable. Not guaranteed: consistency and rates of the amortized estimator itself — this is the same open-theory status as the broader train-once/amortized-inference class (prior-fitted networks, neural processes, amortized SBI), where finite-sample theory is an active open problem. We do not claim otherwise anywhere in the paper; the revision adds an explicit "guarantees and non-guarantees" subsection so readers need not infer this. Our empirical substitutes are targeted at exactly the failure modes theory would bound: out-of-distribution family tests (new; parity with per-case nonparametric refits on 9 excluded families), a fixed-checkpoint sample-size sweep (n=1e3–1e5), depth-stratified recovery against analytic ground truth, and the now-significant self-consistency battery above.

**5. "Difficult to read."**

Together with Reviewer VcZn's line-level list (which we accept essentially in full — citations for U-Net/FiLM, explicit min/max form of clip, forward-definition of IPFP before first use, definitions of "corruption," "one-shot," "suite," seeds, and the VaR protocol at first mention), the revision defines every acronym at first use, moves an algorithm box into the method section, and restructures as described in §3 above. We would welcome the reviewer pointing at any remaining passage they find impenetrable — we will rewrite it.

Given that (a) the statistical concerns are now addressed with 8× the trial count and a full tolerance-sensitivity analysis, (b) the specification and organization requests are implemented, and (c) the theory scope is stated honestly and matches the accepted norm for amortized-inference methods, we hope the reviewer will consider whether "technical flaws, weak evaluation, inadequate reproducibility" still describes the paper, and we remain eager to run any additional check they consider decisive.
