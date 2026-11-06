# Vine Denoising Copula v0.2.2

Final camera-ready reproduction corrections. The estimator implementation, checkpoint and fitted-model outputs are unchanged from v0.2.0; this patch includes the v0.2.1 figure and ablation corrections.

- Figure A5 calls the portfolio-normal VaR baseline **Gaussian returns**. It is not a fitted Gaussian copula; no numerical results change.
- The projection-iteration table uses a synchronized H100 rerun. Each K has ten warmups and 100 randomized-order repetitions, with CUDA synchronization immediately before and after each whole-batch projection. Median K=30 cost is 1.23 ms, and maximum marginal error rounds to 1.8e-4. The previous timer stopped before GPU completion.
- `paper/neurips2026/runners/ipfp_iterations.py` provides the portable timing protocol. The saved input includes all 100 times per K and the source manifest links the complete generation record.
- The table inventory consistently counts 25 generated tables.

The local v0.2.1 tag is preserved. This patch is prepared locally; publication requires the author's instruction.
