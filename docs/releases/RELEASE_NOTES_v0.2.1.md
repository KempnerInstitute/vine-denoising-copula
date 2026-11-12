# Vine Denoising Copula v0.2.1

Corrections to the NeurIPS 2026 reproduction directory found in the final camera-ready audit. The `vdc`
library code and the released model `hsafaai/vdc-denoiser-m64-v1` (checkpoint sha256
`8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f`) are unchanged from v0.2.0.

- Figure A4 (`figures/scripts/figA_mi_sample_size_cr.py`): relative MI errors are measured against the exact
  MI of the Clayton(theta = 3) pair copula, 0.6363 nats, instead of the saved artifact's 2048-cell grid value,
  0.6401 nats (0.6% too high). The paper's quoted errors change accordingly (for example 2.8% and 2.7% at
  n = 30,000 and 100,000, previously 2.2% and 3.3%).
- Figure A1 (`figures/scripts/figA_training_diagnostics_cr.py`): panel (b) is titled for what it plots, the
  row marginals after the training-time projection, not before it.
- Projection-iteration ablation (`tables/inputs/ipfp/ipfp_iter_ablation.json`, appendix table
  `tab_ipfp_iter_ablation`): rerun with the network's coordinate channels in the order used in training and
  deployment; the original script transposed them. A replay of the original script reproduces every saved
  error bitwise. The maximum marginal error at the deployed K = 30 is 1.8e-4 (previously reported 5.4e-5).

Verification: all ten figures regenerate pixel-identically and all 25 generated tables byte-identically
against the corrected camera-ready paper.
