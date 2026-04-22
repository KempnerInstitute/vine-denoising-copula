# Vine Denoising Copula v0.1.0

First public release of Vine Denoising Copula (VDC).

This release includes:

- the public `vdc` Python package and CLI
- the released pretrained checkpoint manifest for `vdc-denoiser-m64-v1`
- vine fitting for D-vine, C-vine, and R-vine workflows
- IPFP projection for valid discrete copula densities
- released-model verification scripts and reports
- examples and user-facing documentation

Released model:

- Hugging Face: `hsafaai/vdc-denoiser-m64-v1`

Repository:

- GitHub: `KempnerInstitute/vine-denoising-copula`

Notes:

- the released checkpoint assumes continuous marginals and pseudo-observations in `[0,1]`
- the current public release centers the frozen `m=64` checkpoint
- paper workspace files remain under `drafts/` and are intentionally separate from the public package surface
