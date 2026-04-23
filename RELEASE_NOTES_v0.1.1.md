# Vine Denoising Copula v0.1.1

Release `v0.1.1` updates the first public VDC release with the cleaned release branch state on `main`.

This release includes:

- the public `vdc` Python package and CLI
- the released pretrained checkpoint manifest for `vdc-denoiser-m64-v1`
- package and documentation cleanup for public release
- released-model verification scripts and reports
- examples and user-facing documentation
- release notes and release metadata aligned with the public repository name

Released model:

- Hugging Face: `hsafaai/vdc-denoiser-m64-v1`

Repository:

- GitHub: `KempnerInstitute/vine-denoising-copula`

Notes:

- the released checkpoint assumes continuous marginals and pseudo-observations in `[0,1]`
- the current public release centers the frozen `m=64` checkpoint
- paper workspace files remain under `drafts/` and are intentionally separate from the public package surface
- `v0.1.1` supersedes the earlier `v0.1.0` tag as the recommended public release point
