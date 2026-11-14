# VDC 0.4.0

This release prepares standard PyPI installation and separates pretrained
inference from optional training and baseline dependencies.

- The base package supports pair-density estimation, vine fitting, likelihoods
  and sampling without pyvinecopulib, Matplotlib, HDF5, Hydra or scikit-learn.
  Install `[train]` for training and synthetic generators, `[baselines]` for
  the qualified pyvinecopulib adapter, `[viz]` for plots and `[data]` for dataset
  downloads. The training and baseline extras retain pyvinecopulib 0.7.x;
  general 1.x compatibility is not claimed.
- Model construction now lives in `vdc.models.factory`. The trainer retains its
  existing `build_model` entry point. Constructor calculations, parameter names,
  checkpoint weights and inference defaults are preserved.
- The probability-mass histogram is named `scatter_to_mass_hist`; its old name
  remains an alias. Unit tests and a public diagnostic distinguish it from the
  density histogram required by pretrained inference (GitHub issue #8).
- Python 3.10 or later is required. CI covers Python 3.10–3.13 and checks a
  minimal installed wheel on Linux, macOS and Windows. Release checks exercise
  all three published checkpoints, small vines, persistence and NumPy sampling
  fallback outside the source checkout.
- GitHub Actions builds packages once, publishes them to TestPyPI, downloads
  and verifies their hashes, runs inference from the downloaded wheel, then
  publishes the same distributions to PyPI using Trusted Publishing. GitHub
  Releases retains the distributions and their checksums.

Install this project as `vine-denoising-copula`; the PyPI project named `vdc`
is unrelated and uses the same import namespace. Keep them in separate virtual
environments. Checkpoints remain on Hugging Face and are hash-verified by the
loader. The paper's original source pins and reproduction environments remain
unchanged.
