# Release guide

## Verify the candidate

1. Start from the maintained `main` branch and record its commit.
2. Update the package version in `pyproject.toml` and `vdc/__init__.py`, the
   software version/date in `CITATION.cff`, and the changelog.
3. Run the complete test suite, build a wheel and sdist, and test imports and
   command-line entry points from the installed wheel.
4. Run the pretrained example with the public model and verify its hash.
5. Build the documentation and check local links. When `paper/` changed,
   regenerate its figures and tables as described in
   [its guide](../paper/README.md) and confirm that they still match the paper.
6. Inspect the public tree and archives for credentials, private paths, internal
   notes, datasets and checkpoints. Update `PUBLIC_RELEASE_FILES.json` when the
   curated source contents change.

## Configure PyPI once

Use accounts with two-factor authentication on PyPI and TestPyPI (separate
accounts). In each account's **Publishing** page, register a pending GitHub
Trusted Publisher:

| Field | PyPI | TestPyPI |
|---|---|---|
| Project | `vine-denoising-copula` | `vine-denoising-copula` |
| Owner | `KempnerInstitute` | `KempnerInstitute` |
| Repository | `vine-denoising-copula` | `vine-denoising-copula` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment | `pypi` | `testpypi` |

Configure the matching GitHub environments and restrict release-tag creation
and environment access to maintainers. The workflow requests short-lived OIDC
credentials only in the publishing jobs; no permanent PyPI upload token is
needed. The first successful upload creates the project. A pending publisher
does not reserve its name.

- [PyPI publishing settings](https://pypi.org/manage/account/publishing/)
- [TestPyPI publishing settings](https://test.pypi.org/manage/account/publishing/)
- [Trusted Publishing documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)

## Publish

After the candidate and its GitHub CI pass, create and push a new `v*` version
tag. The workflow verifies that the tag matches the package version, runs the
full suite, builds and checks the wheel and source distribution, and exercises
all three released checkpoints in a fresh base-only installation. Its record
includes the checkpoint hashes, dependency versions and persistence/fallback
checks. The published model defaults and weights are not changed by packaging.

The workflow uploads the verified distributions to GitHub Releases and TestPyPI.
It then downloads both distributions from TestPyPI and compares their SHA-256
hashes against the build artifacts, installs the downloaded wheel with
**dependencies from ordinary PyPI**, and exercises pretrained inference again.
Only after this succeeds does it publish the same files to PyPI, whose published
hashes are checked as the final step. Publication status must be verified on
both indexes; a GitHub release alone does not mean the PyPI upload succeeded.

If account configuration blocks publishing, fix the Trusted Publisher and
rerun **failed jobs**, preserving the successful build's artifacts. Do not
rebuild or replace the same version. Workflow artifacts are retained for 30
days. Manual dispatch is supported only on a matching version tag.

Never move a historical tag or replace a checkpoint under an existing model ID.
The paper's saved outputs and the figure archives attached to v0.2.4–v0.2.6
retain their generation pins when later software versions are published.

## Installed package check

Run on a machine with several GB of available RAM, outside the checkout:

```bash
python /path/to/source/scripts/check_installation.py --require-minimal --all-models --output installed-models.json
```

The base environment must not contain pyvinecopulib, Matplotlib, HDF5, Hydra or
scikit-learn. This check loads public checkpoints, estimates pair densities,
fits small vines, evaluates likelihoods and PITs, saves/reloads the fits, and
checks the automatic NumPy sampler fallback. It is an installation check, not a
new accuracy or timing benchmark. CPU and GPU users choose an appropriate
PyTorch build before installing VDC; platform-specific compiled sampling stays
optional.

## Publish a model

`scripts/package_pretrained_model.py --help` describes staging a model bundle.
The staging process requires a source checkpoint; the released public manifest
is a download descriptor and does not contain a personal source path.
Use `scripts/upload_pretrained_to_hf.py --help` for publication. Supply credentials
through the supported environment variable or credential store, never source
files. Use a new model ID for changed weights and record the exact host revision,
checkpoint hash, configuration and evaluation protocol in its manifest/model card.

The model loader and curated source archive must work without access to a
maintainer's home directory, cluster account or private manuscript repository.
