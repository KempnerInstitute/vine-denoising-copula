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

## Publish

Create a new version tag after verification. Pushing a `v*` tag runs the
release workflow: it verifies the package version, tests the code, builds the
wheel and source distribution, and publishes the packages and checksums to
GitHub Releases. Never move a historical tag or replace a
checkpoint under an existing model ID. The paper's saved outputs in `paper/`
and the figure archives attached to releases v0.2.4 to v0.2.6 retain their
generation pins even when later software versions are published.

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
