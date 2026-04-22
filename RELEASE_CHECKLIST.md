# Release Checklist

Use this checklist before publishing a tagged release of Vine Denoising Copula.

## Code and metadata

- Confirm the package version in `pyproject.toml`
- Confirm the release date and repository URL in `CITATION.cff`
- Confirm the GitHub remote points at the intended public repository
- Confirm the README and docs refer to the released model and current URLs

## Package verification

- Create a fresh environment from `environment.yml`
- Install the package with `pip install -e .`
- Run the test suite
- Run `vdc list-models`
- Run `python examples/use_pretrained_model.py --model-id vdc-denoiser-m64-v1`

## Released model verification

- Run `python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1`
- Run `python scripts/verify_pretrained_release.py --model-id vdc-denoiser-m64-v1 --device cpu --out-dir docs/reports/pretrained_release`
- Check that the verification reports and figures regenerate cleanly

## Documentation

- Review `README.md`
- Review `docs/index.html`
- Review `docs/GETTING_STARTED.md`
- Review `docs/MODEL_RELEASES.md`

## Paper and arXiv

- If uploading the preprint, rebuild `drafts/vine_diffusion_arxiv.pdf`
- Confirm the arXiv PDF uses the public repository URL
- Add the arXiv link back into the repo only after the paper is live

## GitHub release

- Create and push the release tag, for example `v0.1.0`
- Use `RELEASE_NOTES_v0.1.0.md` as the initial release note text
- Attach or link the Hugging Face model id `vdc-denoiser-m64-v1`

