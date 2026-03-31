# Model Releases

This project treats pretrained checkpoints as versioned artifacts. The repository contains the manifest, loader, packaging scripts, and verification tooling. The large model weights should live in a model store such as Hugging Face, not in git.

## Official Released Model

Current packaged model id:

- `vdc-denoiser-m64-v1`

Canonical paper sources:

- local private `analysis/PAPER_CHECKPOINT.txt` and `analysis/PAPER_BEST_MODEL.json` when available
- otherwise the packaged pretrained manifest plus Hugging Face release

Packaged manifest:

- `vdc/resources/pretrained/vdc_denoiser_m64_v1.json`

## What Lives In The Repo

- packaged manifests in `vdc/resources/pretrained`
- a versioned loader in `vdc/pretrained.py`
- `scripts/package_pretrained_model.py`
- `scripts/upload_pretrained_to_hf.py`
- `scripts/download_pretrained.py`
- `scripts/verify_pretrained_release.py`

## What Lives Outside The Repo

- the full model checkpoint
- the staged release bundle
- the public Hugging Face model repository

## Release Workflow

### 1. Stage the bundle

Run this from repo root:

```bash
python scripts/package_pretrained_model.py \
  --model-id vdc-denoiser-m64-v1 \
  --out-dir /tmp/vdc-denoiser-m64-v1 \
  --repo-id YOUR_USER_OR_ORG/vdc-denoiser-m64-v1
```

The staged directory will contain:

- `vdc-denoiser-m64-v1.pt`
- `train_config.yaml`
- `manifest.json`
- `paper_best_model.json`
- `model_selection_joint_best.json`
- a model-card `README.md`

### 2. Create or choose the Hugging Face repo

Recommended repo name:

- `YOUR_USER_OR_ORG/vdc-denoiser-m64-v1`

You can let the upload script create it automatically.

### 3. Provide Hugging Face authentication

Set a token in your shell:

```bash
export HF_TOKEN=YOUR_HF_WRITE_TOKEN
```

The upload script reads `HF_TOKEN` by default.

### 4. Upload the bundle

```bash
python scripts/upload_pretrained_to_hf.py \
  --bundle-dir /tmp/vdc-denoiser-m64-v1 \
  --repo-id YOUR_USER_OR_ORG/vdc-denoiser-m64-v1 \
  --create-repo
```

If you want the repo private during staging:

```bash
python scripts/upload_pretrained_to_hf.py \
  --bundle-dir /tmp/vdc-denoiser-m64-v1 \
  --repo-id YOUR_USER_OR_ORG/vdc-denoiser-m64-v1 \
  --create-repo \
  --private
```

### 5. Update the packaged manifest

After upload succeeds, update:

- `vdc/resources/pretrained/vdc_denoiser_m64_v1.json`

Fill in:

- `sources.huggingface.repo_id`
- `sources.huggingface.revision`
- `sources.huggingface.filename`

This makes the packaged loader portable for external users.

### 6. Re-verify the published model

```bash
python scripts/verify_pretrained_release.py \
  --model-id vdc-denoiser-m64-v1 \
  --device cpu \
  --out-dir docs/reports/pretrained_release
```

## Download Workflow For Users

Once the Hugging Face repo is public and the manifest points to it, users can run:

```bash
python scripts/download_pretrained.py --model-id vdc-denoiser-m64-v1
python examples/use_pretrained_model.py --model-id vdc-denoiser-m64-v1
```

## Notes On Versioning

- Keep the paper checkpoint frozen.
- If you publish a better later model, give it a new model id and version.
- Do not silently replace the checkpoint behind `vdc-denoiser-m64-v1`.

## Notes For This Repository

At the time of writing, the repo-side release tooling is ready. The only thing required to perform the actual upload is a valid Hugging Face write token in `HF_TOKEN`.
