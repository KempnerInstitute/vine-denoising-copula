#!/usr/bin/env python3
# ruff: noqa: E402
"""Stage a publishable pretrained-model bundle from the canonical paper checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.pretrained import (
    DEFAULT_PRETRAINED_MODEL_ID,
    load_pretrained_manifest,
    stage_release_bundle,
)


def build_model_card(manifest: dict, repo_id: str | None) -> str:
    model_id = manifest["model_id"]
    display_name = manifest.get("display_name", model_id)
    method_tag = manifest.get("method_tag", "unknown")
    model_type = manifest.get("model_type", "unknown")
    checkpoint_step = manifest.get("checkpoint_step", "unknown")
    selection_rule = manifest.get("selection_rule", "unknown")
    frozen_on = manifest.get("frozen_on", "unknown")
    sha256 = manifest.get("sha256", "")
    intended = "\n".join(f"- {item}" for item in manifest.get("intended_use", []))
    limitations = "\n".join(f"- {item}" for item in manifest.get("limitations", []))
    hf_repo = repo_id or "<set-this-before-public-release>"
    return f"""---
library_name: custom
license: mit
tags:
- copula
- vine-copula
- density-estimation
- mutual-information
---

# {display_name}

This repository contains the official pretrained checkpoint for the VDC paper model.

Model id: `{model_id}`
Suggested Hugging Face repo id: `{hf_repo}`

## What This Is

- Method tag: `{method_tag}`
- Model type: `{model_type}`
- Checkpoint step: `{checkpoint_step}`
- Selection rule: `{selection_rule}`
- Frozen on: `{frozen_on}`
- SHA256: `{sha256}`

## Files

- `manifest.json`: portable manifest for the released checkpoint
- `train_config.yaml`: exact training config embedded in the paper run
- `model_selection_joint_best.json`: checkpoint-selection provenance
- `{manifest.get('checkpoint_filename', 'model.pt')}`: pretrained weights

## Intended Use

{intended or "- pretrained vine pair-copula estimation"}

## Limitations

{limitations or "- see paper"}

## Usage

```bash
python scripts/download_pretrained.py --model-id {model_id} --repo-id {hf_repo}
python examples/use_pretrained_model.py --model-id {model_id} --repo-id {hf_repo}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage a pretrained-model release bundle")
    parser.add_argument("--model-id", default=DEFAULT_PRETRAINED_MODEL_ID)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--include-local-paths", action="store_true")
    args = parser.parse_args()

    manifest = load_pretrained_manifest(args.model_id)
    out_dir = stage_release_bundle(
        args.model_id,
        out_dir=args.out_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        include_local_paths=args.include_local_paths,
    )
    (out_dir / "README.md").write_text(build_model_card(manifest, args.repo_id))
    (out_dir / "paper_best_model.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(out_dir)


if __name__ == "__main__":
    main()
