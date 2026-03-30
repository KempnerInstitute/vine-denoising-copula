#!/usr/bin/env python3
"""Download or resolve an official pretrained VDC model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.pretrained import (
    DEFAULT_PRETRAINED_MODEL_ID,
    list_pretrained_models,
    resolve_pretrained_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve or download a pretrained VDC model")
    parser.add_argument("--model-id", default=DEFAULT_PRETRAINED_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-local", action="store_true", help="Do not use packaged local checkpoint paths")
    parser.add_argument("--list", action="store_true", help="List known packaged model ids and exit")
    args = parser.parse_args()

    if args.list:
        for payload in list_pretrained_models():
            print(f"{payload['model_id']}: {payload.get('display_name', '')}")
        return

    path = resolve_pretrained_checkpoint(
        args.model_id,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
        prefer_local=not args.no_local,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    print(path)


if __name__ == "__main__":
    main()
