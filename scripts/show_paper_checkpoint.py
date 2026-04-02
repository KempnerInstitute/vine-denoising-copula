#!/usr/bin/env python3
"""Print the canonical paper checkpoint and manifest details."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdc.utils.paper import (
    PAPER_BEST_MODEL_CANDIDATES,
    PAPER_CHECKPOINT_CANDIDATES,
    resolve_canonical_paper_checkpoint,
)


def main() -> None:
    ckpt = resolve_canonical_paper_checkpoint()
    print("checkpoint_candidates:")
    for p in PAPER_CHECKPOINT_CANDIDATES:
        print(f"  - {p}")
    print("manifest_candidates:")
    for p in PAPER_BEST_MODEL_CANDIDATES:
        print(f"  - {p}")
    print(f"resolved_ckpt:  {ckpt if ckpt is not None else 'NONE'}")

    for manifest_path in PAPER_BEST_MODEL_CANDIDATES:
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text())
            method = str(payload.get("method_tag", ""))
            model_type = str(payload.get("model_type", ""))
            run_dir = str(payload.get("run_dir", ""))
            print(f"manifest_used:   {manifest_path}")
            print(f"method_tag:     {method}")
            print(f"model_type:     {model_type}")
            print(f"run_dir:        {run_dir}")
            break
        except Exception as e:
            print(f"manifest_parse_error ({manifest_path}): {e}")

    if ckpt is None or (not Path(ckpt).exists()):
        raise SystemExit("ERROR: canonical paper checkpoint is not resolvable.")


if __name__ == "__main__":
    main()
