#!/usr/bin/env python3
"""Packaging step: write the compact extract of the OOD re-scoring output used by ood.py.

The full re-scoring output (about 1.2 MB) also holds every replicate's raw scores and a second,
unrelated check. The table needs only the nine per-case summaries and their means, so the release
keeps those and records the size and sha256 of the file they were taken from.

usage: compact_inputs.py --ood-full FULL.json --out e26_rescored.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ood-full", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    raw = a.ood_full.read_bytes()
    full = json.loads(raw)
    e26 = full["e26"]
    assert len(e26["cases"]) == 9
    out = {
        "e26": {"cases": [{"summary": c["summary"]} for c in e26["cases"]], "means": e26["means"]},
        "checkpoint": full.get("checkpoint"),
        "device": full.get("device"),
        "extracted_from": {"file": a.ood_full.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                           "dropped": "per-replicate records, helper checks and the separate e31 section"},
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out} ({a.out.stat().st_size} bytes) from {len(raw)} bytes")


if __name__ == "__main__":
    main()
