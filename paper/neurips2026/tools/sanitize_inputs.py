#!/usr/bin/env python3
"""Copy saved experiment outputs into the release tree without internal storage details.

JSON files are parsed and rewritten (numbers are unchanged: Python's float repr round-trips exactly):
absolute paths become their final component, cluster hostnames become "<host>", and keys holding
reviewer identifiers or scheduler job ids are dropped. LaTeX files lose their full-line comments.
Other files (NPZ arrays, CSV) are copied byte for byte. Every copy is recorded in MANIFEST.json with
the sha256 of the copied file and of its source, so a released file can be matched to the original
output it came from.

Usage:
    python sanitize_inputs.py --out DIR --manifest DIR/MANIFEST.json  SRC=DEST [SRC=DEST ...]

DEST is the path relative to --out; SRC is the original file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ABS_PATH = re.compile(r"(?<![\w.])(?:/n/|/tmp/|/home/|/scratch/|/net/)[^\s\"',;:)\]]*")
HOST = re.compile(r"\b[a-z0-9][a-z0-9-]*\.rc\.fas\.harvard\.edu\b|\bholy[a-z0-9]*\d{3,}\b|\bboslogin\d+\b")
DROP_KEYS = re.compile(r"^(reviewer|reviewers|slurm_job_id|slurm_jobid|job_id|jobid|job|hostname)$", re.IGNORECASE)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_string(s: str, counts: dict) -> str:
    def repl_path(m):
        counts["paths"] += 1
        tail = m.group(0).rstrip("/").split("/")[-1]
        return tail or "<path>"

    s2 = ABS_PATH.sub(repl_path, s)
    s3, n = HOST.subn("<host>", s2)
    counts["hosts"] += n
    return s3


def clean(obj, counts: dict):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if DROP_KEYS.match(str(k)):
                counts["dropped_keys"] += 1
                continue
            out[clean_string(k, counts) if isinstance(k, str) else k] = clean(v, counts)
        return out
    if isinstance(obj, list):
        return [clean(v, counts) for v in obj]
    if isinstance(obj, str):
        return clean_string(obj, counts)
    return obj


def strip_tex_comments(text: str) -> str:
    return "".join(x for x in text.splitlines(keepends=True) if not x.lstrip().startswith("%"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("pairs", nargs="+", help="SRC=DEST")
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text()) if a.manifest.exists() else {"files": {}}
    for pair in a.pairs:
        src, dest = pair.split("=", 1)
        src_p, dest_p = Path(src), a.out / dest
        raw = src_p.read_bytes()
        counts = {"paths": 0, "hosts": 0, "dropped_keys": 0}
        if src_p.suffix == ".json":
            data = (json.dumps(clean(json.loads(raw), counts), indent=2, ensure_ascii=False) + "\n").encode()
        elif src_p.suffix == ".tex":
            data = strip_tex_comments(raw.decode()).encode()
        else:
            data = raw
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        dest_p.write_bytes(data)
        manifest["files"][dest] = {"sha256": sha256(data), "source_sha256": sha256(raw), **counts}
        print(f"{dest}: {counts}")
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
