#!/usr/bin/env python3
"""Download published distributions and compare them with the verified build."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=["pypi", "testpypi"], required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    host = "pypi.org" if args.index == "pypi" else "test.pypi.org"
    url = f"https://{host}/pypi/vine-denoising-copula/{args.version}/json"
    for attempt in range(12):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                release = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 404 or attempt == 11:
                raise
            time.sleep(5)
    expected = sorted(args.dist_dir.glob("*.whl")) + sorted(args.dist_dir.glob("*.tar.gz"))
    if len(expected) != 2 or not any(p.suffix == ".whl" for p in expected):
        raise RuntimeError("Expected exactly one wheel and one source distribution")
    remote = {entry["filename"]: entry for entry in release["urls"]}
    if set(remote) != {p.name for p in expected}:
        raise RuntimeError("Published release file list differs from the verified build")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for artifact in expected:
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        entry = remote[artifact.name]
        if entry["digests"]["sha256"] != digest:
            raise RuntimeError(f"Published hash differs for {artifact.name}")
        with urllib.request.urlopen(entry["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Downloaded hash differs for {artifact.name}")
        (args.output_dir / artifact.name).write_bytes(data)
        print(f"Verified {host}: {artifact.name} {digest}")


if __name__ == "__main__":
    main()
