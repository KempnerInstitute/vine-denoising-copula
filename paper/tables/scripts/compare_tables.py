#!/usr/bin/env python3
"""Compare generated tables with reference tables, ignoring full-line LaTeX comments.

usage: compare_tables.py GENERATED_DIR REFERENCE_DIR
Prints IDENTICAL or DIFFERS for every generated table (with a unified diff of the differing lines)
and exits with status 1 if any table differs or has no reference.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path


def body(path: Path) -> list[str]:
    return [x for x in path.read_text().splitlines(keepends=True) if not x.lstrip().startswith("%")]


def main() -> int:
    gen, ref = Path(sys.argv[1]), Path(sys.argv[2])
    bad = 0
    for g in sorted(gen.glob("*.tex")):
        r = ref / g.name
        if not r.exists():
            print(f"MISSING REFERENCE {g.name}")
            bad += 1
            continue
        a, b = body(r), body(g)
        if a == b:
            print(f"IDENTICAL {g.name}")
        else:
            bad += 1
            print(f"DIFFERS {g.name}")
            sys.stdout.writelines(difflib.unified_diff(a, b, fromfile=f"reference/{g.name}", tofile=f"generated/{g.name}", n=0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
