#!/usr/bin/env python3
"""Run a paper experiment against an explicitly selected VDC checkout.

Example: python paper/run.py --library /path/to/vdc runners/run_uci_maf.py --help
Numerical options are passed through unchanged. Figure/table rendering needs no library.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=Path(__file__).resolve().parents[1],
                        help='VDC checkout or exported release (default: this repository)')
    parser.add_argument('script', help='Path relative to paper/, e.g. runners/run_uci_maf.py')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    library = args.library.resolve()
    target = (root / args.script).resolve()
    if not (library / 'vdc/__init__.py').is_file():
        parser.error('--library must name a VDC checkout or exported release root')
    if not target.is_file() or not target.is_relative_to(root) or target.suffix != '.py':
        parser.error('script must be a Python file inside paper/')
    paths = [str(library), str(library / 'scripts'), str(root / 'experiments')]
    sys.path.insert(0, str(library))
    spec = importlib.util.find_spec('vdc')
    if spec is None or Path(spec.origin).resolve() != library / 'vdc/__init__.py':
        parser.error('VDC import does not resolve to the selected library')
    env = {**os.environ, 'VDC_LIBRARY_ROOT': str(library), 'PYTHONPATH': os.pathsep.join(paths),
           'PYTHONDONTWRITEBYTECODE': '1'}
    return subprocess.call([sys.executable, str(target), *args.arguments], env=env)


if __name__ == '__main__':
    raise SystemExit(main())
