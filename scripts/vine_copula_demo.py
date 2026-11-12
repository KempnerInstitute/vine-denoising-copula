#!/usr/bin/env python3
"""Compatibility entry point; see examples/fit_vine_copula.py --help."""
from pathlib import Path
import runpy
import warnings

if __name__ == "__main__":
    warnings.warn("Use examples/fit_vine_copula.py; this entry point is deprecated.",
                  FutureWarning, stacklevel=1)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "fit_vine_copula.py"),
                  run_name="__main__")
