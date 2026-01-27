"""Common path conventions for paper datasets/results.

User request: *use the OUTPUT_BASE folder for dataset locations*.

We therefore define:
  DATA_ROOT = $DATA_ROOT if set, else ($OUTPUT_BASE / "datasets") if OUTPUT_BASE set,
              else <repo>/data (fallback for local dev).
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    # vdc/data/paths.py -> vdc/ -> repo root
    return Path(__file__).resolve().parents[2]


def output_base() -> Path:
    v = os.environ.get("OUTPUT_BASE")
    if v:
        return Path(v).expanduser()
    # Fallback for local runs (keeps behavior sane outside the cluster)
    return repo_root() / "results"


def data_root() -> Path:
    v = os.environ.get("DATA_ROOT")
    if v:
        return Path(v).expanduser()
    ob = os.environ.get("OUTPUT_BASE")
    if ob:
        return Path(ob).expanduser() / "datasets"
    return repo_root() / "data"

