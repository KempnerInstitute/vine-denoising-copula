"""Utilities for paper experiment orchestration (run discovery, checkpoint selection)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RUN_DIR_RE = re.compile(r"^vdc_paper_(?P<method>.+?)_(?P<ts>\d{8}_\d{6})_(?P<jobid>.+)$")


@dataclass(frozen=True)
class PaperRun:
    run_dir: Path
    method: str
    timestamp: str
    jobid: str

    @property
    def model_selection_json(self) -> Path:
        return self.run_dir / "results" / "model_selection.json"

    @property
    def checkpoint_path_txt(self) -> Path:
        return self.run_dir / "results" / "checkpoint_path.txt"


def discover_paper_runs(output_bases: Iterable[Path]) -> List[PaperRun]:
    runs: List[PaperRun] = []
    for base in output_bases:
        base = Path(base)
        if not base.exists():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            m = RUN_DIR_RE.match(child.name)
            if not m:
                continue
            runs.append(PaperRun(run_dir=child, method=m.group("method"), timestamp=m.group("ts"), jobid=m.group("jobid")))
    return runs


def choose_best_checkpoint(
    *,
    output_bases: Iterable[Path],
    preferred_methods: List[str],
    metric: str = "mean_ise",
) -> Optional[Path]:
    """Choose the best available checkpoint from paper run directories.

    Selection:
      - Consider runs with results/model_selection.json available
      - Filter by preferred_methods (in the given order)
      - Choose minimum of `metric` (ties broken by timestamp)

    Returns:
      Path to checkpoint, or None if none found.
    """
    preferred = [str(m).strip() for m in preferred_methods if str(m).strip()]
    runs = discover_paper_runs(output_bases)

    # Index by method
    by_method: Dict[str, List[PaperRun]] = {}
    for r in runs:
        by_method.setdefault(r.method, []).append(r)
    for m in by_method:
        by_method[m].sort(key=lambda rr: rr.timestamp, reverse=True)

    best: Optional[Tuple[float, str, Path]] = None  # (metric, ts, ckpt)

    for method in preferred:
        for r in by_method.get(method, []):
            msj = r.model_selection_json
            if not msj.exists():
                continue
            try:
                payload = json.loads(msj.read_text())
                results = payload.get("results", [])
                if not isinstance(results, list) or not results or not isinstance(results[0], dict):
                    continue
                s0: Dict[str, Any] = results[0]
                val = float(s0.get(metric))
                ckpt = Path(str(s0.get("checkpoint", "")).strip())
                if not ckpt.exists() and r.checkpoint_path_txt.exists():
                    try:
                        ckpt = Path(r.checkpoint_path_txt.read_text().strip())
                    except Exception:
                        ckpt = Path("")
                if not ckpt.exists():
                    continue
            except Exception:
                continue

            cand = (val, r.timestamp, ckpt)
            if best is None or cand < best:
                best = cand

        if best is not None:
            break  # respect preferred method order

    return best[2] if best is not None else None

