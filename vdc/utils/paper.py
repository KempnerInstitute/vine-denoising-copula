"""Utilities for paper experiment orchestration (run discovery, checkpoint selection)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RUN_DIR_RE = re.compile(r"^vdc_paper_(?P<method>.+?)_(?P<ts>\d{8}_\d{6})_(?P<jobid>.+)$")
REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_CHECKPOINT_CANDIDATES = [REPO_ROOT / "analysis" / "PAPER_CHECKPOINT.txt"]
PAPER_BEST_MODEL_CANDIDATES = [REPO_ROOT / "analysis" / "PAPER_BEST_MODEL.json"]
PAPER_CHECKPOINT_TXT = PAPER_CHECKPOINT_CANDIDATES[0]
PAPER_BEST_MODEL_JSON = PAPER_BEST_MODEL_CANDIDATES[0]


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

    @property
    def model_selection_joint_json(self) -> Path:
        return self.run_dir / "results" / "model_selection_joint_best.json"

    @property
    def checkpoint_path_joint_txt(self) -> Path:
        return self.run_dir / "results" / "checkpoint_path_joint.txt"


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


def _normalize_checkpoint_candidate(raw: str) -> Optional[Path]:
    cand = str(raw).strip()
    if not cand:
        return None
    p = Path(cand).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p if p.exists() else None


def resolve_canonical_paper_checkpoint() -> Optional[Path]:
    """Resolve a canonical checkpoint path for paper runs.

    Resolution order:
      1) `PAPER_CHECKPOINT` environment variable
      2) legacy local `analysis/PAPER_CHECKPOINT.txt`
      3) legacy local `analysis/PAPER_BEST_MODEL.json`
      4) packaged pretrained release (local path or Hugging Face download)
    """
    ckpt = _normalize_checkpoint_candidate(os.environ.get("PAPER_CHECKPOINT", ""))
    if ckpt is not None:
        return ckpt

    for txt_path in PAPER_CHECKPOINT_CANDIDATES:
        if not txt_path.exists():
            continue
        try:
            ckpt = _normalize_checkpoint_candidate(txt_path.read_text())
            if ckpt is not None:
                return ckpt
        except Exception:
            pass

    for manifest_path in PAPER_BEST_MODEL_CANDIDATES:
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text())
            if isinstance(payload, dict):
                for key in ("checkpoint", "checkpoint_path"):
                    ckpt = _normalize_checkpoint_candidate(str(payload.get(key, "")))
                    if ckpt is not None:
                        return ckpt
        except Exception:
            pass

    try:
        from vdc.pretrained import DEFAULT_PRETRAINED_MODEL_ID, resolve_pretrained_checkpoint

        ckpt = resolve_pretrained_checkpoint(DEFAULT_PRETRAINED_MODEL_ID, prefer_local=True)
        if ckpt is not None and ckpt.exists():
            return ckpt
    except Exception:
        pass

    return None


def choose_best_checkpoint(
    *,
    output_bases: Iterable[Path],
    preferred_methods: List[str],
    metric: str = "mean_ise",
    prefer_joint: bool = False,
    prefer_canonical: bool = True,
) -> Optional[Path]:
    """Choose the best available checkpoint from paper run directories.

    Selection:
      - If `prefer_canonical=True`, first try canonical checkpoint files
        (`PAPER_CHECKPOINT`, `analysis/PAPER_CHECKPOINT.txt`,
        `analysis/PAPER_BEST_MODEL.json`, then the packaged pretrained release)
      - Consider runs with results/model_selection.json available
      - Filter by preferred_methods (in the given order)
      - Choose minimum of `metric` (ties broken by timestamp)
      - If prefer_joint=True, first try latest valid checkpoint from
        `results/checkpoint_path_joint.txt` for preferred methods.

    Returns:
      Path to checkpoint, or None if none found.
    """
    if bool(prefer_canonical):
        canonical = resolve_canonical_paper_checkpoint()
        if canonical is not None:
            return canonical

    preferred = [str(m).strip() for m in preferred_methods if str(m).strip()]
    runs = discover_paper_runs(output_bases)

    # Index by method
    by_method: Dict[str, List[PaperRun]] = {}
    for r in runs:
        by_method.setdefault(r.method, []).append(r)
    for m in by_method:
        by_method[m].sort(key=lambda rr: rr.timestamp, reverse=True)

    if prefer_joint:
        for method in preferred:
            for r in by_method.get(method, []):
                p = r.checkpoint_path_joint_txt
                if not p.exists():
                    continue
                try:
                    ckpt = Path(p.read_text().strip())
                except Exception:
                    continue
                if ckpt.exists():
                    return ckpt

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
