#!/usr/bin/env python3
"""Reduce the E16 rolling-dependence output to what the rolling-dependence figure reads.

Keeps the run metadata (the checkpoint path is cut to its file name) and, per window, window_index,
day_index, fit_s, mean_total_log_copula, per_tree_mean and top_tree1_edges; drops tree1_tau. Values pass
through Python floats, which round-trip exactly, so the figure drawn from the extract is unchanged.

Usage:
    python extract_rolling_dependence_inputs.py FULL_JSON OUT_JSON
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

META = ["generated_at", "checkpoint", "model_type", "vine_type", "window", "refit_every", "top_k", "n_windows",
        "mean_fit_s"]
WINDOW = ["window_index", "day_index", "fit_s", "mean_total_log_copula", "per_tree_mean", "top_tree1_edges"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("full_json", type=Path)
    ap.add_argument("out_json", type=Path)
    a = ap.parse_args()
    obj = json.loads(a.full_json.read_text())
    out = {k: obj[k] for k in META}
    out["checkpoint"] = Path(str(obj["checkpoint"])).name
    out["windows"] = [{k: w[k] for k in WINDOW} for w in obj["windows"]]
    assert out["n_windows"] == len(out["windows"])
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"{a.out_json}: {len(out['windows'])} windows, {a.out_json.stat().st_size} bytes")


if __name__ == "__main__":
    main()
