#!/usr/bin/env python3
"""Audit referenced paper figures/tables for existence and placeholder content."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tex_path = repo_root / "drafts" / "vine_diffusion.tex"
    tex = tex_path.read_text()

    fig_refs = sorted(set(re.findall(r"figures/[^}\s]+\.(?:pdf|png)", tex)))
    tab_refs = sorted(set(re.findall(r"tables/[^}\s]+\.tex", tex)))

    out: Dict[str, object] = {
        "tex": str(tex_path),
        "figures": [],
        "tables": [],
        "summary": {},
    }

    missing_figs: List[str] = []
    for rel in fig_refs:
        p = repo_root / "drafts" / rel
        rec = {"ref": rel, "exists": p.exists()}
        if p.exists():
            rec["size_bytes"] = int(p.stat().st_size)
        else:
            missing_figs.append(rel)
        out["figures"].append(rec)

    missing_tabs: List[str] = []
    flagged_tabs: List[str] = []
    bad_pat = re.compile(r"placeholder|\\emph\{TBD|TBD:", re.IGNORECASE)
    for rel in tab_refs:
        p = repo_root / "drafts" / rel
        rec = {"ref": rel, "exists": p.exists(), "flagged_placeholder": False}
        if p.exists():
            txt = p.read_text()
            rec["size_bytes"] = int(p.stat().st_size)
            if bad_pat.search(txt):
                rec["flagged_placeholder"] = True
                flagged_tabs.append(rel)
        else:
            missing_tabs.append(rel)
        out["tables"].append(rec)

    out["summary"] = {
        "n_figure_refs": len(fig_refs),
        "n_table_refs": len(tab_refs),
        "missing_figures": missing_figs,
        "missing_tables": missing_tabs,
        "flagged_placeholder_tables": flagged_tabs,
        "ok": (len(missing_figs) == 0 and len(missing_tabs) == 0 and len(flagged_tabs) == 0),
    }

    out_json = repo_root / "analysis" / "paper_asset_audit_latest.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {out_json}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
