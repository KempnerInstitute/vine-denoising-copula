#!/usr/bin/env python3
"""Thread sweep of the earlier implementation (tab_pyvine_threads).

Inputs:
  threads/e25_pyvine_threads.json        pyvine-TLL fits at d = 100, 200 with 1, 8 and 16 CPU threads
                                         (seeds 7, 17, 42; synthetic Clayton vine, n_train 20,000)
  threads/e25_pyvine_threads_d500.json   the same at d = 500 with 8 and 16 threads; one 8-thread fit
                                         ran past the 7,200 s cap and is marked as a timeout
  threads/d500_single_thread.json        the single-thread d = 500 fit, which never returned: when it
                                         started and when the sweep job reached its time limit
  scaling/e22_highdim_scaling_seed_merged.json   VDC fit times (mean and seed SD) of the main
                                         scaling run, against which the speedups are computed
Means and +- values are over seeds (population standard deviation, as in the sweep's own writer).
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from common import load, parse, write


def cell(ts) -> str:
    ts = np.asarray(ts, float)
    return f"{ts.mean():.1f} {{\\scriptsize $\\pm$ {ts.std():.1f}}}"


def floor_sig(x: float, sig: int) -> tuple[float, int]:
    e = math.floor(math.log10(x))
    m = math.floor(x / 10 ** (e - sig + 1)) / 10 ** (sig - 1)
    return m, e


def main() -> None:
    a = parse(__doc__)
    sweep = load(a.inputs, "threads/e25_pyvine_threads.json")
    d500 = load(a.inputs, "threads/e25_pyvine_threads_d500.json")
    single = load(a.inputs, "threads/d500_single_thread.json")
    e22 = load(a.inputs, "scaling/e22_highdim_scaling_seed_merged.json")
    vdc = {r["d"]: (r["fit_s"], r["fit_s_std"]) for r in e22["records"] if r["method"] == "VDC"}
    recs = sweep["records"] + d500["records"]
    ok = {}
    for r in recs:
        if r["status"] == "ok":
            ok.setdefault((r["d"], r["num_threads"]), []).append(r["fit_s"])
    timeouts = [r for r in recs if r["status"].startswith("timeout")]
    assert len(timeouts) == 1 and (timeouts[0]["d"], timeouts[0]["num_threads"]) == (500, 8)
    cap = int(d500["settings"]["pyvine_timeout_s"])
    bound_s = (datetime.fromisoformat(single["job_ended"]) - datetime.fromisoformat(single["fit_started"])).total_seconds()
    bound_h = math.floor(bound_s / 360) / 10
    m, e = floor_sig(bound_s, 2)
    lo = ok[(500, 16)]

    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"$d$ & VDC (s) & TLL $T{=}1$ (s) & TLL $T{=}8$ (s) & TLL $T{=}16$ (s) & VDC speedup ($T{=}1$ / $8$ / $16$) \\",
             r"\midrule"]
    for d in (100, 200, 500):
        v_mean, v_std = vdc[d]
        vdc_cell = f"{v_mean:.1f} {{\\scriptsize $\\pm$ {v_std:.1f}}}"
        if d == 500:
            t1 = f"$>\\!{m:.1f}\\times10^{{{e}}}$\\,$^{{\\dagger}}$"
            sp1 = f"$>\\!{math.floor(bound_s / v_mean)}\\times$"
            t8 = cell(ok[(d, 8)]) + r"\,$^{\ddagger}$"
        else:
            t1 = cell(ok[(d, 1)])
            sp1 = f"${np.mean(ok[(d, 1)]) / v_mean:.1f}\\times$"
            t8 = cell(ok[(d, 8)])
        sp8 = f"${np.mean(ok[(d, 8)]) / v_mean:.1f}\\times$"
        sp16 = f"${np.mean(ok[(d, 16)]) / v_mean:.1f}\\times$"
        lines.append(f"{d} & {vdc_cell} & {t1} & {t8} & {cell(ok[(d, 16)])} & {sp1} / {sp8} / {sp16} \\\\")
    note = (r"\multicolumn{6}{p{0.95\linewidth}}{\scriptsize $^{\dagger}$A single single-threaded fit at $d{=}500$ "
            f"had not completed after ${bound_h:.1f}$ hours, when the sweep job reached its time limit (one run; lower bound). "
            f"The $d{{=}}500$ $T{{=}}16$ "
            f"dispersion reflects node contention across runs (range ${min(lo):.0f}$--${max(lo):.0f}$\\,s); pyvine times "
            "in this sweep were measured on different nodes from the main-table runs. $^{\\ddagger}$Mean over two seeds; "
            f"the third $d{{=}}500$, $T{{=}}8$ fit exceeded the ${cap}$\\,s cap at ${timeouts[0]['fit_s']:.0f}$\\,s.}} \\\\")
    lines += [r"\midrule", note, r"\bottomrule", r"\end{tabular}"]
    write(a.out, "tab_pyvine_threads.tex", "\n".join(lines) + "\n", "threads.py")


if __name__ == "__main__":
    main()
