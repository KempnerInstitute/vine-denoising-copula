#!/usr/bin/env python3
"""Information self-consistency tables: tab_self_consistency, tab_self_consistency_scaleup,
tab_dpi_tolerance and tab_additivity_learned.

Inputs (self_consistency/):
  mi_self_consistency.json           original protocol (60 DPI trials per estimator) and
  mi_self_consistency_scaleup.json   scaled protocol (480 DPI trials: 160 base datasets at three
                                     noise levels), both written by the library's
                                     scripts/mi_self_consistency_tests_v2.py
  e35_additivity_learned.json        additivity error with the oracle block structure, a learned
                                     (Dissmann) structure and KSG on identical data (120 trials)
A DPI trial is a violation when the MI estimated after the noisy (garbling) channel exceeds the MI
before it by more than the tolerance; the protocol tolerance is 1e-3 nats. Fisher exact tests are
two-sided and pool the 480 trials of each estimator.
"""
from __future__ import annotations

from scipy.stats import fisher_exact

from common import load, parse, write

NAMES = {"KSG": "KSG", "DCD-Vine": "VDC", "VDC": "VDC"}  # the original protocol labels VDC "DCD-Vine"
TOLERANCES = [(0.0, "0"), (1e-4, "$10^{-4}$"), (3e-4, r"$3\times10^{-4}$"), (1e-3, "$10^{-3}$ (paper)"),
              (3e-3, r"$3\times10^{-3}$"), (1e-2, "$10^{-2}$")]


def summary_table(rows) -> str:
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Method & DPI Viol. (\%) $\downarrow$ & Add. Err. (nats) $\downarrow$ & Mono. Err. (nats) $\downarrow$ \\",
             r"\midrule"]
    for r in rows:
        dpi = float(r["dpi"]["summary"]["violation_rate"]) * 100.0
        add = float(r["additivity"]["summary"]["mean_abs_error"])
        mono = float(r["monotone_invariance"]["summary"]["mean_abs_error"])
        lines.append(f"{NAMES[r['estimator']]} & {dpi:.1f} & {add:.3f} & {mono:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def violations(row, tol: float) -> int:
    return sum(r["mi_noisy"] > r["mi_base"] + tol for r in row["dpi"]["records"])


def tolerance_table(rows) -> str:
    by = {NAMES[r["estimator"]]: r for r in rows}
    n = len(by["VDC"]["dpi"]["records"])
    assert n == len(by["KSG"]["dpi"]["records"]) == 480
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Tolerance $\tau$ (nats) & VDC violations & KSG violations & Fisher $p$ \\", r"\midrule"]
    for tol, label in TOLERANCES:
        kv, kk = violations(by["VDC"], tol), violations(by["KSG"], tol)
        p = fisher_exact([[kv, n - kv], [kk, n - kk]])[1]
        lines.append(f"{label} & {kv}/{n} ({100 * kv / n:.1f}\\%) & {kk}/{n} ({100 * kk / n:.1f}\\%) & {p:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def additivity_table(summary) -> str:
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Estimator & Additivity err.\ (nats) $\downarrow$ & Trials \\",
             r"\midrule"]
    for name, key in [("VDC (oracle structure)", "vdc_oracle"), ("VDC (learned structure)", "vdc_learned"), ("KSG", "ksg")]:
        s = summary[key]
        lines.append(f"{name} & ${s['mean_abs_error']:.3f} \\pm {s['sd_abs_error']:.3f}$ & {s['n_trials']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    a = parse(__doc__)
    write(a.out, "tab_self_consistency.tex", summary_table(load(a.inputs, "self_consistency/mi_self_consistency.json")),
          "self_consistency.py")
    scaleup = load(a.inputs, "self_consistency/mi_self_consistency_scaleup.json")
    write(a.out, "tab_self_consistency_scaleup.tex", summary_table(scaleup), "self_consistency.py")
    write(a.out, "tab_dpi_tolerance.tex", tolerance_table(scaleup), "self_consistency.py")
    write(a.out, "tab_additivity_learned.tex",
          additivity_table(load(a.inputs, "self_consistency/e35_additivity_learned.json")["summary"]), "self_consistency.py")


if __name__ == "__main__":
    main()
