#!/usr/bin/env python3
"""Structural control for the AR(1) total-correlation benchmark (paper Appendix, TC control paragraph).

VDC estimates the TC of this first-order Markov benchmark on a D-vine truncated after its first tree,
TC_hat = sum_i I_hat(U_{i-1}; U_i). The control applies the same decomposition to KSG: bivariate KSG on
each adjacent pair, summed. It regenerates the seed-42 samples of the Figure 3(b) sweep exactly (the
run's own generator and trial seeds, seed 42 + 10000 d + 1000 t) and records every trial, next to the
VDC and full chain-rule KSG values of the reference run.

Usage: ksg_tree1.py LIBRARY_ROOT REFERENCE_JSON OUTPUT_JSON
  LIBRARY_ROOT    checkout of the vdc library to import (its commit is recorded)
  REFERENCE_JSON  paper_outputs/tc_benchmark_neural_full.json, an earlier seed-42 run whose datasets and
                  VDC and KSG estimates equal those of seed 42 in the Figure 3(b) sweep
"""
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

lib, ref_path, out_path = sys.argv[1:4]
sys.path.insert(0, lib)
import scipy  # noqa: E402
import vdc  # noqa: E402
from vdc.data.generators import generate_gaussian_vine  # noqa: E402
from vdc.utils.information import ksg_mutual_information  # noqa: E402


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(*args):
    """Output of a git command in the library checkout, or None outside a git checkout (e.g. a ZIP)."""
    try:
        return subprocess.check_output(["git", "-C", lib, *args], text=True, stderr=subprocess.DEVNULL,
                                       env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


ref = json.load(open(ref_path))
assert ref["truncation_level"] == 1 and ref["ksg_k"] == 5
rho, n, k, seed = ref["rho"], ref["n_samples"], ref["ksg_k"], 42
trials = []
for d in ref["dimensions"]:
    for t in range(ref["n_trials"]):
        trial_seed = seed + 10_000 * d + 1_000 * t
        U = generate_gaussian_vine(n=n, d=d, rho=rho, seed=trial_seed)
        t0 = time.perf_counter()
        tc = sum(max(0.0, float(ksg_mutual_information(U[:, i - 1:i], U[:, i], k=k, seed=trial_seed + 10_000 * i)))
                 for i in range(1, d))
        elapsed = time.perf_counter() - t0
        r = next(x for x in ref["raw_records"] if x["d"] == d and x["trial"] == t)
        trials.append(dict(d=d, trial=t, trial_seed=trial_seed, tc_true=r["tc_true"], ksg_tree1_tc=tc,
                           ksg_tree1_time_s=elapsed, vdc_tree1_tc=r["tc_dcd"], ksg_full_tc=r["tc_ksg"]))
        print(f"d={d} t={t}: true {r['tc_true']:.4f}  KSG tree1 {tc:.4f}  VDC tree1 {r['tc_dcd']:.4f}  "
              f"KSG full {r['tc_ksg']:.4f}  ({elapsed:.2f} s)", flush=True)

summary = {}
for d in ref["dimensions"]:
    rows = [x for x in trials if x["d"] == d]
    summary[str(d)] = {key: float(np.mean([x[key] for x in rows]))
                       for key in ("tc_true", "ksg_tree1_tc", "vdc_tree1_tc", "ksg_full_tc")}
    summary[str(d)]["ksg_tree1_time_s_median"] = float(np.median([x["ksg_tree1_time_s"] for x in rows]))

status = git("status", "--porcelain", "--untracked-files=no")
record = dict(
    description=__doc__.split("\n\n")[0],
    reference=dict(file=os.path.basename(ref_path), sha256=sha256(ref_path), generated_at=ref["generated_at"],
                   note="VDC and full chain-rule KSG values are those of the reference run; its host is not recorded"),
    settings=dict(rho=rho, n=n, ksg_k=k, seed=seed, n_trials=ref["n_trials"], dimensions=ref["dimensions"],
                  trial_seed="42 + 10000 d + 1000 t", pair_seed="trial_seed + 10000 i"),
    library=dict(commit=git("rev-parse", "HEAD"), tracked_files_modified=None if status is None else bool(status),
                 vdc_version=vdc.__version__),
    execution=dict(host=socket.gethostname(), cpu=cpu_model(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                   threads={v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                   python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                   finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")),
    trials=trials, summary=summary)
with open(out_path, "w") as f:
    json.dump(record, f, indent=2)
    f.write("\n")
print(json.dumps(summary, indent=2))
