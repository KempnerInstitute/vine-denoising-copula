#!/usr/bin/env python3
"""Reproduce the precision table of GitHub issue #8 (pyvinecopulib PR 217) and test its cause.

The issue reports that vdc-denoiser-m64-v1 has 10-25x larger pdf IAE than TLL and spikes at the
anti-diagonal corners for independent input. In May 2026 the vdc wheel omitted vdc.inference, so
pyvinecopulib's shim replaced vdc.inference.density.scatter_to_hist (density, mean 1) with
vdc.data.hist.scatter_to_hist (probability mass, sum 1). This script runs the issue's benchmark
(families, n, seeds, metric) with:
  tll        pyvinecopulib C++ TLL (default controls)
  v1         vdc-denoiser-m64-v1 through vdc's own histogram (as released)
  v1_shim    vdc-denoiser-m64-v1 with the shim's mass-normalized histogram (the issue's setting)
  v2_m64     vdc-denoiser-m64-v2 (linear grid)
  v2_m128    vdc-denoiser-m128-v2 (linear grid)
v1 grids are evaluated as pyvinecopulib does: replicate-padded to [0, cell centres, 1] and
interpolated bilinearly. IAE = mean |fit - true| on 10,000 iid points in [0.02, 0.98]^2.

usage: reproduce.py --output results.json [--methods v1,v1_shim] [--seeds 42]
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pyvinecopulib as pv
import scipy
import torch
from scipy.interpolate import RegularGridInterpolator

import vdc
import vdc.inference.density as vdens
from vdc.data.hist import scatter_to_hist as hist_mass
from vdc.pretrained import (
    estimate_pair_copula_from_samples,
    estimate_pair_density_from_samples,
    load_pretrained_model,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seeds", default="42,43,44,45,46")
parser.add_argument("--methods", default="tll,v1,v1_shim,v2_m64,v2_m128")
parser.add_argument("--reference", type=Path, help="Compare every accuracy field with a saved run")
args = parser.parse_args()
selected = set(args.methods.split(","))
if not selected or not selected <= {"tll", "v1", "v1_shim", "v2_m64", "v2_m128"}:
    parser.error("unknown or empty method selection")
if args.output.suffix != ".json":
    parser.error("--output must end in .json")
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
OUT = str(args.output)
FAM = pv.BicopFamily
SPECS = [("gaussian", FAM.gaussian, 0.3), ("gaussian", FAM.gaussian, 0.7),
         ("clayton", FAM.clayton, 2.0), ("clayton", FAM.clayton, 5.0)]
NS = [500, 2000, 10000]
SEEDS = [int(s) for s in args.seeds.split(",")]
if not SEEDS or len(set(SEEDS)) != len(SEEDS):
    parser.error("seeds must be nonempty and distinct")
M_EVAL = 10000
record = {"vdc_version": vdc.__version__, "torch": torch.__version__, "pyvinecopulib": pv.__version__,
          "python": platform.python_version(), "scipy": scipy.__version__,
          "numpy": np.__version__, "device": "cpu", "threads": 4, "rows": [], "independence": [],
          "errors": [], "input_hashes": [], "models": {}, "seeds": SEEDS, "methods": sorted(selected),
          "tll_controls": {"method": "constant", "grid_size": 30, "multiplier": 1.0},
          "evaluation": "mean absolute pdf error on uniform [0.02, 0.98]^2 points; not full-square IAE"}

LOG = open(str(args.output.with_suffix(".jsonl")), "w")   # every row as soon as it exists


def log_row(kind, row):
    LOG.write(json.dumps({"kind": kind, **row}) + "\n")
    LOG.flush()


bundles = {}
for key, mid in [("v1", "vdc-denoiser-m64-v1"), ("v2_m64", "vdc-denoiser-m64-v2"),
                 ("v2_m128", "vdc-denoiser-m128-v2")]:
    if key not in selected and not (key == "v1" and "v1_shim" in selected):
        continue
    try:
        t0 = time.time()
        bundles[key] = load_pretrained_model(mid, device="cpu")
        record["models"][key] = {"model_id": mid, "sha256": bundles[key].manifest["sha256"]}
        print(f"loaded {mid} in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:  # recorded, never silently dropped
        record["errors"].append(f"load {mid}: {e!r}")
        print("LOAD FAILED", mid, repr(e), flush=True)


def interp(grid):
    """pyvinecopulib's evaluation of a vdc cell grid: replicate pad, bilinear."""
    m = grid.shape[0]
    pts = np.concatenate([[0.0], (np.arange(m) + 0.5) / m, [1.0]])
    f = RegularGridInterpolator((pts, pts), np.pad(np.asarray(grid, float), 1, mode="edge"))
    return lambda u: f(u)


def v1_grid(u, shim):
    orig = vdens.scatter_to_hist
    if shim:
        vdens.scatter_to_hist = hist_mass   # what pyvinecopulib's shim installs
    try:
        return estimate_pair_density_from_samples(bundles["v1"], u)
    finally:
        vdens.scatter_to_hist = orig


def fitters():
    out = {"tll": lambda u: (lambda c: (lambda x: c.pdf(x)))(
        pv.Bicop.from_data(u, controls=pv.FitControlsBicop(family_set=[FAM.tll])))}
    if "v1" in bundles:
        out["v1"] = lambda u: interp(v1_grid(u, False))
        out["v1_shim"] = lambda u: interp(v1_grid(u, True))
    for key in ("v2_m64", "v2_m128"):
        if key in bundles:
            out[key] = (lambda b: lambda u: (lambda c: (lambda x: c.pdf(x[:, 0], x[:, 1])))(
                estimate_pair_copula_from_samples(b, u)))(bundles[key])
    return {key: value for key, value in out.items() if key in selected}


FITS = fitters()
if set(FITS) != selected:
    raise RuntimeError(f"Failed to load requested models: {record['errors']}")


def record_inputs(label, seed, n, u, ue):
    def digest(a):
        return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()
    record["input_hashes"].append({"case": label, "seed": seed, "n": n,
                                    "train_sha256": digest(u), "eval_sha256": digest(ue)})

for seed in SEEDS:
    rng = np.random.default_rng(seed)   # same draw order as bench_torch_precision.py
    for fam_label, fam, par in SPECS:
        cop = pv.Bicop(family=fam, parameters=np.array([[par]]))
        for n in NS:
            sim_seeds = [int(x) for x in rng.integers(1, 2**31 - 1, size=3)]
            u = cop.simulate(n, seeds=sim_seeds)
            ue = np.random.default_rng(int(rng.integers(1, 2**31 - 1))).uniform(0.02, 0.98, (M_EVAL, 2))
            truth = cop.pdf(ue)
            record_inputs(f"{fam_label}({par:g})", seed, n, u, ue)
            for name, fit in FITS.items():
                try:
                    t0 = time.time()
                    pdf = fit(u)
                    fit_s = time.time() - t0
                    iae = float(np.mean(np.abs(pdf(ue) - truth)))
                except Exception as e:
                    record["errors"].append(f"{name} {fam_label} {par} n={n} seed={seed}: {e!r}")
                    continue
                record["rows"].append({"seed": seed, "family": fam_label, "param": par, "n": n,
                                       "method": name, "iae": iae, "fit_s": fit_s})
                log_row("bicop", record["rows"][-1])
            print(f"seed {seed} {fam_label}({par}) n={n} done", flush=True)

# independence probe (the issue's "c = 1 everywhere" check)
for seed in SEEDS:
    rng = np.random.default_rng(1000 + seed)
    ue = rng.uniform(0.02, 0.98, (M_EVAL, 2))
    for n in (2000, 10000):
        u = rng.uniform(size=(n, 2))
        record_inputs("independence", seed, n, u, ue)
        for name, fit in FITS.items():
            pdf = fit(u)(ue)
            row = {"seed": seed, "n": n, "method": name, "iae": float(np.mean(np.abs(pdf - 1))),
                   "max_eval": float(pdf.max())}
            if name.startswith("v1"):
                g = v1_grid(u, name == "v1_shim")
                row.update(grid_max=float(g.max()), grid_min=float(g.min()),
                           corner_0_1=float(g[0, -1]), corner_1_0=float(g[-1, 0]),
                           corner_0_0=float(g[0, 0]), corner_1_1=float(g[-1, -1]))
            record["independence"].append(row)
            log_row("independence", row)
    print(f"independence seed {seed} done", flush=True)

LOG.close()
with open(OUT, "w") as handle:
    json.dump(record, handle, indent=2)
if record["errors"] or len(record["rows"]) != 12 * len(SEEDS) * len(selected):
    raise RuntimeError(f"Incomplete benchmark: {record['errors']}")
if len(record["independence"]) != 2 * len(SEEDS) * len(selected):
    raise RuntimeError("Incomplete independence probe")

if args.reference:
    reference = json.loads(args.reference.read_text())
    checked = 0
    for group, keys in (("rows", ("seed", "family", "param", "n", "method")),
                        ("independence", ("seed", "n", "method"))):
        expected = {tuple(r[k] for k in keys): r for r in reference[group]}
        for row in record[group]:
            previous = expected[tuple(row[k] for k in keys)]
            for key, value in previous.items():
                if key not in keys and key != "fit_s":
                    np.testing.assert_allclose(row[key], value, atol=5e-6, rtol=5e-4,
                                               err_msg=f"{group} {keys}: {key}")
            checked += 1
    print(f"Reference comparison passed: {checked} evaluations", flush=True)


def table(rows, key="iae"):
    meth = list(FITS)
    print(f"{'case':24s}" + "".join(f"{m:>11s}" for m in meth))
    cases = sorted({(r["family"], r["param"], r["n"]) for r in rows}, key=lambda c: (c[0] != "gaussian", c[1], c[2]))
    for c in cases:
        cells = []
        for m in meth:
            v = [r[key] for r in rows if (r["family"], r["param"], r["n"]) == c and r["method"] == m]
            cells.append(f"{np.mean(v):11.4f}" if v else f"{'-':>11s}")
        print(f"{c[0]}({c[1]:g}) n={c[2]:<6d}".ljust(24) + "".join(cells))


print("\n== seed 42 (the issue's draw): pdf IAE")
table([r for r in record["rows"] if r["seed"] == 42])
print(f"\n== mean over seeds {SEEDS}: interior pdf absolute error")
table(record["rows"])
print("\n== independence (mean over seeds): IAE vs 1, max on eval points; v1 grid corners")
for n in (2000, 10000):
    for m in FITS:
        rr = [r for r in record["independence"] if r["n"] == n and r["method"] == m]
        s = f"n={n:<6d} {m:9s} IAE {np.mean([r['iae'] for r in rr]):.4f}  max {np.mean([r['max_eval'] for r in rr]):8.3f}"
        if "grid_max" in rr[0]:
            s += (f"  grid max {np.mean([r['grid_max'] for r in rr]):8.3f}  corners (0,1) "
                  f"{np.mean([r['corner_0_1'] for r in rr]):8.3f} (1,0) {np.mean([r['corner_1_0'] for r in rr]):8.3f}")
        print(s)
print("\nerrors:", record["errors"] or "none")
