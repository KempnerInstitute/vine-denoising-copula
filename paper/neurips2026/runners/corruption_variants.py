#!/usr/bin/env python3
"""Compare the retained training-corruption variants of the edge estimator (paper appendix).

The protocol was fixed before the run (paper repository, analysis/corruption_variants_20261028/PROTOCOL.md).
Every checkpoint scores identical inputs:
  E1a  the nine Table 1 cases: ISE against exact cell averages, held-out KL, exact grid Kendall-tau error;
  E1b  the nine out-of-distribution cases: held-out KL, ISE against exact cell averages;
  E2   a common panel of real-data vine edges: MAF data, one D-vine order, reference TLL pair copulas whose
       h-functions give every checkpoint the same edge inputs; held-out edge log-density by tree group;
  E3   each checkpoint's own D-vine on the MAF data, as run_uci_maf.py: held-out NLL in bits/dim.
The first checkpoint must be the deployed one; it is checked against the paper's saved records
(../tables/inputs) and the run stops before any alternative is scored if it differs.

Usage:
  corruption_variants.py --data-root DIR --output NEW_DIR --expected-module-root ROOT \
      --checkpoint uniform=PATH --checkpoint direct=PATH ...
Writes only into the new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "tables" / "inputs"
sys.path.insert(0, str(HERE))

M, K = 64, 30
DEPLOYED_SHA256 = "8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f"
T1_CASES = [("indep", [], 0), ("gaussian", [[.7]], 0), ("gaussian", [[-.7]], 0), ("student", [[.7], [5.]], 0),
            ("clayton", [[3.]], 0), ("clayton", [[3.]], 90), ("gumbel", [[2.5]], 0), ("joe", [[3.]], 0),
            ("frank", [[5.]], 0)]
T1_SEEDS, T1_TEST_OFFSET, T1_N, T1_N_TEST = [101, 202, 303, 404, 505], 777_777, 2000, 20000
OOD_CASES = [("bb6", [2.0, 1.5], 0, "BB6(theta=2.0, delta=1.5)"), ("bb6", [3.0, 2.0], 0, "BB6(theta=3.0, delta=2.0)"),
             ("bb8", [3.0, 0.8], 0, "BB8(theta=3.0, delta=0.8)"), ("bb8", [5.0, 0.6], 0, "BB8(theta=5.0, delta=0.6)"),
             ("tawn", [0.7, 1.0, 2.5], 0, "Tawn(theta=2.5, psi1=0.7, psi2=1.0)"),
             ("plackett", [5.0], 0, "Plackett(theta=5.0)"), ("plackett", [20.0], 0, "Plackett(theta=20.0)"),
             ("bb6", [2.0, 1.5], 90, "BB6-90(theta=2.0, delta=1.5)"), ("bb8", [3.0, 0.8], 90, "BB8-90(theta=3.0, delta=0.8)")]
OOD_SEEDS, OOD_TEST_OFFSET, OOD_N = [7, 17, 42], 10000, 2000
DATASETS = ["power", "gas", "hepmass", "miniboone"]   # the UCI runner's order (its row RNG runs through them)
GROUPS = [("T1", 1, 1), ("T2-T3", 2, 3), ("T4+", 4, 10 ** 6)]
N_BOOT, BOOT_SEED = 10_000, 20261028


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nll(vals):
    return float(-np.mean(np.log(np.maximum(vals, 1e-300))))


def logd(vals):
    return np.log(np.maximum(np.asarray(vals, dtype=np.float64), 1e-300))


def cellavg_from_cdf(cdf):
    """Exact cell averages m^2 P(cell) of a copula from its CDF on the grid corners (exact boundaries)."""
    e = np.arange(M + 1) / M
    U, V = np.meshgrid(e, e, indexing="ij")
    C = np.asarray(cdf(U, V), dtype=float).reshape(M + 1, M + 1)
    C[0, :] = 0.0
    C[:, 0] = 0.0
    C[-1, :] = e
    C[:, -1] = e
    return np.maximum(C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1], 0.0) * M * M


def tau_exact_grid(d):
    """Exact Kendall tau of a piecewise-constant density: 4 E[C(U, V)] - 1."""
    p = np.asarray(d, dtype=float) / np.sum(d)
    C = np.zeros((M + 1, M + 1))
    C[1:, 1:] = np.cumsum(np.cumsum(p, axis=0), axis=1)
    return float(4.0 * np.sum(p * 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])) - 1.0)


def plackett_pdf(u, v, th):
    num = th * (1.0 + (th - 1.0) * (u + v - 2.0 * u * v))
    disc = np.maximum((1.0 + (th - 1.0) * (u + v)) ** 2 - 4.0 * th * (th - 1.0) * u * v, 1e-30)
    return num / np.power(disc, 1.5)


def plackett_cdf(u, v, th):
    s = 1.0 + (th - 1.0) * (u + v)
    return (s - np.sqrt(np.maximum(s * s - 4.0 * th * (th - 1.0) * u * v, 0.0))) / (2.0 * (th - 1.0))


def plackett_sample(n, th, rng):
    """Closed-form conditional inversion, as the out-of-distribution benchmark."""
    u, t = rng.uniform(size=n), rng.uniform(size=n)
    a = t * (1.0 - t)
    b = th + a * (th - 1.0) ** 2
    c = 2.0 * a * (u * th ** 2 + 1.0 - u) + th * (1.0 - 2.0 * a)
    d = np.sqrt(th) * np.sqrt(th + 4.0 * a * u * (1.0 - u) * (1.0 - th) ** 2)
    return np.stack([u, np.clip((c - (1.0 - 2.0 * t) * d) / (2.0 * b), 1e-10, 1.0 - 1e-10)], axis=1)


def boot_ci(diffs, rng):
    """Mean and 95% percentile-bootstrap interval of each row of `diffs` (k x n), with shared resamples."""
    diffs = np.atleast_2d(np.asarray(diffs, dtype=np.float64))
    n = diffs.shape[1]
    means = np.empty((diffs.shape[0], N_BOOT))
    for s in range(0, N_BOOT, 200):
        idx = rng.integers(0, n, size=(min(200, N_BOOT - s), n))
        means[:, s:s + idx.shape[0]] = diffs[:, idx].mean(axis=-1)
    lo, hi = np.quantile(means, [0.025, 0.975], axis=1)
    return [dict(mean=float(m), lo=float(a), hi=float(b)) for m, a, b in zip(diffs.mean(axis=1), lo, hi)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True, help="folder with uci/{power,gas,hepmass,miniboone}.npz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-module-root", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH; the first is the deployed model")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    a = p.parse_args()
    import pyvinecopulib as pv
    import torch
    import vdc
    import run_uci_maf as U
    from vdc.baselines.pyvinecopulib import fit_bicop
    from vdc.data.tabular import EmpiricalMarginals, maybe_load_uci
    from vdc.models.hfunc import HFuncLookup
    from vdc.pretrained import estimate_pair_density_from_samples, load_checkpoint_bundle
    from vdc.vine.api import VineCopulaModel
    from vdc.vine.vine_types import build_dvine_structure

    assert Path(vdc.__file__).resolve().is_relative_to(a.expected_module_root.resolve()), vdc.__file__
    ckpts = [tuple(s.split("=", 1)) for s in a.checkpoint]
    assert digest(ckpts[0][1]) == DEPLOYED_SHA256, "the first checkpoint must be the deployed model"
    if a.device == "cuda":
        assert torch.cuda.is_available(), "no silent CPU fallback"
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()

    models, manifest = {}, dict(
        protocol="analysis/corruption_variants_20261028/PROTOCOL.md (paper repository)",
        source_sha256=digest(__file__), module_path=str(Path(vdc.__file__).resolve()), vdc_version=vdc.__version__,
        host=platform.node(), job=os.environ.get("SLURM_JOB_ID"), device=a.device,
        gpu=torch.cuda.get_device_name() if a.device == "cuda" else None,
        versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                      pyvinecopulib=pv.__version__),
        precision=dict(dtype="float32", matmul_tf32=False, cudnn_tf32=True, torch_threads=1, grid=M, ipfp_iters=K),
        checkpoints={})
    for name, path in ckpts:
        bundle = load_checkpoint_bundle(Path(path), device=a.device)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        bundle.model.load_state_dict(ck["model_state_dict"], strict=True)
        bundle.model.eval()
        noise = ck["config"]["training"].get("hist_noise", {})
        manifest["checkpoints"][name] = dict(path=str(path), sha256=digest(path), step=int(re.search(r"step_(\d+)", Path(path).name).group(1)),
                                             hist_noise=noise, t_scalar_mode=ck["config"]["training"].get("t_scalar_mode"),
                                             projection_iters=int(ck["config"]["training"].get("projection_iters", K)))
        del ck
        models[name] = bundle
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["checkpoints"], indent=1), flush=True)

    def grid(bundle, x):
        with torch.no_grad():
            g = np.asarray(estimate_pair_density_from_samples(bundle, x, m=M, projection_iters=K), dtype=float)
        return g / g.mean()

    # ---------------------------------------------------------------- E1a, E1b inputs (shared)
    t1 = []
    for family, params, rotation in T1_CASES:
        bc = pv.Bicop(family=getattr(pv.BicopFamily, family), rotation=rotation,
                      **({"parameters": np.asarray(params)} if params else {}))
        truth = cellavg_from_cdf(lambda u, v: bc.cdf(np.stack([u.ravel(), v.ravel()], 1)))
        tau = 0.0 if family == "indep" else float(bc.tau)
        for s in T1_SEEDS:
            tr = np.clip(bc.simulate(T1_N, seeds=[s, s + 1]), 1e-10, 1 - 1e-10)
            te = np.clip(bc.simulate(T1_N_TEST, seeds=[s + T1_TEST_OFFSET, s + T1_TEST_OFFSET + 1]), 1e-10, 1 - 1e-10)
            t1.append(dict(family=family, rotation=rotation, seed=s, tr=tr, te=te, truth=truth, tau=tau,
                           nll_true=nll(bc.pdf(te))))
    ood = []
    for family, params, rotation, label in OOD_CASES:
        if family == "plackett":
            th = float(params[0])
            truth = cellavg_from_cdf(lambda u, v: plackett_cdf(u, v, th))
            sample = (lambda n, s, th=th: plackett_sample(n, th, np.random.default_rng(s)))
            pdf = (lambda x, th=th: plackett_pdf(x[:, 0], x[:, 1], th))
        else:
            bc = pv.Bicop.from_family(family=getattr(pv.BicopFamily, family), rotation=rotation,
                                      parameters=np.asarray(params, dtype=float).reshape(-1, 1))
            truth = cellavg_from_cdf(lambda u, v, bc=bc: bc.cdf(np.stack([u.ravel(), v.ravel()], 1)))
            sample = (lambda n, s, bc=bc: np.clip(np.asarray(bc.simulate(n, seeds=[s, s + 1]), dtype=float),
                                                  1e-10, 1 - 1e-10))
            pdf = (lambda x, bc=bc: bc.pdf(x))
        for s in OOD_SEEDS:
            te = sample(OOD_N, s + OOD_TEST_OFFSET)
            ood.append(dict(case=label, seed=s, tr=sample(OOD_N, s), te=te, truth=truth, nll_true=nll(pdf(te))))

    def score_t1(bundle):
        out = []
        for c in t1:
            d = grid(bundle, c["tr"])
            out.append(dict(family=c["family"], rotation=c["rotation"], seed=c["seed"],
                            kl=nll(HFuncLookup(d, use_spline=False).pdf(c["te"][:, 0], c["te"][:, 1])) - c["nll_true"],
                            ise=float(np.mean((d - c["truth"]) ** 2)), tau_err=abs(tau_exact_grid(d) - c["tau"])))
        return out

    def score_ood(bundle):
        out = []
        for c in ood:
            d = grid(bundle, c["tr"])
            lib = nll(HFuncLookup(d, use_spline=False).pdf(c["te"][:, 0], c["te"][:, 1]))
            out.append(dict(case=c["case"], seed=c["seed"], nll_lib=lib, kl=lib - c["nll_true"],
                            ise=float(np.mean((d - c["truth"]) ** 2))))
        return out

    # ---------------------------------------------------------------- MAF rows (as run_uci_maf.py)
    rng = np.random.default_rng(42)
    data, data_hashes = {}, {}
    for name in DATASETS:
        X_tr, X_te = maybe_load_uci(name, a.data_root)
        X_tr, X_te = np.asarray(X_tr, dtype=np.float64), np.asarray(X_te, dtype=np.float64)
        data_hashes[name] = {"train": U._array_sha256(X_tr), "test": U._array_sha256(X_te)}
        assert data_hashes[name] == U.MAF_ARRAY_SHA256[name], f"{name} is not the paper's MAF split"
        if X_tr.shape[0] > 5000:
            X_tr = X_tr[rng.choice(X_tr.shape[0], size=5000, replace=False)]
        if X_te.shape[0] > 50000:
            X_te = X_te[rng.choice(X_te.shape[0], size=50000, replace=False)]
        marg = EmpiricalMarginals.fit(X_tr)
        data[name] = (U._clip_unit(marg.transform(X_tr)), U._clip_unit(marg.transform(X_te)))

    def score_e3(bundle, name):
        U_tr, U_te = data[name]
        vine = VineCopulaModel(vine_type="dvine", m=M, device=a.device, projection_iters=K, hfunc_use_spline=False,
                               batch_edges=True, edge_batch_size=256)
        with torch.no_grad():
            vine.fit(U_tr, diffusion_model=bundle.model, diffusion=None, verbose=False)
            lp = np.asarray(vine.logpdf(U_te), dtype=np.float64)
        return lp, list(map(int, vine.structure.order))

    # ---------------------------------------------------------------- qualification with the deployed model
    dep = ckpts[0][0]
    saved_t1 = json.loads((INPUTS / "table1/table1_v2.json").read_text())["records"]
    saved_ood = {c["summary"]["name"]: c["summary"] for c in json.loads((INPUTS / "ood/e26_rescored.json").read_text())["e26"]["cases"]}
    saved_uci = {r["dataset"]: next(m for m in r["methods"] if m["method"] == "ours")["nll_bits_per_dim"]
                 for r in json.loads((INPUTS / "uci/e21_maf5k.json").read_text())["records"]}
    res_t1, res_ood, res_e3, orders = {dep: score_t1(models[dep])}, {dep: score_ood(models[dep])}, {}, {}
    q = dict(t1=[], ood=[], e3=[])
    for r, s in zip(res_t1[dep], saved_t1):
        assert (r["seed"], r["rotation"]) == (s["seed"], s["rotation"])
        v = s["methods"]["vdc"]
        q["t1"].append(dict(dkl=abs(r["kl"] - v["kl"]), rel_dise=abs(r["ise"] - v["ise"]) / v["ise"]))
    for label in dict.fromkeys(c["case"] for c in ood):
        rows = [r for r in res_ood[dep] if r["case"] == label]
        q["ood"].append(dict(case=label, dnll=abs(np.mean([r["nll_lib"] for r in rows]) - saved_ood[label]["vdc_nll_lib"]),
                             rel_dise=abs(np.mean([r["ise"] for r in rows]) - saved_ood[label]["vdc_ise_cellavg"])
                             / saved_ood[label]["vdc_ise_cellavg"]))
    res_e3[dep] = {}
    for name in DATASETS:
        lp, orders[name] = score_e3(models[dep], name)
        res_e3[dep][name] = lp
        got = U._nll_bits_from_logpdf_rows(lp, data[name][0].shape[1])
        q["e3"].append(dict(dataset=name, nll=got, saved=saved_uci[name], d=abs(got - saved_uci[name])))
    passed = (all(x["dkl"] <= 1e-5 and x["rel_dise"] <= 1e-4 for x in q["t1"])
              and all(x["dnll"] <= 1e-4 and x["rel_dise"] <= 1e-3 for x in q["ood"])
              and all(x["d"] <= 1e-3 for x in q["e3"]))
    summary_q = dict(passed=passed, max_t1_dkl=max(x["dkl"] for x in q["t1"]),
                     max_t1_rel_dise=max(x["rel_dise"] for x in q["t1"]),
                     max_ood_dnll=max(x["dnll"] for x in q["ood"]), max_ood_rel_dise=max(x["rel_dise"] for x in q["ood"]),
                     e3=q["e3"])
    (a.output / "qualification.json").write_text(json.dumps(dict(summary=summary_q, detail=q), indent=2) + "\n")
    print("qualification", json.dumps(summary_q, indent=1), flush=True)
    if not passed:
        sys.exit(3)

    # ---------------------------------------------------------------- E1 and E3 for the alternatives
    for name, _ in ckpts[1:]:
        res_t1[name], res_ood[name] = score_t1(models[name]), score_ood(models[name])
        res_e3[name] = {}
        for ds in DATASETS:
            res_e3[name][ds], order = score_e3(models[name], ds)
            assert order == orders[ds], "the D-vine order depends only on the data"
        print("E1, E3 done:", name, flush=True)

    # ---------------------------------------------------------------- E2 common real-data edge panel
    sources = [n for n, _ in ckpts] + ["tll_reference"]
    e2_rows, e2_edges = {}, {}
    for ds in DATASETS:
        U_tr, U_te = data[ds]
        order = orders[ds]
        assert order == list(map(int, build_dvine_structure(U_tr).order))
        X_tr, X_te = U_tr[:, order], U_te[:, order]
        d = X_tr.shape[1]
        acc = {s: {g: np.zeros(X_te.shape[0]) for g, _, _ in GROUPS} for s in sources}
        edge_means = {s: [] for s in sources}
        edge_tree = []
        a_tr, b_tr = [X_tr[:, j] for j in range(d - 1)], [X_tr[:, j + 1] for j in range(d - 1)]
        a_te, b_te = [X_te[:, j] for j in range(d - 1)], [X_te[:, j + 1] for j in range(d - 1)]
        for t in range(1, d):
            group = next(g for g, lo, hi in GROUPS if lo <= t <= hi)
            fits = []
            for j in range(d - t):
                ptr = np.clip(np.column_stack([a_tr[j], b_tr[j]]), 1e-10, 1 - 1e-10)
                pte = np.clip(np.column_stack([a_te[j], b_te[j]]), 1e-10, 1 - 1e-10)
                fit = fit_bicop(ptr, mode="nonparametric", selection_criterion="bic", allow_rotations=True,
                                num_threads=1).model
                fits.append((fit, ptr, pte))
                lps = {"tll_reference": logd(fit.pdf(pte))}
                for name, _ in ckpts:
                    g = grid(models[name], ptr)
                    lps[name] = logd(HFuncLookup(g, use_spline=False).pdf(pte[:, 0], pte[:, 1]))
                for s in sources:
                    acc[s][group] += lps[s]
                    edge_means[s].append(float(lps[s].mean()))
                edge_tree.append(t)
            if t < d - 1:   # pseudo-observations of tree t+1 from the reference h-functions
                a_tr = [np.asarray(fits[j][0].hfunc2(fits[j][1])) for j in range(d - t - 1)]
                b_tr = [np.asarray(fits[j + 1][0].hfunc1(fits[j + 1][1])) for j in range(d - t - 1)]
                a_te = [np.asarray(fits[j][0].hfunc2(fits[j][2])) for j in range(d - t - 1)]
                b_te = [np.asarray(fits[j + 1][0].hfunc1(fits[j + 1][2])) for j in range(d - t - 1)]
        e2_rows[ds], e2_edges[ds] = acc, dict(tree=edge_tree, **edge_means)
        print("E2 done:", ds, d * (d - 1) // 2, "edges", flush=True)

    # ---------------------------------------------------------------- statistics and outputs
    brng = np.random.default_rng(BOOT_SEED)
    alts = [n for n, _ in ckpts[1:]]
    ln2 = np.log(2.0)
    out = dict(status="complete", manifest=manifest, qualification=summary_q, data_sha256=data_hashes,
               dvine_order=orders, suites={}, wall_seconds=None)
    for suite, res, metrics in (("E1a_table1", res_t1, ("kl", "ise", "tau_err")), ("E1b_ood", res_ood, ("kl", "ise"))):
        s = {"means": {n: {m: float(np.mean([r[m] for r in res[n]])) for m in metrics} for n in res}, "vs_uniform": {}}
        for n in alts:
            diffs = [[r[m] - u[m] for r, u in zip(res[n], res[dep])] for m in metrics]
            cis = boot_ci(diffs, brng)
            s["vs_uniform"][n] = {m: dict(ci, alt_better=int(sum(x < 0 for x in dd)), n=len(dd))
                                  for m, ci, dd in zip(metrics, cis, diffs)}
        out["suites"][suite] = s
    e2, e3 = {"means_bits_per_dim": {}, "vs_uniform": {}}, {"nll_bits_per_dim": {}, "vs_uniform": {}}
    for ds in DATASETS:
        dim = data[ds][0].shape[1]
        nll_rows = {s: {g: -v / (dim * ln2) for g, v in e2_rows[ds][s].items()} for s in sources}
        for s in sources:
            nll_rows[s]["total"] = sum(nll_rows[s][g] for g, _, _ in GROUPS)
        keys = ["total"] + [g for g, _, _ in GROUPS]
        e2["means_bits_per_dim"][ds] = {s: {k: float(nll_rows[s][k].mean()) for k in keys} for s in sources}
        cis = boot_ci([nll_rows[n][k] - nll_rows[dep][k] for n in alts for k in keys], brng)
        tree = np.asarray(e2_edges[ds]["tree"])
        e2["vs_uniform"][ds] = {}
        for i, n in enumerate(alts):
            better = np.asarray(e2_edges[ds][n]) > np.asarray(e2_edges[ds][dep])   # higher held-out log-density
            e2["vs_uniform"][ds][n] = {k: cis[i * len(keys) + j] for j, k in enumerate(keys)}
            e2["vs_uniform"][ds][n]["edges_alt_better"] = {g: [int(better[(tree >= lo) & (tree <= hi)].sum()),
                                                               int(((tree >= lo) & (tree <= hi)).sum())]
                                                           for g, lo, hi in GROUPS}
        rows = {n: -res_e3[n][ds] / (dim * ln2) for n in res_e3}
        e3["nll_bits_per_dim"][ds] = {n: float(v.mean()) for n, v in rows.items()}
        e3["vs_uniform"][ds] = dict(zip(alts, boot_ci([rows[n] - rows[dep] for n in alts], brng)))
    out["suites"]["E2_edge_panel"], out["suites"]["E3_vine"] = e2, e3
    out["per_fit"] = dict(E1a_table1=res_t1, E1b_ood=res_ood)
    np.savez_compressed(a.output / "rows.npz",
                        **{f"e3_{ds}_{n}": res_e3[n][ds] for n in res_e3 for ds in DATASETS},
                        **{f"e2_{ds}_{s}_{g}": e2_rows[ds][s][g] for ds in DATASETS for s in sources for g, _, _ in GROUPS},
                        **{f"e2edge_{ds}_{s}": np.asarray(e2_edges[ds][s]) for ds in DATASETS for s in sources},
                        **{f"e2edge_{ds}_tree": np.asarray(e2_edges[ds]["tree"]) for ds in DATASETS})
    out["rows_sha256"] = digest(a.output / "rows.npz")
    out["wall_seconds"] = time.perf_counter() - started
    (a.output / "results.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: out["suites"][k].get("means", out["suites"][k].get("means_bits_per_dim",
                      out["suites"][k].get("nll_bits_per_dim"))) for k in out["suites"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
