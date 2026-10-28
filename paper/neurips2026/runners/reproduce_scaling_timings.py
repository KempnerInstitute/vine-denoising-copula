#!/usr/bin/env python3
"""Portable fixed-thread timing protocol for the original VDC checkpoint.

Each fit runs in a separate sequential worker. Defaults reproduce the data,
configuration and method-order protocol of the camera-ready timing study;
elapsed times depend on hardware. --device cpu and small dimensions permit a
workflow smoke test, not a replay of the paper's H100 timing measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

CHECKPOINT = "8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(a):
    import gc
    import numpy as np
    import scipy
    import pyvinecopulib as pv
    import torch
    import vdc
    from vdc.baselines.pyvinecopulib import fit_vinecop
    from vdc.train.unified_trainer import build_model
    from vdc.vine.api import VineCopulaModel
    assert Path(vdc.__file__).resolve().is_relative_to(a.expected_module_root.resolve())
    assert sha(a.checkpoint) == CHECKPOINT
    u = np.load(a.input, allow_pickle=False)
    assert u.shape == (a.n_train + a.n_test, a.dimension)
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(a.cpu_threads)
    torch.manual_seed(202610151)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    def sync():
        if a.device == "cuda":
            torch.cuda.synchronize()
    config = dict(device=a.device, cpu_threads=a.cpu_threads, m=64,
                  projection_iters=15, edge_batch_size=256, matrix_tf32=False,
                  convolution_tf32=True, cudnn_benchmark=False)
    result = dict(status="running", method=a.method, d=a.dimension, seed=a.seed,
                  input_sha256=sha(a.input), checkpoint_sha256=CHECKPOINT,
                  script_sha256=sha(__file__), module_path=str(Path(vdc.__file__).resolve()),
                  hostname=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  configuration=config, versions=dict(python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                  pyvinecopulib=pv.__version__), gpu=torch.cuda.get_device_name() if a.device == "cuda" else None)
    save(a.output / "results.json", result)
    if a.method.startswith("vdc_"):
        started = time.perf_counter()
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model = build_model(ck["config"]["model"]["type"], ck["config"], torch.device("cpu"))
        model.load_state_dict(ck["model_state_dict"], strict=True)
        del ck
        model.eval().to(a.device)
        sync()
        result["load_seconds"] = time.perf_counter() - started
        kwargs = dict(vine_type="dvine", order=None, m=64, device=a.device,
                      projection_iters=15, hfunc_use_spline=False, batch_edges=True,
                      edge_batch_size=256, hfunc_vectorized_inverse=True,
                      fast_fit=a.method == "vdc_fast")
        config.update(kwargs)
        warm = VineCopulaModel(**kwargs)
        warm.fit(u[:min(200, a.n_train), :2], diffusion_model=model, verbose=False)
        sync()
        del warm
        gc.collect()
        if a.device == "cuda":
            torch.cuda.empty_cache()
        vine = VineCopulaModel(**kwargs)
        sync()
        started = time.perf_counter()
        vine.fit(u[:a.n_train], diffusion_model=model, diffusion=None, verbose=False)
        sync()
        result["fit_seconds"] = time.perf_counter() - started
        h = hashlib.sha256()
        for tree, copulas in zip(vine.structure.trees, vine.vine.pair_copulas):
            assert len(tree.edges) == len(copulas)
            for edge, tau, cop in zip(tree.edges, tree.tau_values, copulas):
                h.update(json.dumps([edge[0], edge[1], sorted(edge[2])]).encode())
                for x in [np.asarray([tau], dtype=np.float64), cop.density_grid,
                          cop.hfunc.density_grid, cop.hfunc.h_u_given_v_grid,
                          cop.hfunc.h_v_given_u_grid]:
                    x = np.ascontiguousarray(x)
                    assert np.isfinite(x).all()
                    h.update(str((x.dtype.str, x.shape)).encode())
                    h.update(memoryview(x).cast("B"))
        probes = min(128, a.n_test)
        uniforms = np.random.default_rng(202610152 + a.dimension).uniform(1e-6, 1-1e-6, (probes, a.dimension))
        arrays = dict(logpdf=np.asarray(vine.logpdf(u[a.n_train:])),
                      pit=np.asarray(vine.rosenblatt(u[a.n_train:a.n_train+probes])),
                      generated=np.asarray(vine.inverse_rosenblatt(uniforms)))
        assert all(np.isfinite(x).all() for x in arrays.values())
        np.savez_compressed(a.output / "diagnostics.npz", **arrays)
        result.update(state_sha256=h.hexdigest(), finite_likelihood=True,
                      nll_bits_dim=-float(arrays["logpdf"].mean()) / (a.dimension*np.log(2)),
                      diagnostics_sha256=sha(a.output / "diagnostics.npz"))
    else:
        threads = int(a.method.split("_")[1])
        kwargs = dict(mode="nonparametric", num_threads=threads, nonparametric_grid_size=30)
        config.update(kwargs, structure="automatic regular vine")
        warm = fit_vinecop(u[:min(200, a.n_train), :2], **kwargs)
        del warm
        gc.collect()
        started = time.perf_counter()
        vine = fit_vinecop(u[:a.n_train], **kwargs)
        result["fit_seconds"] = time.perf_counter() - started
        ll = float(vine.loglik(u[a.n_train:], num_threads=threads))
        finite = bool(np.isfinite(ll))
        result.update(finite_likelihood=finite, nonfinite_loglik=None if finite else repr(ll),
                      nll_bits_dim=-ll / (a.n_test*a.dimension*np.log(2)) if finite else None)
    result["status"] = "complete"
    save(a.output / "results.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--expected-module-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--dimensions", type=int, nargs="+", default=[100, 200, 500])
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 42])
    p.add_argument("--cpu-threads", type=int, default=16)
    p.add_argument("--tll-threads", type=int, nargs="+", default=[8, 16])
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--dimension", type=int, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    p.add_argument("--method", help=argparse.SUPPRESS)
    a = p.parse_args()
    assert "login" not in platform.node(), "Run on a compute node"
    if a.worker:
        worker(a)
        return
    assert min(a.dimensions) >= 2 and min(a.cpu_threads, *a.tll_threads, a.n_train, a.n_test) > 0
    assert max(a.tll_threads) <= a.cpu_threads
    import numpy as np
    import vdc
    from vdc.data.generators import generate_clayton_vine
    assert Path(vdc.__file__).resolve().is_relative_to(a.expected_module_root.resolve())
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", scope="Portable protocol replay; hardware-dependent times, no modification of paper results.",
                  script_sha256=sha(__file__), rows=[], exact_comparisons=[])
    save(a.output / "results.json", result)
    rng = random.Random(20261018)
    try:
        for d in a.dimensions:
            for seed in a.seeds:
                u = generate_clayton_vine(n=a.n_train+a.n_test, d=d, theta=2.0, seed=seed+1000*d)
                data = (a.output / f"inputs_d{d}_s{seed}.npy").resolve()
                np.save(data, u)
                methods = ["vdc_original", "vdc_fast", *[f"tll_{n}" for n in a.tll_threads]]
                rng.shuffle(methods)
                outputs = {}
                for method in methods:
                    out = (a.output / f"d{d}_s{seed}_{method}").resolve()
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--checkpoint", str(a.checkpoint.resolve()), "--expected-module-root", str(a.expected_module_root.resolve()),
                           "--output", str(out), "--device", a.device, "--cpu-threads", str(a.cpu_threads),
                           "--n-train", str(a.n_train), "--n-test", str(a.n_test), "--dimension", str(d),
                           "--seed", str(seed), "--method", method, "--input", str(data)]
                    env = {**os.environ, "OMP_NUM_THREADS": str(a.cpu_threads),
                           "MKL_NUM_THREADS": str(a.cpu_threads), "OPENBLAS_NUM_THREADS": "1"}
                    with (a.output / f"d{d}_s{seed}_{method}.log").open("w") as log:
                        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    row = json.loads((out / "results.json").read_text())
                    result["rows"].append(row)
                    outputs[method] = out
                    save(a.output / "results.json", result)
                old = json.loads((outputs["vdc_original"] / "results.json").read_text())
                new = json.loads((outputs["vdc_fast"] / "results.json").read_text())
                assert old["state_sha256"] == new["state_sha256"]
                with np.load(outputs["vdc_original"] / "diagnostics.npz") as x, np.load(outputs["vdc_fast"] / "diagnostics.npz") as y:
                    for key in x.files:
                        np.testing.assert_array_equal(x[key], y[key])
                result["exact_comparisons"].append(dict(d=d, seed=seed, passed=True))
                save(a.output / "results.json", result)
    except Exception as e:
        result.update(status="failed", error=repr(e))
        save(a.output / "results.json", result)
        raise
    result["status"] = "complete"
    save(a.output / "results.json", result)


if __name__ == "__main__":
    main()
