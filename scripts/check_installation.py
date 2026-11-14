#!/usr/bin/env python3
"""Exercise installed pretrained models, vine inference and sampler fallback.

Run outside the checkout with the installed interpreter. Checkpoints are
downloaded and hash-verified by the public loader. This is a release integration
check, not an accuracy benchmark; allow several GB of RAM and checkpoint cache.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.special import ndtr

import vdc
from vdc import VineCopulaModel, estimate_pair_density_from_samples, load_pretrained_model
from vdc.pretrained import linear_grid_settings_for

OPTIONAL_MODULES = ("pyvinecopulib", "matplotlib", "h5py", "hydra", "sklearn")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--require-minimal", action="store_true")
    parser.add_argument("--snapshot", type=Path, help="Save numerical arrays for migration checks")
    args = parser.parse_args()
    if args.require_minimal:
        present = [name for name in OPTIONAL_MODULES if importlib.util.find_spec(name) is not None]
        if present:
            raise RuntimeError(f"Minimal environment contains optional dependencies: {present}")
        if not Path(vdc.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError("VDC must be installed in this environment, not imported from a checkout")

    torch.set_num_threads(2)
    rng = np.random.default_rng(20261113)
    z = rng.normal(size=(256, 3))
    z[:, 1] = 0.6 * z[:, 0] + 0.8 * z[:, 1]
    z[:, 2] = -0.4 * z[:, 1] + np.sqrt(0.84) * z[:, 2]
    uniforms = ndtr(z)
    evaluation = rng.uniform(0.02, 0.98, (32, 3))
    model_ids = ["vdc-denoiser-m64-v1"]
    if args.all_models:
        model_ids += ["vdc-denoiser-m64-v2", "vdc-denoiser-m128-v2"]
    report = {"vdc": vdc.__version__, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
              "device": "cpu", "threads": 2, "models": []}
    snapshots = {}
    for model_id in model_ids:
        torch.manual_seed(7)
        bundle = load_pretrained_model(model_id, device="cpu")
        m = bundle.manifest["grid_size"]
        density = estimate_pair_density_from_samples(bundle, uniforms[:, :2])
        assert density.shape == (m, m)
        assert np.isfinite(density).all() and (density > 0).all()
        if linear_grid_settings_for(bundle) is None:
            vine = VineCopulaModel(vine_type="dvine", device="cpu", m=m, batch_edges=True,
                                  edge_batch_size=16, hfunc_use_spline=False,
                                  hfunc_vectorized_inverse=True, fast_fit=True)
        else:
            vine = VineCopulaModel.from_pretrained(bundle)
        vine.fit(uniforms, bundle.model, diffusion=bundle.diffusion, verbose=False)
        logpdf = vine.logpdf(evaluation)
        pit = vine.rosenblatt(evaluation)
        samples = vine.simulate(32, seed=19)
        assert logpdf.shape == (32,) and np.isfinite(logpdf).all()
        assert pit.shape == samples.shape == (32, 3)
        assert np.isfinite(pit).all() and ((pit >= 0) & (pit <= 1)).all()
        assert np.isfinite(samples).all() and ((samples >= 0) & (samples <= 1)).all()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vine.pkl"
            vine.save(path)
            restored = VineCopulaModel.load(path)
            np.testing.assert_array_equal(restored.logpdf(evaluation), logpdf)
            sampler = restored.prepare_sampler(compiler="vdc-deliberately-missing-compiler",
                                                cache_dir=Path(tmp) / "kernel")
            assert sampler.backend == "numpy" and sampler.fallback_reason
            np.testing.assert_array_equal(sampler.simulate(32, seed=19), samples)
        for name, value in {"density": density, "logpdf": logpdf,
                            "pit": pit, "samples": samples}.items():
            snapshots[model_id + "/" + name] = value
        report["models"].append({"model_id": model_id, "sha256": bundle.manifest["sha256"],
                                 "grid_size": m, "vine_dimension": 3, "sample_count": 32,
                                 "mean_logpdf": float(logpdf.mean()),
                                 "save_load_exact": True, "numpy_fallback_exact": True})
        print(f"PASS {model_id}: pair density, vine, persistence, NumPy fallback", flush=True)
        del bundle, vine, restored, sampler
        gc.collect()
    if args.require_minimal:
        assert not any(name in sys.modules for name in OPTIONAL_MODULES)
        assert "vdc.train.unified_trainer" not in sys.modules
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.snapshot, **snapshots)


if __name__ == "__main__":
    main()
