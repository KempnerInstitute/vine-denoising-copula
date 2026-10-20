#!/usr/bin/env python3
"""Compute the real data drawn in the camera-ready Fig. 1 and the appendix diagnostics figure.

Everything shown is a genuine output: pyvinecopulib samples, their 64x64 rank histograms, and
densities predicted by the frozen paper checkpoint through the pinned library's public API
(vdc.pretrained.estimate_pair_density_from_samples; projection_iters=0 gives the mass-normalized
raw output, 30 the deployed IPFP projection). Written to one small .npz plus a provenance JSON;
the drawing script (fig1_overview_cr.py) reads only that file.

Contents
  zoo_<k>          four pretraining-zoo samples (n=250) for the pretraining icon
  pre_*            pretraining example, Clayton theta=3, n=1500, seed 123 (the sample of the earlier
                   IPFP panel): histogram, uniform-mixed histogram (alpha=0.3), raw and projected
                   outputs, true density on the cell centres
  dep<i>_*         three edges of one tree for the deployment strip (samples, histograms, densities)
  cx_*             Complex-X example of Fig. 1(d) (protocol of fig1e_complex_x.py): true density, a fresh
                   sample (seed CX_SEED), pyvine-TLL and VDC densities, ISE values
  loss_*           training-loss history of the deployed run (downsampled), for the appendix

The training log (--loss-csv) is not distributed; --loss-npz copies the loss_* arrays from an existing
fig1_overview_data.npz instead, so the model outputs can be recomputed without it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

M = 64
# Seed of the Complex-X example. Checkpoint selection (scripts/model_selection.py) drew its Complex-X
# sample with default_rng(123), so the figure uses a seed that selection never saw.
CX_SEED = 20261020


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--loss-csv", type=Path, help="loss_history.csv of the deployed training run")
    ap.add_argument("--loss-npz", type=Path, help="take the loss_* arrays from this fig1_overview_data.npz instead")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if (a.loss_csv is None) == (a.loss_npz is None):
        ap.error("give exactly one of --loss-csv and --loss-npz")
    sys.path.insert(0, str(a.lib))
    import torch
    import pyvinecopulib as pv
    import vdc
    from vdc.data.complex_copulas import complex_copula_density_grid
    from vdc.inference.density import scatter_to_hist
    from vdc.pretrained import estimate_pair_density_from_samples, load_checkpoint_bundle
    from vdc.vine.copula_diffusion import DiffusionCopulaModel

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_checkpoint_bundle(a.checkpoint, device=dev)
    u = (np.arange(M) + 0.5) / M
    UU, VV = np.meshgrid(u, u, indexing="ij")
    grid_pts = np.c_[UU.ravel(), VV.ravel()]
    out = {}

    def bicop(fam, params, rot=0):
        return pv.Bicop(family=getattr(pv.BicopFamily, fam), rotation=rot, parameters=np.array(params, float).reshape(-1, 1))

    def hist(x):
        h = np.asarray(scatter_to_hist(np.clip(x, 1e-10, 1 - 1e-10), m=M, reflect=True), float)  # as the network sees it
        return h / h.mean()

    def vdc_density(x, k):
        d = np.asarray(estimate_pair_density_from_samples(bundle, np.clip(x, 1e-10, 1 - 1e-10), projection_iters=k), float)
        return d / d.mean()

    # pretraining-zoo icon samples
    zoo = [("gaussian", [0.7], 0), ("clayton", [3.0], 0), ("gumbel", [2.0], 90), ("frank", [-8.0], 0)]
    for k, (fam, par, rot) in enumerate(zoo):
        out[f"zoo_{k}"] = bicop(fam, par, rot).simulate(250, seeds=[11 + k])

    # pretraining example
    cop = bicop("clayton", [3.0])
    x = cop.simulate(1500, seeds=[123])
    h = hist(x)
    out["pre_sample"] = x
    out["pre_hist"] = h
    out["pre_hist_mixed"] = 0.7 * h + 0.3
    out["pre_raw"] = vdc_density(x, 0)
    out["pre_proj"] = vdc_density(x, 30)
    tru = np.asarray(cop.pdf(grid_pts), float).reshape(M, M)
    out["pre_true"] = tru / tru.mean()

    # three edges of one tree
    edges = [("gaussian", [0.6], 0, 1000), ("gumbel", [2.0], 90, 1000), ("student", [-0.5, 4.0], 0, 1000)]
    for i, (fam, par, rot, n) in enumerate(edges):
        xe = bicop(fam, par, rot).simulate(n, seeds=[21 + i])
        out[f"dep{i}_sample"] = xe
        out[f"dep{i}_hist"] = hist(xe)
        out[f"dep{i}_proj"] = vdc_density(xe, 30)

    # Complex-X example (same protocol as fig1e_complex_x.py)
    d_true = np.asarray(complex_copula_density_grid("x", {"sigma": 0.03, "w2": 1.0}, m=M, device=torch.device("cpu"),
                                                    projection_iters=80), float)
    d_true = d_true / d_true.mean()
    xc = DiffusionCopulaModel.sample_from_density(density=d_true, n_samples=2000, rng=np.random.default_rng(CX_SEED))
    d_vdc = np.asarray(estimate_pair_density_from_samples(bundle, xc, projection_iters=30), float)
    d_vdc = d_vdc / d_vdc.mean()
    tll = pv.Bicop.from_data(np.clip(xc, 1e-10, 1 - 1e-10), controls=pv.FitControlsBicop(family_set=[pv.BicopFamily.tll]))
    d_tll = np.asarray(tll.pdf(grid_pts), float).reshape(M, M)
    d_tll = d_tll / d_tll.mean()
    out.update(cx_true=d_true, cx_sample=xc, cx_vdc=d_vdc, cx_tll=d_tll,
               cx_ise=np.array([np.mean((d_tll - d_true) ** 2), np.mean((d_vdc - d_true) ** 2)]))

    # training-loss history (downsampled every 200 steps, plus the per-step minimum)
    if a.loss_csv is not None:
        cols = np.genfromtxt(a.loss_csv, delimiter=",", names=True, usecols=("step", "loss", "row_marg_rms"))
        step, loss, rms = cols["step"], cols["loss"], cols["row_marg_rms"]
        keep = slice(None, None, 200)
        out.update(loss_step=step[keep], loss_value=loss[keep], loss_row_marg_rms=rms[keep],
                   loss_min=np.array([np.nanmin(loss)]))
    else:
        with np.load(a.loss_npz) as prev:
            out.update({k: prev[k] for k in ("loss_step", "loss_value", "loss_row_marg_rms", "loss_min")})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    prov = {"script": "drafts/scripts/fig1_overview_data.py", "vdc_imported_from": vdc.__file__,
            "library_pin": (a.lib / "PIN.txt").read_text().strip() if (a.lib / "PIN.txt").exists() else None,
            "checkpoint": str(a.checkpoint), "checkpoint_sha256": sha256(a.checkpoint),
            **({"loss_csv": str(a.loss_csv)} if a.loss_csv is not None else {"loss_npz": str(a.loss_npz)}),
            "device": str(dev), "pyvinecopulib": pv.__version__,
            "torch": torch.__version__, "numpy": np.__version__, "out_sha256": sha256(a.out),
            "cx_ise_tll_vdc": out["cx_ise"].tolist()}
    a.out.with_suffix(".json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
