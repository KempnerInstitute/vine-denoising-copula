#!/usr/bin/env python3
"""Compute the real data drawn in the camera-ready Fig. 1 and the appendix diagnostics figure.

Everything shown is a genuine output: pyvinecopulib samples, their 64x64 rank histograms, and
densities predicted by the frozen paper checkpoint through the pinned library's public API
(vdc.pretrained.estimate_pair_density_from_samples; projection_iters=0 gives the mass-normalized
raw output, 30 the deployed IPFP projection). The training step of Fig. 1(a) repeats that API's
forward pass on the pictured corrupted histogram with its noise level t (the mirror is checked bitwise
against the API at t=0). Written to one small .npz plus a provenance JSON; the drawing script
(fig1_overview_cr.py) reads only that file.

Contents
  zoo_<k>          four pretraining-zoo samples (n=250) for the pretraining icon
  pre_*            pretraining example, Clayton theta=3, n=1500, seed 123 (the sample of the earlier
                   IPFP panel): clean histogram with its raw and projected outputs (appendix); the
                   uniform-mixed histogram of one training step (alpha=0.3, i.e. noise level
                   t=alpha/0.65) with the projected output of that forward pass (Fig. 1(a)); true
                   density on the cell centres
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
# Training corrupts the input histogram H as (1 - a) H + a (mean-one grids) with a = 0.65 t, t ~ U(0, 1), and
# conditions the network on t (deployed run: training.hist_noise max_strength 0.65, power 1).
HIST_NOISE_MAX = 0.65
PRE_ALPHA = 0.3


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
    from vdc.models.projection import copula_project
    from vdc.utils.probit_transform import copula_logdensity_to_probit_logdensity, probit_logdensity_to_copula_logdensity

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

    def forward(h, n, t, k):
        """The single-pass branch of estimate_pair_density_from_samples (pinned library), applied to a given
        histogram h from n points with noise level t instead of the clean histogram at t = 0."""
        model = bundle.model
        hist_t = torch.from_numpy(np.asarray(h, np.float64)).float().unsqueeze(0).unsqueeze(0).to(dev)
        probit = bool(getattr(model, "vdc_transform_to_probit_space", False))
        if probit:
            hist_t = torch.exp(copula_logdensity_to_probit_logdensity(torch.log(hist_t.clamp_min(1e-12)), M)).clamp(min=1e-12, max=1e6)
        in_ch = None
        if hasattr(model, "in_conv"):
            in_ch = int(model.in_conv.in_channels)
        elif hasattr(model, "input_conv"):
            in_ch = int(model.input_conv[0].in_channels)
        elif hasattr(model, "conv_in"):
            in_ch = int(model.conv_in.in_channels)
        use_log_n = bool(getattr(model, "vdc_use_log_n", False))
        use_coords = bool(getattr(model, "vdc_use_coordinates", False))
        if in_ch is not None and not hasattr(model, "vdc_use_log_n") and in_ch in (2, 4):
            use_log_n = True
        if in_ch is not None and not hasattr(model, "vdc_use_coordinates") and in_ch in (3, 4):
            use_coords = True
        x = hist_t
        if use_log_n:
            x = torch.cat([x, torch.full((1, 1, M, M), float(np.log(max(1, n))), device=dev, dtype=x.dtype)], dim=1)
        if use_coords:
            g = torch.linspace(0.5 / M, 1.0 - 0.5 / M, M, device=dev, dtype=x.dtype)
            uu, vv = torch.meshgrid(g, g, indexing="ij")
            if bool(getattr(model, "vdc_use_probit_coords", False)):
                eps = max(float(getattr(model, "vdc_probit_coord_eps", 1e-4)), 1.0 / (M * M))
                uu = torch.erfinv(2 * uu.clamp(eps, 1 - eps) - 1) * (2.0 ** 0.5)
                vv = torch.erfinv(2 * vv.clamp(eps, 1 - eps) - 1) * (2.0 ** 0.5)
            x = torch.cat([x, torch.stack([uu, vv], dim=0).unsqueeze(0)], dim=1)
        assert in_ch is None or x.shape[1] == in_ch
        with torch.no_grad():
            out = model(x, torch.full((1,), float(t), device=dev))
        if isinstance(out, dict):
            if "density" in out:
                d = out["density"]
            elif "log_density" in out:
                d = torch.exp(out["log_density"].clamp(min=-20, max=20))
            else:
                d = torch.exp((torch.log(hist_t.clamp_min(1e-12)) + out["residual"]).clamp(min=-20, max=20))
        else:
            d = out
        if probit:
            d = torch.exp(probit_logdensity_to_copula_logdensity(torch.log(d.clamp_min(1e-12)), M).clamp(min=-20, max=20)).clamp(min=1e-12, max=1e6)
        d = torch.nan_to_num(d, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=1e-12, max=1e6)
        du = 1.0 / M
        d = d / ((d * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
        d = copula_project(d, iters=int(k))
        d = d / ((d * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
        d = np.asarray(d[0, 0].detach().cpu().numpy(), float)
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
    out["pre_hist_mixed"] = (1 - PRE_ALPHA) * h + PRE_ALPHA   # bitwise equal to the earlier 0.7 h + 0.3
    out["pre_raw"] = vdc_density(x, 0)
    out["pre_proj"] = vdc_density(x, 30)
    # the mirrored forward pass reproduces the API on the clean histogram at t = 0 ...
    h_api = scatter_to_hist(np.clip(x, 1e-10, 1 - 1e-10), m=M, reflect=True)
    assert np.array_equal(forward(h_api, len(x), 0.0, 30), out["pre_proj"])
    # ... and gives the output of the pictured training step: the mixed histogram at its noise level
    out["pre_mixed_t"] = np.array([PRE_ALPHA / HIST_NOISE_MAX])
    out["pre_mixed_proj"] = forward(out["pre_hist_mixed"], len(x), PRE_ALPHA / HIST_NOISE_MAX, 30)
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
