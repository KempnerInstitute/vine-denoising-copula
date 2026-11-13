# ruff: noqa: E702, N803, N806, N812  (verbatim port of the research implementation)
"""Pretraining from scratch ("pilot") for the linear-grid models.

This is the initialization of ``vdc-denoiser-m64-v2`` and
``vdc-denoiser-m128-v2``: a 60,000-step run of the original denoiser
architecture on exact synthetic copula targets, trained from random weights.
It is not descended from ``vdc-denoiser-m64-v1``.

Held fixed relative to the original estimator: the network
``CopulaDenoiser(m=64, in=4, base=128, depth=4, blocks=2, temb=256,
dropout=.1, log output)``; inputs ``[histogram, log n, probit u, probit v]``;
uniform-mixing corruption with strength ``.65 t``, ``t ~ U(0, 1)``; the loss
(cross-entropy on IPFP-projected cell masses, weight 1.0, plus tail
log-ratio, 0.2 with tau = .15, and marginal KL, 0.02); and a zoo of
Gaussian, Student-t, Clayton, Gumbel, Frank, Joe, BB1, BB7, independence and
X/ring/double-banana copulas, with mixtures of 2-4 components (probability
0.2) and ``n ~ LogUniform(200, 8000)``. BB6, BB8 and Tawn are not used.

Changed: targets are exact 64 x 64 cell masses from pyvinecopulib CDFs; inputs
are histograms of rank pseudo-observations of points drawn from those masses
on the GPU, so input and target describe the same copula; exact dihedral (D4)
augmentation; exact mixtures; no conditional-copula examples. Recipe: bf16
autocast, batch 128, AdamW with 2,000 warmup steps and cosine decay, EMA 0.9995.

Use :func:`run_pretraining` (``scripts/train_linear_grid_estimator.py`` with
``configs/train/linear_grid_pilot.yaml``). A CUDA device is required for the
recorded numerics; CPU execution is supported for tests only.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.ipfp_log import ipfp_project_log, marginal_deviation

M = 64

FAMILY_WEIGHTS = {  # original zoo weights without the two conditional families
    "bb1": .08, "bb7": .08, "clayton": .10, "complex_double_banana": .02, "complex_ring": .03,
    "complex_x": .03, "frank": .05, "gaussian": .12, "gumbel": .12, "independence": .03,
    "joe": .12, "student": .08,
}
RANGES = {"bb1_delta": (1.0, 5.0), "bb1_theta": (0.1, 5.0), "bb7_delta": (0.1, 5.0), "bb7_theta": (1.0, 5.0),
          "clayton_theta": (0.1, 14.0), "frank_theta": (-15.0, 15.0), "gaussian_rho": (-0.95, 0.95),
          "gumbel_theta": (1.1, 12.0), "joe_theta": (1.1, 12.0), "student_df": (2.1, 30.0),
          "student_rho": (-0.9, 0.9)}


def exact_masses(bicop):
    """Exact 64 x 64 cell probabilities of a pyvinecopulib copula (floored, renormalized)."""
    e = np.linspace(0.0, 1.0, M + 1)
    E1, E2 = np.meshgrid(e, e, indexing="ij")
    C = bicop.cdf(np.clip(np.c_[E1.ravel(), E2.ravel()], 1e-12, 1 - 1e-12)).reshape(M + 1, M + 1)
    C[0, :] = 0.0; C[:, 0] = 0.0; C[-1, :] = e; C[:, -1] = e
    P = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    P = np.clip(P, 1e-14, None)
    return P / P.sum()


def build_bank(n_items, seed, progress_every=20000):
    """Build ``n_items`` target cell-mass grids (float32) and their family labels."""
    import pyvinecopulib as pv

    from ..data.complex_copulas import complex_copula_density_grid
    rng = np.random.default_rng(seed)
    fams = list(FAMILY_WEIGHTS)
    p = np.array([FAMILY_WEIGHTS[f] for f in fams]); p /= p.sum()
    bank = np.empty((n_items, M, M), dtype=np.float32)
    labels = []
    U = lambda key: float(rng.uniform(*RANGES[key]))  # noqa: E731
    for i in range(n_items):
        fam = str(rng.choice(fams, p=p))
        if fam.startswith("complex_"):
            kind = fam.split("_", 1)[1]
            if kind == "x":
                par = {"sigma": float(rng.uniform(.02, .06)), "w2": float(rng.uniform(.5, 2.))}
            elif kind == "ring":
                par = {"r0": float(rng.uniform(.20, .36)), "sigma": float(rng.uniform(.02, .06))}
            else:
                par = {"amp": float(rng.uniform(.10, .22)), "offset": float(rng.uniform(.08, .25)),
                       "sigma": float(rng.uniform(.02, .06))}
            d = complex_copula_density_grid(kind, par, m=M, device=torch.device("cpu"), projection_iters=200)
            P = np.clip(np.asarray(d, dtype=np.float64), 1e-14, None); P /= P.sum()
        elif fam == "independence":
            P = np.full((M, M), 1.0 / (M * M))
        else:
            if fam in ("gaussian",):
                b = pv.Bicop(family=pv.BicopFamily.gaussian, parameters=np.array([[U("gaussian_rho")]]))
            elif fam == "student":
                b = pv.Bicop(family=pv.BicopFamily.student, parameters=np.array([[U("student_rho")], [U("student_df")]]))
            elif fam in ("clayton", "gumbel", "frank", "joe"):
                th = U(f"{fam}_theta")
                if fam == "frank" and abs(th) < 1e-3:
                    th = 1e-3
                b = pv.Bicop(family=getattr(pv.BicopFamily, fam), parameters=np.array([[th]]))
            else:  # bb1 / bb7
                b = pv.Bicop(family=getattr(pv.BicopFamily, fam),
                             parameters=np.array([[U(f"{fam}_theta")], [U(f"{fam}_delta")]]))
            P = exact_masses(b)
        bank[i] = P.astype(np.float32)
        labels.append(fam)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  bank {i + 1}/{n_items}", flush=True)
    return bank, labels


def d4(x, code):
    """Apply one of 8 dihedral transforms to (..., M, M) grids; code in [0,8)."""
    if code & 4:
        x = x.transpose(-1, -2)
    if code & 2:
        x = x.flip(-2)
    if code & 1:
        x = x.flip(-1)
    return x


class Batcher:
    """GPU batch builder: mixed D4-augmented targets and rank-histogram inputs."""

    def __init__(self, bank, device, n_min=200, n_max=8000, mixture_prob=0.2, seed=0):
        self.bank = torch.from_numpy(bank).to(device)
        self.N = self.bank.shape[0]
        self.device = device
        self.n_min, self.n_max, self.mix_p = n_min, n_max, mixture_prob
        self.g = torch.Generator(device=device); self.g.manual_seed(seed)
        u = (torch.arange(M, device=device, dtype=torch.float32) + .5) / M
        eps = max(1e-4, 1.0 / (M * M))
        z = torch.erfinv(2 * u.clamp(eps, 1 - eps) - 1) * math.sqrt(2.0)
        uu, vv = torch.meshgrid(z, z, indexing="ij")
        self.coords = torch.stack([uu, vv], 0).unsqueeze(0)  # probit coordinate channels

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.device)

    def target_masses(self, B):
        idx = torch.randint(0, self.N, (B, 4), generator=self.g, device=self.device)
        comps = self.bank[idx]  # (B,4,M,M)
        codes = torch.randint(0, 8, (B, 4), generator=self.g, device=self.device)
        out = torch.empty_like(comps)
        for c in range(8):  # exact symmetry augmentation, independently per component
            sel = codes == c
            if sel.any():
                out[sel] = d4(comps[sel], c)
        k = torch.randint(2, 5, (B,), generator=self.g, device=self.device)
        w = -torch.log(self._rand(B, 4).clamp_min(1e-12))  # Dirichlet(1) via exponentials
        w = w * (torch.arange(4, device=self.device)[None] < k[:, None])
        is_mix = self._rand(B) < self.mix_p
        w = torch.where(is_mix[:, None], w, F.one_hot(torch.zeros(B, dtype=torch.long, device=self.device), 4).float())
        w = w / w.sum(1, keepdim=True)
        P = (w[:, :, None, None] * out).sum(1)
        return P / P.sum((-2, -1), keepdim=True)

    def rank_histograms(self, P, n):
        """Histogram of rank pseudo-observations of n_b points drawn from cell masses P_b."""
        B = P.shape[0]
        nmax = int(n.max())
        cells = torch.multinomial(P.view(B, -1), nmax, replacement=True, generator=self.g)  # (B,nmax)
        iu = torch.div(cells, M, rounding_mode="floor").float(); iv = (cells % M).float()
        x = torch.stack([(iu + self._rand(B, nmax)) / M, (iv + self._rand(B, nmax)) / M], -1)  # within-cell jitter
        valid = torch.arange(nmax, device=self.device)[None] < n[:, None]
        x = torch.where(valid[..., None], x, torch.full_like(x, 2.0))  # invalid points sort last
        ranks = x.argsort(1).argsort(1).float() + 1.0  # 1..nmax per coordinate
        po = ranks / (n[:, None, None].float() + 1.0)  # pseudo-observations of the n_b valid points
        bins = (po * M).long().clamp(0, M - 1)
        flat = bins[..., 0] * M + bins[..., 1]
        h = torch.zeros(B, M * M, device=self.device)
        h.scatter_add_(1, flat, valid.float())
        h = h.view(B, M, M) / n[:, None, None].float() * (M * M)  # density (integral 1)
        return h

    def batch(self, B):
        P = self.target_masses(B)
        logn = math.log(self.n_min) + self._rand(B) * (math.log(self.n_max) - math.log(self.n_min))
        n = logn.exp().round().long().clamp(self.n_min, self.n_max)
        hist = self.rank_histograms(P, n)
        return P, hist, n


def tail_mask(device, tau=0.15):
    u = torch.linspace(.5 / M, 1 - .5 / M, M, device=device)
    U, V = torch.meshgrid(u, u, indexing="ij")
    return ((U < tau) | (V < tau) | (U > 1 - tau) | (V > 1 - tau)).float()[None, None]


def density_losses(pred, target, tmask, w_ce=1.0, w_tail=0.2, w_marg=0.02):
    area = 1.0 / (M * M)
    T = (target * area).clamp_min(1e-12)
    D = (pred * area).clamp_min(1e-12)
    ce = -(T * D.log()).sum((-2, -1)).view(-1)
    tail = (tmask * (D.log() - T.log()) ** 2).flatten(1).mean(-1)
    row = (pred / M).sum(-1).clamp_min(1e-12); col = (pred / M).sum(-2).clamp_min(1e-12)
    marg = (row * row.log()).sum(-1).view(-1) + (col * col.log()).sum(-1).view(-1)
    total = w_ce * ce.mean() + w_tail * tail.mean() + w_marg * marg.mean()
    return total, {"ce": float(ce.mean()), "tail": float(tail.mean()), "marg": float(marg.mean())}


def project_like_training(dens):
    quick = ipfp_project_log(dens, iters=3)
    if float(marginal_deviation(quick)) > 0.01:
        return ipfp_project_log(dens, iters=30)
    return quick


def normalize(d):
    return d / (d.sum((-2, -1), keepdim=True) / (M * M)).clamp_min(1e-12)


PRETRAINING_DEFAULTS: Dict[str, Any] = {
    "output_dir": None, "bank_path": None, "bank_size": 200000, "bank_seed": None,
    "bank_sha256": None, "bank_only": False, "steps": 60000, "batch_size": 128, "lr": 2e-4,
    "warmup": 2000, "ema": 0.9995, "save_every": 10000, "base_channels": 128, "seed": 42,
    "device": "cuda", "stop_after": None,
}


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_pretraining(settings: Dict[str, Any]) -> Path:
    """Build or load the target bank and pretrain from random weights.

    Settings (defaults in ``PRETRAINING_DEFAULTS``): ``output_dir``,
    ``bank_path`` (built with ``bank_seed``, default ``seed``, if missing),
    ``bank_sha256`` (optional check), ``steps``, ``batch_size``, ``lr``,
    ``warmup``, ``ema``, ``save_every``, ``base_channels``, ``seed``,
    ``device``, ``bank_only`` and ``stop_after`` (end early without changing
    the learning-rate schedule). Writes ``config.json``, ``train_log.jsonl``
    (every 100 steps) and ``pilot_{raw,ema}_step{N}.pt``.
    """
    from ..models.copula_denoiser import CopulaDenoiser

    unknown = set(settings) - set(PRETRAINING_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown pretraining settings: {sorted(unknown)}")
    a = {**PRETRAINING_DEFAULTS, **settings}
    if a["output_dir"] is None or a["bank_path"] is None:
        raise ValueError("output_dir and bank_path are required")
    out, bank_path = Path(a["output_dir"]), Path(a["bank_path"])
    steps, batch_size, warmup = int(a["steps"]), int(a["batch_size"]), int(a["warmup"])
    stop_after = None if a["stop_after"] is None else int(a["stop_after"])
    if stop_after is not None and not 0 < stop_after <= steps:
        raise ValueError("stop_after must be within the schedule")
    seed = int(a["seed"])
    torch.manual_seed(seed); np.random.seed(seed)

    if bank_path.exists():
        bank = np.load(bank_path)
        print("loaded bank", bank.shape, flush=True)
    else:
        start = time.time()
        bank_seed = seed if a["bank_seed"] is None else int(a["bank_seed"])
        bank, labels = build_bank(int(a["bank_size"]), bank_seed)
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(bank_path, bank)
        Path(str(bank_path) + ".labels.json").write_text(json.dumps(labels))
        print(f"built bank {bank.shape} in {time.time() - start:.0f}s", flush=True)
    if a["bank_sha256"] and _sha256(bank_path) != a["bank_sha256"]:
        raise ValueError(f"Bank hash mismatch: {bank_path}")
    if a["bank_only"]:
        return bank_path

    dev = torch.device(a["device"])
    base = int(a["base_channels"])
    cfg = {"model": {"type": "denoiser", "grid_size": M, "base_channels": base, "depth": 4,
                     "blocks_per_level": 2, "dropout": 0.1, "time_emb_dim": 256, "output_mode": "log",
                     "use_coordinates": True, "use_probit_coords": True, "use_log_n": True,
                     "time_conditioning": True, "transform_to_probit_space": False},
           "data": {"m": M, "families": FAMILY_WEIGHTS, "param_ranges": RANGES, "mixture_prob": 0.2,
                    "n_samples_per_copula": {"mode": "log_uniform", "min": 200, "max": 8000},
                    "targets": "exact pyvinecopulib cell masses", "inputs": "rank pseudo-observation histograms",
                    "augmentation": "D4 exact", "bank_size": int(bank.shape[0])},
           "training": {"projection_iters": 30, "batch_size": batch_size, "lr": float(a["lr"]), "warmup": warmup,
                        "schedule": "cosine", "ema": float(a["ema"]), "steps": steps, "amp": "bf16",
                        "loss_weights": {"ce": 1.0, "tail": 0.2, "marg_kl": 0.02}, "tail_tau": 0.15,
                        "hist_noise": {"max_strength": 0.65, "power": 1.0}, "seed": seed}}
    model = CopulaDenoiser(m=M, input_channels=4, base_channels=base, depth=4, blocks_per_level=2,
                           time_emb_dim=256, dropout=0.1, output_mode="log").to(dev)
    ema = CopulaDenoiser(m=M, input_channels=4, base_channels=base, depth=4, blocks_per_level=2,
                         time_emb_dim=256, dropout=0.1, output_mode="log").to(dev)
    ema.load_state_dict(model.state_dict()); ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    ema_decay = float(a["ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(a["lr"]), weight_decay=0.01, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) *
                                              (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))) + 1e-3)
    batcher = Batcher(bank, dev, seed=seed)
    tmask = tail_mask(dev)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log = (out / "train_log.jsonl").open("a")
    t0 = time.time()
    last = stop_after or steps
    for step in range(1, last + 1):
        model.train()
        with torch.no_grad():
            P, hist, n = batcher.batch(batch_size)
            target = P * (M * M)
            t = torch.rand(batch_size, device=dev)
            s = (0.65 * t).view(-1, 1, 1, 1)
            x_hist = normalize((1 - s) * hist[:, None] + s * torch.ones_like(hist[:, None]))
            ln = torch.log(n.float()).view(-1, 1, 1, 1).expand(-1, 1, M, M)
            x = torch.cat([x_hist, ln, batcher.coords.expand(batch_size, -1, -1, -1)], 1)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            out_t = model(x, t)
        logd = out_t["log_density"].float().clamp(-15, 15)
        pred = normalize(torch.exp(logd).clamp(1e-12, 1e6))
        proj = normalize(project_like_training(pred))
        loss, parts = density_losses(proj, target[:, None], tmask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - ema_decay)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        if step % 100 == 0:
            rec = {"step": step, "loss": float(loss), "grad_norm": float(gn), "lr": sched.get_last_lr()[0],
                   "s_per_step": (time.time() - t0) / step, **parts}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 1000 == 0:
                print(json.dumps(rec), flush=True)
        if step % int(a["save_every"]) == 0 or step == steps or step == stop_after:
            for tag, net in (("raw", model), ("ema", ema)):
                torch.save({"step": step, "model_state_dict": net.state_dict(), "config": cfg},
                           out / f"pilot_{tag}_step{step}.pt")
    log.close()
    print("done", time.time() - t0, flush=True)
    return out / f"pilot_ema_step{last}.pt"
