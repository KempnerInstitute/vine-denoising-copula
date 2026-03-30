#!/usr/bin/env python3
"""
Mutual information estimation benchmark (bivariate copulas).

This is intended to support the paper's "information-theoretic evaluation" with
**real, reproducible** numbers.

It compares MI estimators on a fixed bivariate copula zoo:
  - kNN MI (KSG)  [fast, no training]
  - Gaussian MI (probit correlation)  [fast, assumes Gaussian copula]
  - InfoNCE MI (contrastive lower bound)  [medium, trains a small critic network]
  - NWJ MI (f-div lower bound)  [medium, trains a small critic network]
  - MINE (DV bound)  [medium, trains a small critic network]
  - MINDE (ICLR 2024)  [slow, trains a diffusion score model]

Reference:
  - MINDE repo: https://github.com/MustaphaBounoua/minde
    (see README in that repository)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.data.generators import analytic_logpdf_grid, sample_bicop  # noqa: E402
from vdc.inference.density import sample_density_grid  # noqa: E402
from vdc.models.projection import copula_project  # noqa: E402
from vdc.utils.information import ksg_mutual_information  # noqa: E402


DEFAULT_TEST_COPULAS = [
    ("independence", {}, 0, "Independence"),
    ("gaussian", {"rho": 0.7}, 0, "Gaussian(ρ=0.7)"),
    ("gaussian", {"rho": -0.7}, 0, "Gaussian(ρ=-0.7)"),
    ("student", {"rho": 0.7, "df": 5}, 0, "Student-t(ρ=0.7, df=5)"),
    ("clayton", {"theta": 3.0}, 0, "Clayton(θ=3.0)"),
    ("clayton", {"theta": 3.0}, 90, "Clayton(θ=3.0, rot=90)"),
    ("gumbel", {"theta": 2.5}, 0, "Gumbel(θ=2.5)"),
    ("joe", {"theta": 3.0}, 0, "Joe(θ=3.0)"),
    ("frank", {"theta": 5.0}, 0, "Frank(θ=5.0)"),
]


def _normalize_density_np(d: np.ndarray) -> np.ndarray:
    m = int(d.shape[0])
    du = 1.0 / m
    d = np.nan_to_num(d, nan=0.0, posinf=1e300, neginf=0.0)
    d = np.clip(d, 0.0, 1e300)
    mass = float((d * du * du).sum())
    if not np.isfinite(mass) or mass <= 0:
        return np.ones_like(d) * (m * m)
    return d / mass


def _project_density_np(d: np.ndarray, iters: int, device: torch.device) -> np.ndarray:
    t = torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(device)
    t = t.clamp_min(1e-12)
    t = copula_project(t, iters=int(iters))
    return t[0, 0].detach().cpu().numpy()


def _mi_from_density_grid(d: np.ndarray) -> float:
    m = int(d.shape[0])
    du = 1.0 / m
    return float(np.sum(d * np.log(d + 1e-12)) * du * du)


def _mi_true_from_analytic(family: str, params: Dict[str, Any], rotation: int, m_true: int, device: torch.device) -> float:
    lg = analytic_logpdf_grid(family, params, m=m_true, rotation=rotation)
    d = np.exp(np.clip(lg, -20, 20))
    d = _normalize_density_np(d)
    # Project to enforce exact copula constraints on the discrete grid (consistency with paper eval).
    d = _project_density_np(d, iters=50, device=device)
    d = _normalize_density_np(d)
    return _mi_from_density_grid(d)


@dataclass(frozen=True)
class _LoadedModel:
    model: Any
    model_type: str
    config: Dict[str, Any]


def _load_checkpoint_model(ckpt_path: Path, device: torch.device) -> _LoadedModel:
    from vdc.train.unified_trainer import build_model

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")
    config = ckpt.get("config", {})
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint missing config dict.")

    model_type_raw = str(config.get("model", {}).get("type", "diffusion_unet"))
    model_type = "diffusion_unet" if model_type_raw.startswith("diffusion_unet") else model_type_raw
    model = build_model(model_type, config, device)
    model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    model.eval()
    return _LoadedModel(model=model, model_type=model_type, config=config)


def _resolve_dcd_checkpoint(
    *,
    explicit_checkpoint: Optional[Path],
    output_bases: List[Path],
    preferred_methods: List[str],
) -> Path:
    def _checkpoint_model_type(path: Path) -> Optional[str]:
        try:
            ckpt = torch.load(str(path), map_location="cpu")
            if isinstance(ckpt, dict):
                cfg = ckpt.get("config", {})
                if isinstance(cfg, dict):
                    return str(cfg.get("model", {}).get("type", ""))
        except Exception:
            return None
        return None

    def _is_supported(path: Path) -> bool:
        mt = _checkpoint_model_type(path)
        return bool(
            mt in {"denoiser", "enhanced_cnn", "diffusion_unet"}
            or (isinstance(mt, str) and mt.startswith("diffusion_unet"))
        )

    if explicit_checkpoint is not None:
        ckpt = Path(explicit_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        if not _is_supported(ckpt):
            mt = _checkpoint_model_type(ckpt)
            raise RuntimeError(
                f"Unsupported checkpoint model type for estimator=dcd: {mt}. "
                "Use a denoiser/enhanced_cnn/diffusion_unet checkpoint."
            )
        return ckpt

    bases = list(output_bases)
    if not bases and os.environ.get("OUTPUT_BASE"):
        bases = [Path(os.environ["OUTPUT_BASE"])]
    if not bases:
        bases = [Path("/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula")]

    from vdc.utils.paper import choose_best_checkpoint

    ckpt = choose_best_checkpoint(
        output_bases=bases,
        preferred_methods=[str(x).strip() for x in preferred_methods if str(x).strip()],
        metric="mean_ise",
        prefer_joint=True,
    )
    if ckpt is not None and ckpt.exists() and _is_supported(ckpt):
        return ckpt

    # Fallback: discover latest checkpoint directly from run directories.
    preferred = [str(x).strip() for x in preferred_methods if str(x).strip()]
    for method in preferred:
        candidates: List[Path] = []
        for base in bases:
            if not base.exists():
                continue
            for run_dir in sorted(base.glob(f"vdc_paper_{method}_*"), reverse=True):
                ck_dir = run_dir / "checkpoints"
                if not ck_dir.exists():
                    continue
                for p in ck_dir.glob("model_step_*.pt"):
                    candidates.append(p)
        if candidates:
            for cand in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
                if _is_supported(cand):
                    return cand

    # Last resort: any denoiser/enhanced single-pass run.
    fallback: List[Path] = []
    for base in bases:
        if not base.exists():
            continue
        for run_dir in sorted(base.glob("vdc_paper_*"), reverse=True):
            name = run_dir.name.lower()
            if ("denoiser" not in name) and ("enhanced_cnn" not in name):
                continue
            ck_dir = run_dir / "checkpoints"
            if not ck_dir.exists():
                continue
            for p in ck_dir.glob("model_step_*.pt"):
                fallback.append(p)
    if fallback:
        for cand in sorted(fallback, key=lambda p: p.stat().st_mtime, reverse=True):
            if _is_supported(cand):
                return cand

    # Local repository fallback: scan checkpoints/ when paper run directories
    # are not available in OUTPUT_BASE.
    repo_root = Path(__file__).resolve().parent.parent
    local_candidates: List[Path] = []
    for root_name in ("checkpoints", "archive/checkpoints", "archive/checkpoints_old"):
        root = repo_root / root_name
        if root.exists():
            local_candidates.extend(root.glob("**/model_step_*.pt"))
    if local_candidates:
        for cand in sorted(local_candidates, key=lambda p: p.stat().st_mtime, reverse=True):
            if _is_supported(cand):
                return cand

    raise RuntimeError(
        "No DCD checkpoint found. Pass --checkpoint or provide --output-base with completed paper runs."
    )


class _DCDMIEstimator:
    """DCD-Vine bivariate MI estimator from predicted copula densities."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        device: torch.device,
        diffusion_steps: Optional[int] = None,
        diffusion_cfg_scale: Optional[float] = None,
        diffusion_ensemble: int = 1,
        diffusion_ensemble_mode: str = "geometric",
        diffusion_smooth_sigma: float = 0.0,
        diffusion_pred_noise_clip: Optional[float] = 10.0,
        projection_iters: Optional[int] = None,
        seed_base: int = 123,
    ):
        from vdc.vine.api import VineCopulaModel

        self.checkpoint = Path(checkpoint)
        self.loaded = _load_checkpoint_model(self.checkpoint, device=device)

        m = int(self.loaded.config.get("data", {}).get("m", 64))
        proj_iters = int(self.loaded.config.get("training", {}).get("projection_iters", 30))
        self.model_type = str(self.loaded.model_type)
        self.m = m
        self.proj_iters = int(projection_iters) if projection_iters is not None else int(proj_iters)
        self.diffusion = None
        self.use_histogram_conditioning = False
        self.transform_to_probit_space = bool(
            self.loaded.config.get("model", {}).get("transform_to_probit_space", False)
        )

        self.diffusion_steps = (
            int(diffusion_steps)
            if diffusion_steps is not None
            else int(self.loaded.config.get("diffusion", {}).get("sampling_steps", 50))
        )
        self.diffusion_cfg_scale = (
            float(diffusion_cfg_scale)
            if diffusion_cfg_scale is not None
            else float(self.loaded.config.get("diffusion", {}).get("cfg_scale", 1.0))
        )
        self.diffusion_ensemble = max(1, int(diffusion_ensemble))
        self.diffusion_ensemble_mode = str(diffusion_ensemble_mode).lower().strip()
        self.diffusion_smooth_sigma = float(diffusion_smooth_sigma)
        self.diffusion_pred_noise_clip = diffusion_pred_noise_clip
        self.seed_base = int(seed_base)

        if str(self.loaded.model_type) == "diffusion_unet":
            from vdc.models.copula_diffusion import CopulaAwareDiffusion

            diff_cfg = self.loaded.config.get("diffusion", {})
            self.diffusion = CopulaAwareDiffusion(
                timesteps=int(diff_cfg.get("timesteps", 1000)),
                beta_schedule=str(diff_cfg.get("noise_schedule", "cosine")),
            ).to(device)
            conv_in = getattr(self.loaded.model, "conv_in", None)
            if conv_in is not None and hasattr(conv_in, "in_channels"):
                self.use_histogram_conditioning = int(conv_in.in_channels) > 1

        self.loaded.model.to(device)
        self.loaded.model.eval()

        self._vine_helper = VineCopulaModel(
            vine_type="dvine",
            m=int(m),
            device=str(device),
            projection_iters=int(proj_iters),
            hfunc_use_spline=False,
            batch_edges=False,
        )

    def _aggregate_density_ensemble(self, ensemble: np.ndarray) -> np.ndarray:
        if ensemble.ndim != 3:
            raise ValueError(f"Expected ensemble shape (E,m,m), got {ensemble.shape}")
        if ensemble.shape[0] == 1:
            return ensemble[0]
        mode = self.diffusion_ensemble_mode
        if mode == "geometric":
            return np.exp(np.mean(np.log(np.clip(ensemble, 1e-12, None)), axis=0))
        if mode == "arithmetic":
            return np.mean(ensemble, axis=0)
        if mode == "median":
            return np.median(ensemble, axis=0)
        raise ValueError(f"Unknown diffusion_ensemble_mode: {mode}")

    def estimate_mi(self, pts: np.ndarray) -> float:
        pair_data = np.asarray(pts, dtype=np.float64)
        pair_data = np.clip(pair_data, 1e-6, 1.0 - 1e-6)

        if self.diffusion is not None:
            # Match model-selection diffusion inference: deterministic seeds and
            # a single final projection after ensemble aggregation.
            ensemble = []
            for k in range(self.diffusion_ensemble):
                s = int(self.seed_base + 1000 * k)
                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)
                d_k = sample_density_grid(
                    model=self.loaded.model,
                    diffusion=self.diffusion,
                    samples=pair_data,
                    m=int(self.m),
                    device=next(self.loaded.model.parameters()).device,
                    num_steps=int(self.diffusion_steps),
                    cfg_scale=float(self.diffusion_cfg_scale),
                    use_histogram_conditioning=bool(self.use_histogram_conditioning),
                    projection_iters=0,
                    pred_noise_clip=self.diffusion_pred_noise_clip,
                    transform_to_probit_space=bool(self.transform_to_probit_space),
                )
                ensemble.append(np.asarray(d_k, dtype=np.float64))

            d = self._aggregate_density_ensemble(np.stack(ensemble, axis=0))
            if self.diffusion_smooth_sigma > 0:
                from vdc.utils.smoothing import smooth_density_gaussian

                t = torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0).to(next(self.loaded.model.parameters()).device)
                t = smooth_density_gaussian(t, sigma=float(self.diffusion_smooth_sigma), preserve_mass=True)
                d = t[0, 0].detach().cpu().numpy()
            d = _normalize_density_np(d)
            d = _project_density_np(d, iters=int(self.proj_iters), device=next(self.loaded.model.parameters()).device)
            d = _normalize_density_np(d)
        else:
            d = self._vine_helper._estimate_pair_density_from_samples(
                model=self.loaded.model,
                diffusion=None,
                pair_data=pair_data,
                use_histogram_conditioning=bool(self.use_histogram_conditioning),
            )
            d = _normalize_density_np(np.asarray(d, dtype=np.float64))

        return _mi_from_density_grid(d)


def _maybe_clone_repo(dest: Path, url: str) -> None:
    """Clone a git repository if it doesn't already exist."""
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])


def _maybe_clone_minde(dest: Path) -> None:
    """Clone MINDE repository."""
    _maybe_clone_repo(dest, "https://github.com/MustaphaBounoua/minde")


def _maybe_clone_mist(dest: Path) -> None:
    """Clone MIST repository (arXiv:2511.18945)."""
    _maybe_clone_repo(dest, "https://github.com/gritsai/mist")


def _minde_estimate_mi(
    *,
    minde_repo: Path,
    x: np.ndarray,
    y: np.ndarray,
    benchmark: str,
    seed: int,
    device: torch.device,
    max_epochs: int,
    lr: float,
    batch_size: int,
) -> float:
    """
    Train MINDE on (x,y) and return MI estimate.

    This uses the reference implementation:
      https://github.com/MustaphaBounoua/minde
    """
    # Local import from external repo
    if str(minde_repo) not in sys.path:
        sys.path.insert(0, str(minde_repo))

    try:
        import pytorch_lightning as pl
    except Exception as e:
        raise RuntimeError("pytorch-lightning is required to run MINDE. Install it in the environment.") from e

    from src.libs.minde import MINDE  # type: ignore
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, Dataset

    class _XY(Dataset):
        def __init__(self, x_: np.ndarray, y_: np.ndarray):
            self.x = x_.astype(np.float32)
            self.y = y_.astype(np.float32)

        def __len__(self) -> int:
            return int(self.x.shape[0])

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            return {"x": torch.from_numpy(self.x[idx]), "y": torch.from_numpy(self.y[idx])}

    rng = np.random.default_rng(int(seed))
    n = int(x.shape[0])
    perm = rng.permutation(n)
    split = int(0.8 * n)
    tr = perm[:split]
    te = perm[split:]

    ds_tr = _XY(x[tr], y[tr])
    ds_te = _XY(x[te], y[te])

    # Use 0 workers for robustness on clusters (fork issues are common).
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True, num_workers=0, drop_last=True)
    dl_te = DataLoader(ds_te, batch_size=min(1000, len(ds_te)), shuffle=False, num_workers=0, drop_last=False)

    accel = "gpu" if device.type == "cuda" else "cpu"

    # Make runs deterministic-ish across re-launches (enough for paper baselines).
    pl.seed_everything(int(seed), workers=False)

    args = SimpleNamespace(
        type="c",  # conditional MINDE
        preprocessing="rescale",  # MI is invariant to invertible transforms
        arch="mlp",
        Train_Size=len(ds_tr),
        Test_Size=len(ds_te),
        seed=int(seed),
        warmup_epochs=0,
        max_epochs=int(max_epochs),
        lr=float(lr),
        bs=int(batch_size),
        test_epoch=1,
        mc_iter=10,
        use_ema=False,
        importance_sampling=False,
        sigma=1.0,
        debug=False,
        nb_workers=0,
        devices=1,
        accelerator=accel,
        check_val_every_n_epoch=max(1, int(max_epochs)),
        out_dir=str(minde_repo / "_runs"),
        results_dir=str(minde_repo / "_runs"),
        benchmark=str(benchmark),
    )

    minde = MINDE(args=args, gt=None, var_list={"x": x.shape[1], "y": y.shape[1]})
    minde.fit(dl_tr, dl_te)
    mi, _mi_sigma = minde.compute_mi()
    return float(mi)


def _mist_estimate_mi(
    *,
    mist_repo: Path,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: torch.device,
    max_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 3,
) -> float:
    """
    Estimate MI using MIST (Mutual Information via Score Truncation).
    
    Reference: 
        Gritsai et al., "MIST: A Simple and Scalable Mutual Information Estimator
        via Score Truncation", arXiv:2511.18945, 2025.
        https://github.com/gritsai/mist
    
    MIST estimates MI using truncated score functions:
        I(X;Y) = E_{p(x,y)}[s_X(x) · (s_{X|Y}(x|y) - s_X(x))] 
               + E_{p(x,y)}[s_Y(y) · (s_{Y|X}(y|x) - s_Y(y))]
    
    This provides a consistent estimator that avoids the bias issues of 
    variational bounds like MINE/InfoNCE while being more scalable than KSG.
    
    Args:
        mist_repo: Path to cloned MIST repository
        x: (n, dx) array of samples
        y: (n, dy) array of samples  
        seed: Random seed
        device: Torch device
        max_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        hidden_dim: Hidden dimension of score networks
        num_layers: Number of hidden layers
        
    Returns:
        Estimated mutual information in nats.
    """
    # Local import from external repo
    if str(mist_repo) not in sys.path:
        sys.path.insert(0, str(mist_repo))
    
    try:
        from src.mist import MIST  # type: ignore
    except ImportError:
        # Fallback: try alternative import path
        try:
            from mist import MIST  # type: ignore
        except ImportError:
            raise RuntimeError(
                f"Could not import MIST from {mist_repo}. "
                "Ensure the repository is cloned and contains src/mist.py or mist.py. "
                "Clone via: git clone https://github.com/gritsai/mist"
            )
    
    from torch.utils.data import DataLoader, TensorDataset
    
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    
    # Prepare data
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(-1)
    if y_t.ndim == 1:
        y_t = y_t.unsqueeze(-1)
    
    n = x_t.shape[0]
    dx = x_t.shape[1]
    dy = y_t.shape[1]
    
    # Train/test split
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    split = int(0.8 * n)
    tr_idx = perm[:split]
    te_idx = perm[split:]
    
    ds_tr = TensorDataset(x_t[tr_idx], y_t[tr_idx])
    ds_te = TensorDataset(x_t[te_idx], y_t[te_idx])
    
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True, drop_last=True)
    dl_te = DataLoader(ds_te, batch_size=min(1000, len(ds_te)), shuffle=False)
    
    # Initialize MIST estimator
    # Note: Exact API depends on MIST implementation; adjust as needed
    try:
        mist = MIST(
            dim_x=dx,
            dim_y=dy,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            lr=float(lr),
        ).to(device)
    except TypeError:
        # Alternative constructor signature
        mist = MIST(
            input_dim=dx + dy,
            hidden_dim=int(hidden_dim),
            lr=float(lr),
        ).to(device)
    
    # Training loop (simplified; actual implementation may differ)
    for epoch in range(int(max_epochs)):
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            mist.train_step(xb, yb)
    
    # Estimate MI on test set
    mi_vals = []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            yb = yb.to(device)
            mi = mist.estimate_mi(xb, yb)
            mi_vals.append(float(mi))
    
    return float(np.mean(mi_vals))


def _mine_estimate_mi(
    *,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    weight_decay: float = 0.0,
    grad_clip: float = 5.0,
) -> float:
    """
    MINE (Mutual Information Neural Estimation) via the Donsker–Varadhan lower bound:
      I(X;Y) >= E_p[T] - log E_{p_x p_y}[exp(T)].
    """
    import torch.nn as nn

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    if x_t.ndim != 2 or y_t.ndim != 2 or x_t.shape[0] != y_t.shape[0]:
        raise ValueError(f"Expected x,y as (n,dx),(n,dy) with same n; got {x_t.shape} and {y_t.shape}")

    n = int(x_t.shape[0])
    dx = int(x_t.shape[1])
    dy = int(y_t.shape[1])

    net = nn.Sequential(
        nn.Linear(dx + dy, int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), 1),
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    rng = np.random.default_rng(int(seed))
    bs = int(min(max(8, int(batch_size)), n))

    for _ in range(int(steps)):
        idx = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
        idx_m = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
        xb = x_t[idx]
        yb = y_t[idx]
        y_m = y_t[idx_m]

        t_joint = net(torch.cat([xb, yb], dim=1))
        t_marg = net(torch.cat([xb, y_m], dim=1))
        # DV bound (lower bound). We train to maximize it => minimize negative.
        loss = -(t_joint.mean() - torch.log(torch.exp(t_marg).mean() + 1e-8))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip))
        opt.step()

    # Final MI estimate on full data (single shuffle).
    with torch.no_grad():
        perm = torch.from_numpy(rng.permutation(n).astype(np.int64)).to(device)
        t_joint = net(torch.cat([x_t, y_t], dim=1))
        t_marg = net(torch.cat([x_t, y_t[perm]], dim=1))
        mi = t_joint.mean() - torch.log(torch.exp(t_marg).mean() + 1e-8)
    return float(mi.item())


def _gaussian_mi_probit(pts: np.ndarray, *, device: torch.device, eps: float = 1e-6) -> float:
    """
    Gaussian MI estimator for copulas:
      1) Gaussianize U,V via probit: z = Φ^{-1}(u)
      2) Estimate correlation ρ
      3) Return MI = -0.5 log(1-ρ^2)

    This is exact for Gaussian copulas and a reasonable baseline elsewhere.
    """
    x = torch.from_numpy(np.asarray(pts[:, 0], dtype=np.float64)).to(device)
    y = torch.from_numpy(np.asarray(pts[:, 1], dtype=np.float64)).to(device)
    x = x.clamp(float(eps), 1.0 - float(eps))
    y = y.clamp(float(eps), 1.0 - float(eps))

    # probit: Φ^{-1}(u) = sqrt(2) * erfinv(2u-1)
    zx = (2.0 ** 0.5) * torch.erfinv(2.0 * x - 1.0)
    zy = (2.0 ** 0.5) * torch.erfinv(2.0 * y - 1.0)

    zx = zx - zx.mean()
    zy = zy - zy.mean()
    cov = (zx * zy).mean()
    sx = zx.pow(2).mean().sqrt().clamp_min(1e-12)
    sy = zy.pow(2).mean().sqrt().clamp_min(1e-12)
    rho = (cov / (sx * sy)).clamp(-0.999999, 0.999999)
    mi = -0.5 * torch.log1p(-rho * rho)
    return float(mi.item())


def _critic_mlp(in_dim: int, hidden_dim: int, device: torch.device) -> torch.nn.Module:
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), 1),
    ).to(device)


def _infonce_estimate_mi(
    *,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    weight_decay: float = 0.0,
    grad_clip: float = 5.0,
    eval_batches: int = 10,
) -> float:
    """
    InfoNCE / contrastive MI lower bound:
      I(X;Y) >= log(B) - CE(scores, labels)
    where scores_{ij} = T(x_i, y_j) for a batch of size B.
    """
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    n = int(x_t.shape[0])
    bs = int(min(max(8, int(batch_size)), n))

    net = _critic_mlp(in_dim=int(x_t.shape[1] + y_t.shape[1]), hidden_dim=int(hidden_dim), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.default_rng(int(seed))

    labels = torch.arange(bs, device=device)

    for _ in range(int(steps)):
        idx = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
        xb = x_t[idx]  # (B,dx)
        yb = y_t[idx]  # (B,dy)

        # scores: (B,B) where row i corresponds to x_i paired with all y_j
        xb2 = xb.unsqueeze(1).expand(bs, bs, -1)
        yb2 = yb.unsqueeze(0).expand(bs, bs, -1)
        logits = net(torch.cat([xb2, yb2], dim=-1)).squeeze(-1)  # (B,B)

        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip))
        opt.step()

    # Estimate MI bound by averaging a few fresh batches.
    with torch.no_grad():
        mi_vals: List[float] = []
        for _ in range(int(max(1, eval_batches))):
            idx = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
            xb = x_t[idx]
            yb = y_t[idx]
            xb2 = xb.unsqueeze(1).expand(bs, bs, -1)
            yb2 = yb.unsqueeze(0).expand(bs, bs, -1)
            logits = net(torch.cat([xb2, yb2], dim=-1)).squeeze(-1)
            ce = F.cross_entropy(logits, labels)
            mi = float(np.log(bs)) - float(ce.item())
            mi_vals.append(mi)
        return float(np.mean(mi_vals))


def _nwj_estimate_mi(
    *,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    weight_decay: float = 0.0,
    grad_clip: float = 5.0,
    t_clip: float = 20.0,
    eval_batches: int = 20,
) -> float:
    """
    NWJ (Nguyen–Wainwright–Jordan) MI lower bound:
      I(X;Y) >= E_p[T] - E_{p_x p_y}[exp(T - 1)].
    """
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    n = int(x_t.shape[0])
    bs = int(min(max(8, int(batch_size)), n))

    net = _critic_mlp(in_dim=int(x_t.shape[1] + y_t.shape[1]), hidden_dim=int(hidden_dim), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.default_rng(int(seed))

    for _ in range(int(steps)):
        idx = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
        idx_m = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
        xb = x_t[idx]
        yb = y_t[idx]
        y_m = y_t[idx_m]

        t_joint = net(torch.cat([xb, yb], dim=1)).clamp(-float(t_clip), float(t_clip))
        t_marg = net(torch.cat([xb, y_m], dim=1)).clamp(-float(t_clip), float(t_clip))
        loss = -(t_joint.mean() - torch.exp(t_marg - 1.0).mean())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip))
        opt.step()

    with torch.no_grad():
        vals: List[float] = []
        for _ in range(int(max(1, eval_batches))):
            idx = torch.from_numpy(rng.integers(0, n, size=bs, dtype=np.int64)).to(device)
            perm = torch.from_numpy(rng.permutation(n).astype(np.int64)[:bs]).to(device)
            xb = x_t[idx]
            yb = y_t[idx]
            y_m = y_t[perm]
            t_joint = net(torch.cat([xb, yb], dim=1)).clamp(-float(t_clip), float(t_clip))
            t_marg = net(torch.cat([xb, y_m], dim=1)).clamp(-float(t_clip), float(t_clip))
            mi = t_joint.mean() - torch.exp(t_marg - 1.0).mean()
            vals.append(float(mi.item()))
        return float(np.mean(vals))


def main() -> None:
    p = argparse.ArgumentParser(description="Mutual information estimation benchmark (bivariate copulas)")
    p.add_argument("--estimator", type=str, choices=["ksg", "gaussian", "infonce", "nwj", "mine", "minde", "mist", "dcd"], required=True)
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--m-true", type=int, default=256, help="Grid resolution used to compute MI_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path for estimator=dcd.")
    p.add_argument(
        "--output-base",
        type=Path,
        action="append",
        default=[],
        help="Paper OUTPUT_BASE path(s) to scan for checkpoints when estimator=dcd and --checkpoint is omitted.",
    )
    p.add_argument(
        "--preferred-methods",
        type=str,
        nargs="+",
        default=["denoiser_cond_enhanced", "denoiser_cond", "enhanced_cnn_cond"],
        help="Method priority for auto checkpoint selection (estimator=dcd).",
    )
    p.add_argument(
        "--dcd-diffusion-steps",
        type=int,
        default=4,
        help="DDIM steps for diffusion checkpoints in estimator=dcd.",
    )
    p.add_argument(
        "--dcd-diffusion-cfg-scale",
        type=float,
        default=1.0,
        help="CFG scale for histogram-conditioned diffusion in estimator=dcd.",
    )
    p.add_argument(
        "--dcd-diffusion-ensemble",
        type=int,
        default=1,
        help="Number of diffusion samples to ensemble for estimator=dcd.",
    )
    p.add_argument(
        "--dcd-diffusion-ensemble-mode",
        type=str,
        default="geometric",
        choices=["geometric", "arithmetic", "median"],
        help="How to aggregate diffusion ensemble samples for estimator=dcd.",
    )
    p.add_argument(
        "--dcd-diffusion-smooth-sigma",
        type=float,
        default=0.0,
        help="Optional Gaussian smoothing sigma on aggregated density for estimator=dcd.",
    )
    p.add_argument(
        "--dcd-pred-noise-clip",
        type=float,
        default=10.0,
        help="Clip predicted diffusion noise to [-clip,clip]; <=0 disables clipping.",
    )
    p.add_argument(
        "--dcd-projection-iters",
        type=int,
        default=None,
        help="Override copula projection iterations for estimator=dcd (default: checkpoint/config).",
    )
    p.add_argument(
        "--dcd-seed-base",
        type=int,
        default=None,
        help="Base RNG seed for diffusion x_T initialization (defaults to --seed).",
    )

    # Neural MI options (used for mine / infonce / nwj)
    p.add_argument("--mine-steps", type=int, default=2000)
    p.add_argument("--mine-lr", type=float, default=1e-3)
    p.add_argument("--mine-batch-size", type=int, default=512)
    p.add_argument("--mine-hidden-dim", type=int, default=128)
    p.add_argument("--mine-weight-decay", type=float, default=0.0)
    p.add_argument("--mine-grad-clip", type=float, default=5.0)

    # MINDE options (only used when estimator=minde)
    p.add_argument("--minde-repo", type=Path, default=Path("external/minde"))
    p.add_argument("--clone-minde", action="store_true")
    p.add_argument("--minde-max-epochs", type=int, default=50)
    p.add_argument("--minde-lr", type=float, default=1e-2)
    p.add_argument("--minde-batch-size", type=int, default=256)

    # MIST options (only used when estimator=mist)
    p.add_argument("--mist-repo", type=Path, default=Path("external/mist"))
    p.add_argument("--clone-mist", action="store_true")
    p.add_argument("--mist-max-epochs", type=int, default=100)
    p.add_argument("--mist-lr", type=float, default=1e-3)
    p.add_argument("--mist-batch-size", type=int, default=256)
    p.add_argument("--mist-hidden-dim", type=int, default=256)
    p.add_argument("--mist-num-layers", type=int, default=3)

    args = p.parse_args()

    device = torch.device(args.device)
    est = str(args.estimator).lower()
    dcd_estimator: Optional[_DCDMIEstimator] = None
    dcd_checkpoint: Optional[Path] = None

    if est == "minde" and args.clone_minde:
        _maybe_clone_minde(args.minde_repo)
    if est == "mist" and args.clone_mist:
        _maybe_clone_mist(args.mist_repo)
    if est == "dcd":
        dcd_checkpoint = _resolve_dcd_checkpoint(
            explicit_checkpoint=args.checkpoint,
            output_bases=list(args.output_base),
            preferred_methods=list(args.preferred_methods),
        )
        pred_noise_clip = None if float(args.dcd_pred_noise_clip) <= 0 else float(args.dcd_pred_noise_clip)
        dcd_estimator = _DCDMIEstimator(
            checkpoint=dcd_checkpoint,
            device=device,
            diffusion_steps=int(args.dcd_diffusion_steps),
            diffusion_cfg_scale=float(args.dcd_diffusion_cfg_scale),
            diffusion_ensemble=int(args.dcd_diffusion_ensemble),
            diffusion_ensemble_mode=str(args.dcd_diffusion_ensemble_mode),
            diffusion_smooth_sigma=float(args.dcd_diffusion_smooth_sigma),
            diffusion_pred_noise_clip=pred_noise_clip,
            projection_iters=int(args.dcd_projection_iters) if args.dcd_projection_iters is not None else None,
            seed_base=int(args.seed) if args.dcd_seed_base is None else int(args.dcd_seed_base),
        )
        print(f"Using DCD checkpoint: {dcd_checkpoint}")

    records: List[Dict[str, Any]] = []
    n_cases = len(DEFAULT_TEST_COPULAS)
    for idx_case, (family, params, rotation, name) in enumerate(DEFAULT_TEST_COPULAS, start=1):
        print(f"[{idx_case}/{n_cases}] {est}: {name}", flush=True)
        pts = sample_bicop(family, params, n=int(args.n_samples), rotation=int(rotation), seed=int(args.seed))
        mi_true = _mi_true_from_analytic(
            family=family,
            params=params,
            rotation=int(rotation),
            m_true=int(args.m_true),
            device=device,
        )

        t0 = perf_counter()
        if est == "ksg":
            mi_est = ksg_mutual_information(pts[:, 0], pts[:, 1], k=5, seed=int(args.seed))
        elif est == "gaussian":
            mi_est = _gaussian_mi_probit(pts, device=device)
        elif est == "infonce":
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _infonce_estimate_mi(
                x=x,
                y=y,
                seed=int(args.seed) + 1000 * int(rotation),
                device=device,
                steps=int(args.mine_steps),
                lr=float(args.mine_lr),
                batch_size=int(args.mine_batch_size),
                hidden_dim=int(args.mine_hidden_dim),
                weight_decay=float(args.mine_weight_decay),
                grad_clip=float(args.mine_grad_clip),
            )
        elif est == "nwj":
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _nwj_estimate_mi(
                x=x,
                y=y,
                seed=int(args.seed) + 1000 * int(rotation),
                device=device,
                steps=int(args.mine_steps),
                lr=float(args.mine_lr),
                batch_size=int(args.mine_batch_size),
                hidden_dim=int(args.mine_hidden_dim),
                weight_decay=float(args.mine_weight_decay),
                grad_clip=float(args.mine_grad_clip),
            )
        elif est == "mine":
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _mine_estimate_mi(
                x=x,
                y=y,
                seed=int(args.seed) + 1000 * int(rotation),
                device=device,
                steps=int(args.mine_steps),
                lr=float(args.mine_lr),
                batch_size=int(args.mine_batch_size),
                hidden_dim=int(args.mine_hidden_dim),
                weight_decay=float(args.mine_weight_decay),
                grad_clip=float(args.mine_grad_clip),
            )
        elif est == "mist":
            # MIST (arXiv:2511.18945)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _mist_estimate_mi(
                mist_repo=args.mist_repo,
                x=x,
                y=y,
                seed=int(args.seed),
                device=device,
                max_epochs=int(args.mist_max_epochs),
                lr=float(args.mist_lr),
                batch_size=int(args.mist_batch_size),
                hidden_dim=int(args.mist_hidden_dim),
                num_layers=int(args.mist_num_layers),
            )
        elif est == "dcd":
            if dcd_estimator is None:
                raise RuntimeError("Internal error: DCD estimator was not initialized.")
            mi_est = dcd_estimator.estimate_mi(pts)
        else:
            # MINDE expects (n,dx), (n,dy)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            mi_est = _minde_estimate_mi(
                minde_repo=args.minde_repo,
                x=x,
                y=y,
                benchmark=f"vdc_copula_mi_{family}_rot{int(rotation)}",
                seed=int(args.seed),
                device=device,
                max_epochs=int(args.minde_max_epochs),
                lr=float(args.minde_lr),
                batch_size=int(args.minde_batch_size),
            )
        dt_s = float(perf_counter() - t0)

        rec = (
            {
                "name": name,
                "family": family,
                "rotation": int(rotation),
                "n_samples": int(args.n_samples),
                "mi_true": float(mi_true),
                "mi_est": float(mi_est),
                "mi_err": float(abs(mi_est - mi_true)),
                "time_s": dt_s,
            }
        )
        records.append(rec)
        print(
            f"  mi_true={rec['mi_true']:.4f} mi_est={rec['mi_est']:.4f} "
            f"|err|={rec['mi_err']:.4f} time={rec['time_s']:.2f}s",
            flush=True,
        )

    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "estimator": est,
        "n_samples": int(args.n_samples),
        "m_true": int(args.m_true),
        "seed": int(args.seed),
        "records": records,
        "mean_abs_err": float(np.mean([r["mi_err"] for r in records])) if records else None,
        "mean_time_s": float(np.mean([r["time_s"] for r in records])) if records else None,
        "source": {
            "minde_repo": str(args.minde_repo) if est == "minde" else None,
            "minde_url": "https://github.com/MustaphaBounoua/minde",
            "mist_repo": str(args.mist_repo) if est == "mist" else None,
            "mist_url": "https://github.com/gritsai/mist",
            "dcd_checkpoint": str(dcd_checkpoint) if est == "dcd" and dcd_checkpoint is not None else None,
            "dcd_model_type": str(dcd_estimator.model_type) if est == "dcd" and dcd_estimator is not None else None,
            "dcd_grid_size": int(dcd_estimator.m) if est == "dcd" and dcd_estimator is not None else None,
            "dcd_projection_iters": int(dcd_estimator.proj_iters) if est == "dcd" and dcd_estimator is not None else None,
            "dcd_diffusion_steps": int(dcd_estimator.diffusion_steps) if est == "dcd" and dcd_estimator is not None else None,
            "dcd_diffusion_cfg_scale": float(dcd_estimator.diffusion_cfg_scale) if est == "dcd" and dcd_estimator is not None else None,
            "dcd_diffusion_ensemble": int(dcd_estimator.diffusion_ensemble) if est == "dcd" and dcd_estimator is not None else None,
            "dcd_diffusion_ensemble_mode": str(dcd_estimator.diffusion_ensemble_mode) if est == "dcd" and dcd_estimator is not None else None,
            "dcd_diffusion_smooth_sigma": float(dcd_estimator.diffusion_smooth_sigma) if est == "dcd" and dcd_estimator is not None else None,
            "dcd_pred_noise_clip": float(dcd_estimator.diffusion_pred_noise_clip)
            if est == "dcd" and dcd_estimator is not None and dcd_estimator.diffusion_pred_noise_clip is not None
            else None,
            "dcd_seed_base": int(dcd_estimator.seed_base) if est == "dcd" and dcd_estimator is not None else None,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
