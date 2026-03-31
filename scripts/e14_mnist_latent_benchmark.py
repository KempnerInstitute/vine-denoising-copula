#!/usr/bin/env python3
# ruff: noqa: N803, N806
"""E14: image-latent density benchmark on MNIST.

This benchmark keeps the paper's evaluation protocol intact:
1. map image examples to a continuous latent representation
2. fit empirical marginals on the latent train set
3. compare copula-space density models on held-out latent codes

The default latent representation is PCA for stability and speed. A small
autoencoder option is also provided for follow-up experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kstest, norm
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, FashionMNIST

LOG_2PI = float(math.log(2.0 * math.pi))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _paper_outputs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "paper_outputs"


def _clip_unit(U: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(U, dtype=np.float64), eps, 1.0 - eps)


def _probit_transform(U: np.ndarray) -> np.ndarray:
    return norm.ppf(_clip_unit(U))


def _probit_logabsdet(z: torch.Tensor) -> torch.Tensor:
    return (0.5 * z.pow(2) + 0.5 * LOG_2PI).sum(dim=-1)


def _nll_bits_per_dim_from_logpdf(logpdf: np.ndarray, d: int) -> float:
    ll = float(np.mean(logpdf))
    return float((-ll) / (max(1, int(d)) * np.log(2.0)))


def _pit_ks_stat(W: np.ndarray) -> float:
    W = np.asarray(W, dtype=np.float64)
    stats: List[float] = []
    for j in range(W.shape[1]):
        stat, _p = kstest(W[:, j], "uniform")
        stats.append(float(stat))
    return float(np.mean(stats)) if stats else float("nan")


def _nearest_corr(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    corr = np.asarray(corr, dtype=np.float64)
    corr = 0.5 * (corr + corr.T)
    w, v = np.linalg.eigh(corr)
    w = np.clip(w, eps, None)
    corr_psd = (v * w[None, :]) @ v.T
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.maximum(1e-12, d[:, None] * d[None, :])
    np.fill_diagonal(corr_psd, 1.0)
    return corr_psd


def _gaussian_copula_logpdf(U: np.ndarray, corr: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    Z = norm.ppf(np.clip(U, 1e-8, 1.0 - 1e-8))
    p = int(Z.shape[1])
    inv_corr = np.linalg.inv(corr)
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        raise RuntimeError("Correlation matrix is not SPD.")
    quad = np.einsum("ni,ij,nj->n", Z, inv_corr - np.eye(p), Z, optimize=True)
    return -0.5 * logdet - 0.5 * quad


def _load_checkpoint_model(ckpt_path: Path, device: str):
    from vdc.train.unified_trainer import build_model

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")
    config = ckpt.get("config", {})
    model_type_raw = str(config.get("model", {}).get("type", "diffusion_unet"))
    model_type = "diffusion_unet" if model_type_raw.startswith("diffusion_unet") else model_type_raw
    model = build_model(model_type, config, torch.device(device))
    model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    model.eval()
    return model, model_type, config


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = dim
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2 * dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_layers: int, mask: torch.Tensor, scale_clip: float = 3.0) -> None:
        super().__init__()
        self.net = _MLP(dim, hidden_dim, hidden_layers)
        self.register_buffer("mask", mask.view(1, dim))
        self.scale_clip = float(scale_clip)

    def _st(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x * self.mask)
        s, t = torch.chunk(h, 2, dim=-1)
        s = torch.tanh(s) * self.scale_clip
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        return s, t

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self._st(y)
        x = y * self.mask + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = (-s).sum(dim=-1)
        return x, logdet


class RealNVP(nn.Module):
    def __init__(self, dim: int, num_layers: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(int(num_layers)):
            mask_np = np.array([(j + i) % 2 for j in range(dim)], dtype=np.float32)
            mask = torch.tensor(mask_np)
            layers.append(_CouplingLayer(dim, hidden_dim, hidden_layers, mask))
        self.layers = nn.ModuleList(layers)

    def log_prob_z(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            logdet = logdet + ld
        base = -0.5 * (x.pow(2) + LOG_2PI).sum(dim=-1)
        return base + logdet

    def log_prob_u(self, u: torch.Tensor) -> torch.Tensor:
        z = torch.special.ndtri(u.clamp(1e-6, 1.0 - 1e-6))
        return self.log_prob_z(z) + _probit_logabsdet(z)


def _fit_flow(
    z_train: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    num_layers: int,
    hidden_dim: int,
    hidden_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    patience: int,
) -> Tuple[RealNVP, float]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(z_train.shape[0])
    n_val = max(1, int(round(float(val_fraction) * z_train.shape[0])))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:] if n_val < len(perm) else perm[:1]

    z_tr = torch.tensor(z_train[tr_idx], dtype=torch.float32)
    z_val = torch.tensor(z_train[val_idx], dtype=torch.float32)

    model = RealNVP(
        dim=int(z_train.shape[1]),
        num_layers=int(num_layers),
        hidden_dim=int(hidden_dim),
        hidden_layers=int(hidden_layers),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-6)
    train_loader = DataLoader(
        TensorDataset(z_tr),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(z_val),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad_epochs = 0
    t0 = perf_counter()

    for _epoch in range(int(epochs)):
        model.train()
        for (zb,) in train_loader:
            zb = zb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = -model.log_prob_z(zb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for (zb,) in val_loader:
                zb = zb.to(device, non_blocking=True)
                val_losses.append(float((-model.log_prob_z(zb).mean()).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    elapsed = float(perf_counter() - t0)
    model.load_state_dict(best_state)
    model.eval()
    return model, elapsed


def _eval_logprob_u(model: RealNVP, U: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    u_t = torch.tensor(_clip_unit(U), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(u_t),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    vals: List[np.ndarray] = []
    with torch.no_grad():
        for (ub,) in loader:
            ub = ub.to(device, non_blocking=True)
            vals.append(model.log_prob_u(ub).detach().cpu().numpy())
    return np.concatenate(vals, axis=0)


class MLPEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MLPAutoEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = MLPEncoder(latent_dim)
        self.decoder = MLPDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


@dataclass
class LatentResult:
    train_latent: np.ndarray
    test_latent: np.ndarray
    metadata: Dict[str, Any]


def _load_image_dataset(name: str, root: Path, download: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    name_l = str(name).lower()
    if name_l == "mnist":
        ds_cls = MNIST
    elif name_l == "fashion_mnist":
        ds_cls = FashionMNIST
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train_ds = ds_cls(root=str(root), train=True, download=bool(download))
    test_ds = ds_cls(root=str(root), train=False, download=bool(download))

    X_train = train_ds.data.numpy().astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy().astype(np.int64)
    X_test = test_ds.data.numpy().astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy().astype(np.int64)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, y_train, X_test, y_test


def _subsample(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= int(max_rows):
        return X, y
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(X.shape[0], size=int(max_rows), replace=False)
    return X[idx], y[idx]


def _fit_pca_latent(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    latent_dim: int,
    seed: int,
) -> LatentResult:
    t0 = perf_counter()
    pca = PCA(n_components=int(latent_dim), svd_solver="randomized", random_state=int(seed))
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    fit_s = float(perf_counter() - t0)
    X_test_recon = pca.inverse_transform(Z_test)
    test_recon_mse = float(np.mean((X_test_recon - X_test) ** 2))
    return LatentResult(
        train_latent=np.asarray(Z_train, dtype=np.float64),
        test_latent=np.asarray(Z_test, dtype=np.float64),
        metadata={
            "latent_method": "pca",
            "latent_dim": int(latent_dim),
            "fit_s": fit_s,
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "test_recon_mse": test_recon_mse,
        },
    )


def _fit_autoencoder_latent(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    latent_dim: int,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> LatentResult:
    torch.manual_seed(int(seed))
    model = MLPAutoEncoder(latent_dim=int(latent_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    train_t = torch.tensor(X_train, dtype=torch.float32)
    test_t = torch.tensor(X_test, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_t),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        TensorDataset(test_t),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    t0 = perf_counter()
    for _epoch in range(int(epochs)):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
    fit_s = float(perf_counter() - t0)

    def _encode(loader: DataLoader) -> Tuple[np.ndarray, float]:
        model.eval()
        latents: List[np.ndarray] = []
        mses: List[float] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=True)
                zb = model.encode(xb)
                xhat = model.decoder(zb)
                latents.append(zb.detach().cpu().numpy())
                mses.append(float(loss_fn(xhat, xb).item()))
        return np.concatenate(latents, axis=0), float(np.mean(mses)) if mses else float("nan")

    Z_train, train_mse = _encode(train_loader)
    Z_test, test_mse = _encode(test_loader)
    return LatentResult(
        train_latent=np.asarray(Z_train, dtype=np.float64),
        test_latent=np.asarray(Z_test, dtype=np.float64),
        metadata={
            "latent_method": "mlp_autoencoder",
            "latent_dim": int(latent_dim),
            "fit_s": fit_s,
            "train_recon_mse": train_mse,
            "test_recon_mse": test_mse,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
        },
    )


def _write_summary_md(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append(f"# E14 latent image benchmark: {payload.get('dataset', 'unknown')}")
    lines.append("")
    latent = payload.get("latent", {})
    lines.append(f"- latent method: `{latent.get('latent_method', '--')}`")
    lines.append(f"- latent dim: `{latent.get('latent_dim', '--')}`")
    lines.append(f"- n_train: `{payload.get('n_train', '--')}`")
    lines.append(f"- n_test: `{payload.get('n_test', '--')}`")
    if isinstance(latent, dict):
        for key in ["explained_variance_ratio_sum", "test_recon_mse", "train_recon_mse", "fit_s"]:
            if key in latent:
                lines.append(f"- {key}: `{latent[key]}`")
    lines.append("")
    lines.append("| Method | Fit s | NLL bits/dim | PIT KS |")
    lines.append("| --- | ---: | ---: | ---: |")
    for rec in payload.get("records", []):
        method = str(rec.get("method", "--"))
        fit_s = rec.get("fit_s", "--")
        nll = rec.get("nll_bits_per_dim", "--")
        pit = rec.get("pit_ks", "--")
        fit_s_s = f"{float(fit_s):.2f}" if isinstance(fit_s, (float, int)) else str(fit_s)
        nll_s = f"{float(nll):.4f}" if isinstance(nll, (float, int)) else str(nll)
        pit_s = f"{float(pit):.4f}" if isinstance(pit, (float, int)) else str(pit)
        lines.append(f"| {method} | {fit_s_s} | {nll_s} | {pit_s} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="E14 latent image benchmark.")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--latent-method", type=str, default="pca", choices=["pca", "mlp_autoencoder"])
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--max-train", type=int, default=20000)
    p.add_argument("--max-test", type=int, default=5000)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--ae-batch-size", type=int, default=512)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--flow-epochs", type=int, default=25)
    p.add_argument("--flow-batch-size", type=int, default=2048)
    p.add_argument("--flow-eval-batch-size", type=int, default=4096)
    p.add_argument("--flow-lr", type=float, default=1e-3)
    p.add_argument("--flow-val-fraction", type=float, default=0.1)
    p.add_argument("--flow-patience", type=int, default=5)
    p.add_argument("--flow-num-layers", type=int, default=8)
    p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-hidden-layers", type=int, default=2)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-summary", type=Path, default=None)
    args = p.parse_args()

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from vdc.data.paths import data_root as _data_root
    from vdc.data.tabular import EmpiricalMarginals
    from vdc.utils.paper import choose_best_checkpoint
    from vdc.vine.api import VineCopulaModel

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    data_dir = Path(args.data_dir) if args.data_dir is not None else (_data_root() / "vision")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_json if args.out_json is not None else (_paper_outputs_dir() / "e14_mnist_latent_results.json")
    out_summary = args.out_summary if args.out_summary is not None else (_paper_outputs_dir() / "e14_mnist_latent_summary.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = _load_image_dataset(str(args.dataset), data_dir, download=bool(args.download))
    X_train, y_train = _subsample(X_train, y_train, max_rows=int(args.max_train), seed=int(args.seed))
    X_test, y_test = _subsample(X_test, y_test, max_rows=int(args.max_test), seed=int(args.seed) + 1)

    device = torch.device(str(args.device))

    if str(args.latent_method) == "pca":
        latent = _fit_pca_latent(X_train, X_test, latent_dim=int(args.latent_dim), seed=int(args.seed))
    else:
        latent = _fit_autoencoder_latent(
            X_train,
            X_test,
            latent_dim=int(args.latent_dim),
            seed=int(args.seed),
            device=device,
            epochs=int(args.ae_epochs),
            batch_size=int(args.ae_batch_size),
            lr=float(args.ae_lr),
        )

    # Tiny jitter helps avoid accidental rank ties in empirical marginals.
    rng = np.random.default_rng(int(args.seed))
    X_lat_tr = latent.train_latent + 1e-6 * rng.standard_normal(latent.train_latent.shape)
    X_lat_te = latent.test_latent + 1e-6 * rng.standard_normal(latent.test_latent.shape)

    marg = EmpiricalMarginals.fit(X_lat_tr)
    U_tr = _clip_unit(marg.transform(X_lat_tr))
    U_te = _clip_unit(marg.transform(X_lat_te))
    d = int(U_tr.shape[1])

    output_bases: List[Path] = []
    if os.environ.get("OUTPUT_BASE"):
        output_bases.append(Path(os.environ["OUTPUT_BASE"]))
    output_bases.append(Path("/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula"))
    ckpt = choose_best_checkpoint(
        output_bases=output_bases,
        preferred_methods=["denoiser_cond_enhanced", "denoiser_cond", "enhanced_cnn_cond"],
        metric="mean_ise",
        prefer_joint=True,
    )
    if ckpt is None or (not Path(ckpt).exists()):
        raise SystemExit("No pretrained VDC checkpoint found.")

    model, model_type, cfg = _load_checkpoint_model(Path(ckpt), device=str(device))
    m_model = int(cfg.get("data", {}).get("m", 64))
    proj_iters = int(cfg.get("training", {}).get("projection_iters", 30))

    records: List[Dict[str, Any]] = []

    vine = VineCopulaModel(
        vine_type="dvine",
        m=int(m_model),
        device=str(device),
        projection_iters=int(proj_iters),
        hfunc_use_spline=False,
        batch_edges=True,
        edge_batch_size=256,
    )
    t0 = perf_counter()
    vine.fit(U_tr, diffusion_model=model, diffusion=None, verbose=False)
    fit_s = float(perf_counter() - t0)
    logpdf = vine.logpdf(U_te)
    pit = vine.rosenblatt(U_te)
    records.append(
        {
            "method": "ours",
            "fit_s": fit_s,
            "nll_bits_per_dim": _nll_bits_per_dim_from_logpdf(logpdf, d),
            "pit_ks": _pit_ks_stat(pit),
            "checkpoint": str(ckpt),
            "model_type": str(model_type),
            "m": int(m_model),
        }
    )

    t0 = perf_counter()
    Z_tr = norm.ppf(np.clip(U_tr, 1e-8, 1.0 - 1e-8))
    corr = np.corrcoef(Z_tr, rowvar=False)
    corr = _nearest_corr(corr, eps=1e-6)
    fit_s = float(perf_counter() - t0)
    logpdf = _gaussian_copula_logpdf(U_te, corr)
    records.append(
        {
            "method": "gaussian_copula",
            "fit_s": fit_s,
            "nll_bits_per_dim": _nll_bits_per_dim_from_logpdf(logpdf, d),
            "pit_ks": None,
        }
    )

    Z_tr = _probit_transform(U_tr)
    flow, fit_s = _fit_flow(
        Z_tr,
        device=device,
        seed=int(args.seed),
        num_layers=int(args.flow_num_layers),
        hidden_dim=int(args.flow_hidden_dim),
        hidden_layers=int(args.flow_hidden_layers),
        epochs=int(args.flow_epochs),
        batch_size=int(args.flow_batch_size),
        lr=float(args.flow_lr),
        val_fraction=float(args.flow_val_fraction),
        patience=int(args.flow_patience),
    )
    logpdf_u = _eval_logprob_u(flow, U_te, device=device, batch_size=int(args.flow_eval_batch_size))
    records.append(
        {
            "method": "flow_realnvp",
            "fit_s": float(fit_s),
            "nll_bits_per_dim": _nll_bits_per_dim_from_logpdf(logpdf_u, d),
            "pit_ks": None,
            "transform": "probit",
            "flow_type": "realnvp",
        }
    )

    payload: Dict[str, Any] = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "dataset": str(args.dataset),
        "n_train": int(U_tr.shape[0]),
        "n_test": int(U_te.shape[0]),
        "latent": latent.metadata,
        "records": records,
        "settings": {
            "device": str(args.device),
            "seed": int(args.seed),
            "data_dir": str(data_dir),
        },
    }

    out_json.write_text(json.dumps(payload, indent=2))
    _write_summary_md(out_summary, payload)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
