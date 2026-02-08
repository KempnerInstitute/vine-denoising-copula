"""Unified training script for copula density models.

Supports model types:
  - baseline_cnn      (CopulaDensityCNN)
  - enhanced_cnn      (EnhancedCopulaDensityCNN)
  - denoiser          (CopulaDenoiser)
  - diffusion_unet    (GridUNet + CopulaAwareDiffusion noise prediction)

Key Features:
  - Single entry point for all architectures
  - Shared loss components (NLL, ISE, marginal, tail, h-function)
  - Optional copula projection frequency control
  - Mixed precision training
  - Compatible with torchrun / SLURM (DDP)

Usage examples:
  torchrun --nproc_per_node=4 scripts/train_unified.py --config configs/train_multinode_cnn_enhanced.yaml --model-type enhanced_cnn
  torchrun --nproc_per_node=4 scripts/train_unified.py --config configs/train_multinode_denoiser.yaml --model-type denoiser

Config expectations:
  data: { m, n_samples_per_copula, copula_families }
  training: { batch_size, max_steps, learning_rate, projection_iters, log_interval, save_interval, loss_weights }
  model: architecture hyperparameters (reused per type)

"""
import os
import sys
import argparse
import yaml
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt

os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

from vdc.data.onthefly import OnTheFlyCopulaDataset  # unified on-the-fly dataset
from vdc.models.copula_cnn import CopulaDensityCNN
from vdc.models.copula_cnn_enhanced import EnhancedCopulaDensityCNN
from vdc.models.copula_denoiser import CopulaDenoiser
from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project, sinkhorn_project_density
from vdc.losses import tail_weighted_loss  # keep tail utility for point sampling fallback
from vdc.utils.ipfp_log import ipfp_project_log, marginal_deviation
from vdc.utils.histogram import anti_aliased_hist
from vdc.utils.metrics import kendall_tau, tail_dependence_from_grid
from vdc.utils.probit_transform import (
    copula_density_to_probit_density, 
    probit_density_to_copula_density,
    copula_logdensity_to_probit_logdensity,
    probit_logdensity_to_copula_logdensity
)


def setup_distributed():
    """Robust distributed setup supporting torchrun and SLURM env vars."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK']); world_size = int(os.environ['WORLD_SIZE']); local_rank = int(os.environ.get('LOCAL_RANK',0))
    elif 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        rank = int(os.environ['SLURM_PROCID']); world_size = int(os.environ['SLURM_NTASKS']); local_rank = int(os.environ.get('SLURM_LOCALID',0))
    else:
        rank = 0; world_size = 1; local_rank = 0
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        # Set 30-minute timeout to handle expensive visualization (1000 diffusion forward passes)
        # Default 600s (10 min) causes watchdog timeout at step 1000
        import datetime
        timeout = datetime.timedelta(seconds=1800)  # 30 minutes
        dist.init_process_group(backend='nccl', init_method='env://', 
                              world_size=world_size, rank=rank, timeout=timeout)
        dist.barrier()
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_model(model_type: str, config: Dict, device: torch.device):
    m = config['data']['m']
    mcfg = config.get('model', {})
    if model_type == 'baseline_cnn':
        model = CopulaDensityCNN(m=m,
                                base_channels=mcfg.get('base_channels',128),
                                n_blocks=mcfg.get('n_blocks',3),
                                dropout=mcfg.get('dropout',0.1)).to(device)
        # Metadata for downstream inference utilities
        model.vdc_use_coordinates = bool(mcfg.get('use_coordinates', False))
        model.vdc_use_probit_coords = bool(mcfg.get('use_probit_coords', False))
        model.vdc_use_log_n = bool(mcfg.get('use_log_n', False))
        model.vdc_probit_coord_eps = float(mcfg.get('probit_coord_eps', 1e-4))
        model.vdc_transform_to_probit_space = bool(mcfg.get('transform_to_probit_space', False))
        return model
    if model_type == 'enhanced_cnn':
        use_coords = bool(mcfg.get('use_coordinates', True))
        use_log_n = bool(mcfg.get('use_log_n', False))
        input_ch = 1 + (1 if use_log_n else 0) + (2 if use_coords else 0)
        model = EnhancedCopulaDensityCNN(m=m,
                        base_channels=mcfg.get('base_channels',128),
                        n_blocks=mcfg.get('n_blocks',3),
                        dropout=mcfg.get('dropout',0.1),
                        input_channels=input_ch,
                        output_mode=mcfg.get('output_mode','log'),
                        time_conditioning=mcfg.get('time_conditioning', False),
                        time_emb_dim=mcfg.get('time_emb_dim',256),
                        multi_scale_aux=mcfg.get('multi_scale_aux', False),
                        aux_scales=tuple(mcfg.get('aux_scales',[2,4]))).to(device)
        model.vdc_use_coordinates = use_coords
        model.vdc_use_probit_coords = bool(mcfg.get('use_probit_coords', False))
        model.vdc_use_log_n = use_log_n
        model.vdc_probit_coord_eps = float(mcfg.get('probit_coord_eps', 1e-4))
        model.vdc_transform_to_probit_space = bool(mcfg.get('transform_to_probit_space', False))
        return model
    if model_type == 'denoiser':
        use_coords = bool(mcfg.get('use_coordinates', True))
        use_log_n = bool(mcfg.get('use_log_n', False))
        input_ch = 1 + (1 if use_log_n else 0) + (2 if use_coords else 0)
        model = CopulaDenoiser(m=m,
                              input_channels=input_ch,
                              base_channels=mcfg.get('base_channels',128),
                              depth=mcfg.get('depth',4),
                              blocks_per_level=mcfg.get('blocks_per_level',2),
                              time_emb_dim=mcfg.get('time_emb_dim',256),
                              dropout=mcfg.get('dropout',0.1),
                              output_mode=mcfg.get('output_mode','log')).to(device)
        model.vdc_use_coordinates = use_coords
        model.vdc_use_probit_coords = bool(mcfg.get('use_probit_coords', False))
        model.vdc_use_log_n = use_log_n
        model.vdc_probit_coord_eps = float(mcfg.get('probit_coord_eps', 1e-4))
        model.vdc_transform_to_probit_space = bool(mcfg.get('transform_to_probit_space', False))
        return model
    if model_type == 'diffusion_unet':
        model = GridUNet(m=m,
                        in_channels=mcfg.get('in_channels', 1),
                        base_channels=mcfg.get('base_channels',64),
                        channel_mults=tuple(mcfg.get('channel_mults',[1,2,3,4])),
                        num_res_blocks=mcfg.get('num_res_blocks',2),
                        attention_resolutions=tuple(mcfg.get('attention_resolutions',[16,8])),
                        dropout=mcfg.get('dropout',0.1),
                        time_emb_dim=mcfg.get('time_emb_dim', 256),
                        upsample_mode=mcfg.get('upsample_mode', 'transpose')).to(device)
        model.vdc_use_coordinates = False
        model.vdc_use_probit_coords = False
        model.vdc_use_log_n = True  # GridUNet supports log_n embedding
        model.vdc_probit_coord_eps = float(mcfg.get('probit_coord_eps', 1e-4))
        model.vdc_transform_to_probit_space = bool(mcfg.get('transform_to_probit_space', False))
        return model
    raise ValueError(f"Unknown model_type {model_type}")


def build_coordinates(B, m, device, probit=False, eps: float = 1e-6):
    """Build (u,v) coordinate channels or their probit transforms.

    Args:
        B: batch size
        m: grid resolution
        device: target device
        probit: whether to apply Φ^{-1}
        eps: clamp epsilon to avoid extreme tails; can be tuned via config (probit_coord_eps)
    Returns:
        Tensor of shape (B,2,m,m)
    """
    u = torch.linspace(0.5/m, 1-0.5/m, m, device=device)
    v = torch.linspace(0.5/m, 1-0.5/m, m, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    if probit:
        # Use configurable epsilon and a soft clipping to reduce very large |z| which destabilizes early training
        clamp_eps = max(eps, 1.0/(m*m))  # heuristic: tighter than uniform grid spacing but avoids erfinv saturation
        uu = torch.erfinv(2*uu.clamp(clamp_eps,1-clamp_eps)-1)* (2**0.5)
        vv = torch.erfinv(2*vv.clamp(clamp_eps,1-clamp_eps)-1)* (2**0.5)
    grid = torch.stack([uu, vv], dim=0).unsqueeze(0).repeat(B,1,1,1)
    return grid


def build_binning(m: int, mode: str = "uniform", z_max: float = 4.5):
    if mode == "uniform":
        edges = np.linspace(0.0, 1.0, m + 1, dtype=np.float64)
    elif mode == "probit":
        z_edges = torch.linspace(-z_max, z_max, m + 1, dtype=torch.float64)
        edges = (0.5 * (1.0 + torch.erf(z_edges / math.sqrt(2.0)))).numpy()
        edges[0] = 0.0
        edges[-1] = 1.0
        np.clip(edges, 0.0, 1.0, out=edges)
        edges = np.maximum.accumulate(edges)
        edges[-1] = 1.0
    else:
        raise ValueError(f"Unknown binning mode: {mode}")
    widths = np.diff(edges)
    centers = edges[:-1] + 0.5 * widths
    return edges, centers, widths


def build_geometry(m: int, mode: str, device: torch.device, dtype: torch.dtype = torch.float32):
    _, row_centers, row_widths_np = build_binning(m, mode)
    _, col_centers, col_widths_np = build_binning(m, mode)
    row_widths = torch.from_numpy(row_widths_np).to(device=device, dtype=dtype)
    col_widths = torch.from_numpy(col_widths_np).to(device=device, dtype=dtype)
    row_widths_bc = row_widths.view(1, 1, m, 1)
    col_widths_bc = col_widths.view(1, 1, 1, m)
    area = row_widths_bc * col_widths_bc
    geometry = {
        "mode": mode,
        "row_widths": row_widths,
        "col_widths": col_widths,
        "row_widths_bc": row_widths_bc,
        "col_widths_bc": col_widths_bc,
        "area": area,
        "row_coords": torch.from_numpy(row_centers).to(device=device, dtype=dtype),
        "col_coords": torch.from_numpy(col_centers).to(device=device, dtype=dtype),
        "tail_mask": None,
    }
    return geometry


def _sanitize_density(grid: Optional[torch.Tensor], min_val: float = 1e-12, max_val: float = 1e6) -> Optional[torch.Tensor]:
    """Replace NaN/Inf values and clamp to a numerically safe range."""
    if grid is None:
        return None
    grid = torch.nan_to_num(grid, nan=min_val, posinf=max_val, neginf=min_val)
    return grid.clamp(min=min_val, max=max_val)


def _filter_metrics_dict(metrics: Dict[str, float]) -> Dict[str, float]:
    """Drop non-finite metric entries before logging or using them for early stopping."""
    clean: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            clean[key] = int(value)
            continue
        if isinstance(value, (float, np.floating)):
            if math.isnan(value) or math.isinf(value):
                continue
            clean[key] = float(value)
            continue
        clean[key] = value
    return clean


def _get_curriculum_weight(training_cfg: Dict, step: int) -> float:
    curr_cfg = training_cfg.get('curriculum', {})
    if not curr_cfg.get('enable', False):
        return 1.0
    warmup = curr_cfg.get('warmup_steps', 0)
    ramp = curr_cfg.get('rampup_steps', 0)
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    frac = (step - warmup) / max(1, ramp)
    return float(max(0.0, min(1.0, frac)))


def _get_tail_weight(training_cfg: Dict, step: int, base_weight: float) -> float:
    cfg = training_cfg.get('tail_curriculum', {})
    if not cfg.get('enable', False):
        return base_weight
    start = cfg.get('start_weight', base_weight)
    full = cfg.get('full_weight', base_weight)
    ramp = cfg.get('ramp_steps', 5000)
    frac = min(1.0, step / max(1, ramp))
    return float(start + frac * (full - start))


def _make_weight_factor(weight_factor, B: int, device: torch.device):
    if weight_factor is None:
        return None
    if isinstance(weight_factor, (int, float)):
        return torch.full((B,), float(weight_factor), device=device)
    if isinstance(weight_factor, torch.Tensor):
        wf = weight_factor.to(device).reshape(-1)
        if wf.shape[0] == 1:
            return wf.expand(B)
        if wf.shape[0] != B:
            raise ValueError("Weight factor tensor must match batch size")
        return wf
    raise TypeError(f"Unsupported weight_factor type: {type(weight_factor)}")


def compute_density_losses(
    pred_density: torch.Tensor,
    target_density: torch.Tensor,
    training_cfg: Dict,
    geometry: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float],
    tail_mask: torch.Tensor,
    model_output: Optional[Dict] = None,
    weight_factor=None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    device = pred_density.device
    B = pred_density.shape[0]
    wf = _make_weight_factor(weight_factor, B, device) if weight_factor is not None else None

    def apply_weight(per_sample: torch.Tensor) -> torch.Tensor:
        if wf is None:
            return per_sample.mean()
        return (per_sample * wf).mean()

    metrics = {}
    total_loss = pred_density.new_zeros(())
    area = geometry['area']  # per-cell area Δu_i * Δv_j (sums to 1 over the grid)
    row_widths_bc = geometry['row_widths_bc']
    col_widths_bc = geometry['col_widths_bc']
    component_losses: Dict[str, torch.Tensor] = {}

    use_log_ise = training_cfg.get('use_log_ise', False)
    if use_log_ise:
        pred_log = torch.log(pred_density.clamp(min=1e-12))
        target_log = torch.log(target_density.clamp(min=1e-12))
        diff = (pred_log - target_log) ** 2
    else:
        diff = (pred_density - target_density) ** 2

    # Integrated squared error with correct cell areas:
    # ISE ≈ Σ_ij (p_ij - q_ij)^2 * Δu_i * Δv_j.
    # For a uniform grid this reduces exactly to the previous mean over cells,
    # since area is constant and Σ area = 1; for probit / non-uniform grids it
    # correctly down-weights the tiny boundary cells and up-weights interior cells.
    ise_tensor = (diff * area).view(B, -1).sum(-1)
    metrics['ise'] = float(ise_tensor.mean().item())
    component_losses['ise'] = apply_weight(ise_tensor)
    if loss_weights.get('ise', 0.0) > 0.0:
        total_loss = total_loss + loss_weights['ise'] * component_losses['ise']

    T_mass = (target_density * area).clamp_min(1e-12)
    D_mass = (pred_density * area).clamp_min(1e-12)
    ce_tensor = -(T_mass * D_mass.log()).sum(dim=(-2, -1)).view(B)
    metrics['ce'] = float(ce_tensor.mean().item())
    component_losses['ce'] = apply_weight(ce_tensor)
    if loss_weights.get('ce', 0.0) > 0.0:
        total_loss = total_loss + loss_weights['ce'] * component_losses['ce']

    if tail_mask is not None:
        log_ratio = torch.log(D_mass) - torch.log(T_mass)
        tail_tensor = (tail_mask * (log_ratio**2)).view(B, -1).mean(-1)
        metrics['tail'] = float(tail_tensor.mean().item())
        component_losses['tail'] = apply_weight(tail_tensor)
        if loss_weights.get('tail', 0.0) > 0.0:
            total_loss = total_loss + loss_weights['tail'] * component_losses['tail']
    else:
        metrics['tail'] = 0.0
        component_losses['tail'] = pred_density.new_zeros(())

    ms_cfg = training_cfg.get('multi_scale', {'enable': True})
    loss_ms_per = None
    # Multi-scale loss encourages scale-consistent structure and reduces high-frequency artifacts.
    # This does *not* require auxiliary model outputs; we compute it by downsampling densities.
    if ms_cfg.get('enable', True) and loss_weights.get('ms', 0.0) > 0.0:
        loss_ms_per = pred_density.new_zeros(B)
        scales = ms_cfg.get('scales', [2, 4])
        ce_w = ms_cfg.get('ce_weight', 0.5)
        ise_w = ms_cfg.get('ise_weight', 0.25)
        for s in scales:
            if pred_density.shape[-1] % s != 0:
                continue
            D_ds = torch.nn.functional.avg_pool2d(pred_density, s)
            T_ds = torch.nn.functional.avg_pool2d(target_density, s)
            du_s = dv_s = 1.0 / (pred_density.shape[-1] // s)
            Tm_s = (T_ds * du_s * dv_s).clamp_min(1e-12)
            Dm_s = (D_ds * du_s * dv_s).clamp_min(1e-12)
            ce_s = -(Tm_s * Dm_s.log()).sum(dim=(-2, -1)).view(B)
            ise_s = ((D_ds - T_ds)**2).view(B, -1).mean(-1)
            loss_ms_per = loss_ms_per + ce_w * ce_s + ise_w * ise_s
        if loss_ms_per is not None:
            metrics['ms'] = float(loss_ms_per.mean().item())
            component_losses['ms'] = apply_weight(loss_ms_per)
            total_loss = total_loss + loss_weights['ms'] * component_losses['ms']
        else:
            metrics['ms'] = 0.0
            component_losses['ms'] = pred_density.new_zeros(())
    else:
        metrics['ms'] = 0.0
        component_losses['ms'] = pred_density.new_zeros(())

    # Smoothness prior: total variation regularization (optionally in log-space).
    # This helps reduce "patchy" high-frequency artifacts on the density grid,
    # especially at higher resolutions (e.g., m=128).
    tv_w = float(loss_weights.get('tv', 0.0) or 0.0)
    tv_in_log = bool(training_cfg.get('tv_in_log_space', True))
    if tv_w > 0.0:
        z = torch.log(pred_density.clamp(min=1e-12)) if tv_in_log else pred_density
        # z: (B,1,m,m). Compute per-sample TV in grid index space.
        diff_h = torch.abs(z[..., :, 1:] - z[..., :, :-1])
        diff_v = torch.abs(z[..., 1:, :] - z[..., :-1, :])
        tv_tensor = (diff_h.mean(dim=(-2, -1)) + diff_v.mean(dim=(-2, -1))).view(B)
        metrics['tv'] = float(tv_tensor.mean().item())
        component_losses['tv'] = apply_weight(tv_tensor)
        total_loss = total_loss + tv_w * component_losses['tv']
    else:
        metrics['tv'] = 0.0
        component_losses['tv'] = pred_density.new_zeros(())

    row_marg = (pred_density * col_widths_bc).sum(-1).clamp_min(1e-12)
    col_marg = (pred_density * row_widths_bc).sum(-2).clamp_min(1e-12)
    kl_row = (row_marg * row_marg.log()).sum(dim=-1)
    kl_col = (col_marg * col_marg.log()).sum(dim=-1)
    kl_per = kl_row + kl_col
    metrics['marg_kl'] = float(kl_per.mean().item())
    if loss_weights.get('marg_kl', 0.0) > 0.0:
        component_losses['marg_kl'] = apply_weight(kl_per)
        total_loss = total_loss + loss_weights['marg_kl'] * component_losses['marg_kl']
    else:
        component_losses['marg_kl'] = apply_weight(kl_per)

    return total_loss, metrics, component_losses


def _build_tail_mask(
    m: int,
    tau: float,
    device,
    row_coords: Optional[torch.Tensor] = None,
    col_coords: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if row_coords is None:
        u = torch.linspace(0.5/m, 1-0.5/m, m, device=device)
    else:
        u = row_coords.to(device)
    if col_coords is None:
        v = torch.linspace(0.5/m, 1-0.5/m, m, device=device)
    else:
        v = col_coords.to(device)
    U, V = torch.meshgrid(u, v, indexing='ij')
    mask = ((U < tau) | (V < tau) | (U > 1 - tau) | (V > 1 - tau)).float()
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,m,m)


def training_step(model_type, model, batch, device, config, diffusion=None, scaler=None, step=0, profiler=None, geometry=None):
    """Single training step with adaptive projection and modern loss suite.

    Handles diffusion separately. Direct models use CE + ISE + tail + multi-scale losses.
    Optionally replaces analytic target density with anti-aliased histogram from samples.
    
    Probit space handling (log-density mode):
    - If transform_to_probit_space=True: density is LOG-density in probit space
    - For projection: exp() to get density, transform to copula space, project, transform back, log()
    - Losses computed in log-space for numerical stability
    """
    density_raw = batch['density'].to(device)  # target density (B,1,m,m)
    is_log_density = batch.get('is_log_density', False)  # Flag indicating log-space
    # DataLoader may collate booleans into a tensor, extract the value
    if isinstance(is_log_density, torch.Tensor):
        # All samples in batch should have same flag, take first
        is_log_density = bool(is_log_density.flatten()[0].item())
    elif not isinstance(is_log_density, bool):
        is_log_density = bool(is_log_density)
    samples = batch.get('samples')
    if samples is not None:
        samples = samples.to(device)
    hist = batch.get('hist')
    if hist is not None:
        hist = hist.to(device)
    B, _, m, _ = density_raw.shape
    dtype = density_raw.dtype
    # Whether the *model space* target is in probit/normal-score coordinates.
    # In this mode, the dataset provides log f_Z(z_u,z_v) where z = Φ^{-1}(u),
    # but we still compute copula losses/projection in copula space.
    transform_to_probit_space = bool(config.get('model', {}).get('transform_to_probit_space', False))
    probit_mode = bool(transform_to_probit_space and is_log_density)

    # log(n) conditioning (sample size). This enables variable-n training without
    # returning variable-length `samples` tensors from the dataset.
    log_n = batch.get('log_n')
    if isinstance(log_n, torch.Tensor):
        log_n = log_n.to(device=device, dtype=dtype).reshape(-1)
        if log_n.numel() == 1:
            log_n = log_n.expand(B)
        elif log_n.numel() != B:
            raise ValueError(f"batch['log_n'] has {log_n.numel()} elements but batch size is {B}")
    elif log_n is not None:
        log_n = torch.full((B,), float(log_n), device=device, dtype=dtype)
    else:
        if samples is not None:
            log_n = torch.full((B,), float(np.log(samples.shape[1])), device=device, dtype=dtype)
        else:
            # Default: log(1000) ≈ 6.9
            log_n = torch.full((B,), 6.9, device=device, dtype=dtype)
    geometry_local = geometry
    if geometry_local is not None:
        area = geometry_local['area'].to(device=device, dtype=dtype)
        row_widths_bc = geometry_local['row_widths_bc'].to(device=device, dtype=dtype)
        col_widths_bc = geometry_local['col_widths_bc'].to(device=device, dtype=dtype)
        row_widths_vec = geometry_local['row_widths'].to(device=device, dtype=dtype)
        col_widths_vec = geometry_local['col_widths'].to(device=device, dtype=dtype)
        row_coords_geom = geometry_local['row_coords'].to(device=device, dtype=dtype)
        col_coords_geom = geometry_local['col_coords'].to(device=device, dtype=dtype)
        nonuniform = geometry_local.get('mode', 'uniform') != 'uniform'
    else:
        width = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
        row_widths_vec = col_widths_vec = width
        row_widths_bc = width.view(1, 1, m, 1)
        col_widths_bc = width.view(1, 1, 1, m)
        area = row_widths_bc * col_widths_bc
        row_coords_geom = torch.linspace(0.5/m, 1-0.5/m, m, device=device, dtype=dtype)
        col_coords_geom = row_coords_geom
        nonuniform = False
        # Ensure downstream loss code always gets a geometry dict (tests often pass None).
        geometry_local = {
            'area': area,
            'row_widths_bc': row_widths_bc,
            'col_widths_bc': col_widths_bc,
            'row_widths': row_widths_vec,
            'col_widths': col_widths_vec,
            'row_coords': row_coords_geom,
            'col_coords': col_coords_geom,
            'mode': 'uniform',
        }

    def normalize_grid(grid: torch.Tensor) -> torch.Tensor:
        if grid is None:
            return None
        grid = _sanitize_density(grid)
        mass = (grid * area).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        grid = grid / mass
        return _sanitize_density(grid)
    # ------------------------------------------------------------------
    # Targets
    #
    # - `target_log_model`: log-density in the model's training space
    #    - copula-space (default): log c(u,v)
    #    - probit-space (if transform_to_probit_space=True): log f_Z(z_u,z_v)
    # - `density` / `density_log`: copula-space density target for losses/projection: c(u,v)
    # ------------------------------------------------------------------
    target_log_model = density_raw.clamp(min=-20, max=20) if is_log_density else torch.log(_sanitize_density(density_raw).clamp_min(1e-12))

    if probit_mode:
        # Convert probit joint log-density back to copula log-density: log c = log f_Z - log φ(z_u) - log φ(z_v)
        target_log_copula = probit_logdensity_to_copula_logdensity(target_log_model, m)
        density = torch.exp(target_log_copula.clamp(min=-20, max=20)).clamp(min=1e-12, max=1e6)
    elif is_log_density:
        # Log-density already corresponds to copula space
        density = torch.exp(target_log_model).clamp(min=1e-12, max=1e6)
    else:
        density = _sanitize_density(density_raw)
    density = normalize_grid(density)
    density_log = torch.log(density.clamp_min(1e-12))

    training = config['training']
    loss_weights = training['loss_weights'].copy()
    tail_tau = training.get('tail_tau', 0.15)
    tail_mask = geometry_local.get('tail_mask') if geometry_local is not None else None
    if tail_mask is None or tail_mask.device != device:
        tail_mask = _build_tail_mask(m, tail_tau, device, row_coords=row_coords_geom, col_coords=col_coords_geom)
        if geometry_local is not None:
            geometry_local['tail_mask'] = tail_mask

    curriculum_weight = _get_curriculum_weight(training, step)
    for key in ('ce', 'ise', 'tail', 'ms', 'marg_kl'):
        loss_weights.setdefault(key, 0.0)
    tail_weight_dynamic = _get_tail_weight(training, step, loss_weights.get('tail', 0.0))
    loss_weights['tail'] = tail_weight_dynamic
    loss_weights_eff = {k: v * curriculum_weight for k, v in loss_weights.items()}
    # Base projection iterations (may be dynamically ramped/adapted)
    projection_iters = training.get('projection_iters',10)
    # Optional linear ramp for projection iterations to avoid early over-constraint
    proj_ramp_cfg = training.get('projection_ramp', {})
    if proj_ramp_cfg.get('enable', False):
        ramp_steps = proj_ramp_cfg.get('ramp_steps', 5000)
        max_iters = proj_ramp_cfg.get('max_iters', projection_iters)
        min_iters = proj_ramp_cfg.get('min_iters', 2)
        frac = min(1.0, step / max(1, ramp_steps))
        projection_iters = int(min_iters + frac * (max_iters - min_iters))
    use_coords = config['model'].get('use_coordinates', False)
    use_probit_coords = config['model'].get('use_probit_coords', False)  # Coordinate transformation
    time_cond = config['model'].get('time_conditioning', False) or (model_type == 'denoiser')
    use_amp = training.get('use_amp', True)
    # Optionally recompute target via anti-aliased histogram
    if training.get('use_antialiased_hist', False) and samples is not None:
        sigma = training.get('hist_sigma', 0.75)
        density = anti_aliased_hist(samples, m, sigma=sigma, normalize=True)
        density = normalize_grid(_sanitize_density(density))
        is_log_density = False
        # Keep diffusion's target in sync with the overridden density (copula space).
        target_log_model = torch.log(density.clamp_min(1e-12))
        density_log = target_log_model
        probit_mode = False

    # Construct conditioning input (histogram observation).
    # Prefer dataset-provided `hist` so we can support variable-n training without
    # having to return variable-length raw sample tensors.
    if hist is not None:
        input_hist = hist
        if input_hist.dim() == 3:
            input_hist = input_hist.unsqueeze(1)
        input_hist = normalize_grid(_sanitize_density(input_hist))
    elif samples is not None:
        input_hist_sigma = training.get('input_hist_sigma', training.get('hist_sigma', 0.0))
        if input_hist_sigma is None:
            input_hist_sigma = 0.0
        input_hist = anti_aliased_hist(samples, m, sigma=input_hist_sigma, normalize=True)
    else:
        # Fallback: if neither hist nor samples are available, use the (copula-space) target density.
        input_hist = density.clone()

    input_hist = input_hist.to(device)
    # Optional diffusion-style corruption of the conditioning histogram (single-pass denoiser training).
    # This is ONLY meaningful when the model consumes `x_in` (denoiser/enhanced_cnn); for diffusion_unet
    # it is used only as an optional conditioning channel.
    hist_noise_cfg = training.get('hist_noise', {})
    # We'll build t_scalar early if needed for noise conditioning.
    t_scalar = torch.rand(B, device=device) if time_cond else None
    if hist_noise_cfg.get('enable', False):
        # Mix towards the uniform density as noise level increases.
        max_strength = float(hist_noise_cfg.get('max_strength', 0.75))
        power = float(hist_noise_cfg.get('power', 1.0))
        if t_scalar is None:
            t_scalar = torch.rand(B, device=device)
        strength = (t_scalar.clamp(0, 1) ** power) * max_strength
        strength = strength.view(B, 1, 1, 1)
        uniform = torch.ones_like(input_hist)
        uniform = normalize_grid(uniform)
        input_hist = (1.0 - strength) * input_hist + strength * uniform
        input_hist = normalize_grid(input_hist)
    input_noise_std = training.get('input_noise_std', 0.0)
    if input_noise_std:
        input_hist = (input_hist + torch.randn_like(input_hist) * input_noise_std).clamp_min(0.0)
    input_hist = normalize_grid(input_hist)

    x_in = input_hist
    use_log_n_channel = bool(config['model'].get('use_log_n', False))
    if model_type in ['enhanced_cnn', 'denoiser'] and use_log_n_channel:
        ln_chan = log_n.view(B, 1, 1, 1).expand(B, 1, m, m)
        x_in = torch.cat([x_in, ln_chan], dim=1)
    if model_type in ['enhanced_cnn','denoiser'] and use_coords:
        coord_eps = config['model'].get('probit_coord_eps', 1e-4)
        # Adaptive probit coordinate epsilon to soften extreme tails based on sample size (rank-style smoothing)
        if config['model'].get('adaptive_probit_coords', False):
            if samples is not None:
                coord_eps = max(coord_eps, 1.0/(samples.shape[1] + 1))  # ensures clamp grows if sample count small
            else:
                # Variable-n training does not return samples; use log_n as proxy.
                try:
                    n_min = float(torch.exp(log_n.detach()).min().item())
                    coord_eps = max(coord_eps, 1.0 / (n_min + 1.0))
                except Exception:
                    pass
        coords = build_coordinates(B, m, device, probit=use_probit_coords, eps=coord_eps)
        x_in = torch.cat([x_in, coords], dim=1)

    optimizer = training_step.optimizer  # attached dynamically
    scaler_obj = scaler
    optimizer.zero_grad()
    t_forward_start = time.time()
    
    metrics = {}  # Initialize metrics dict

    def _coerce_float(value, name, default=None):
        """Best-effort cast of config values to float, with optional fallback."""
        if value is None:
            return None if default is None else default
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            if step == 0:
                print(f"[WARN] Non-numeric {name}={value!r}; using {default!r}")
                sys.stdout.flush()
            return default

    with autocast(device_type='cuda', enabled=use_amp):
        if model_type == 'diffusion_unet':
            t = torch.randint(0, diffusion.timesteps, (B,), device=device)
            
            aux_loss_weights_scalar = (1.0 - t.float() / diffusion.timesteps).clamp(min=0.0) ** 2

            # Target log-density in the model's training space (copula by default; probit if enabled).
            target_log = target_log_model
            
            real_noise = torch.randn_like(target_log)
            noisy = diffusion.q_sample(target_log, t, real_noise)
            # Optional histogram conditioning (CFG-style) if model expects 2 channels.
            log_n_model = log_n.to(device=device, dtype=noisy.dtype)
            if getattr(model, "conv_in").in_channels > 1:
                # Conditioning channel = log histogram density
                cond_hist = input_hist
                # Optional: smooth the conditioning histogram before taking log.
                # This reduces sensitivity to histogram speckle at high resolution (e.g., m=128),
                # and typically makes DDIM outputs less "patchy".
                cond_sigma = float(training.get("cond_hist_smooth_sigma", 0.0) or 0.0)
                if cond_sigma > 0.0:
                    from vdc.utils.smoothing import smooth_density_gaussian

                    # Optionally randomize sigma per batch for robustness.
                    if bool(training.get("cond_hist_smooth_random", True)):
                        # Use CPU RNG to avoid a device sync from `.item()` on a CUDA tensor.
                        cond_sigma_eff = cond_sigma * float(torch.rand(()).item())
                    else:
                        cond_sigma_eff = cond_sigma
                    cond_hist = smooth_density_gaussian(cond_hist.clamp_min(1e-12), sigma=float(cond_sigma_eff), preserve_mass=True)

                cond = torch.log(cond_hist.clamp(min=1e-12))
                if probit_mode:
                    # Convert log copula histogram to probit-space joint log-density for conditioning.
                    cond = copula_logdensity_to_probit_logdensity(cond, m)
                # Classifier-free dropout: randomly drop conditioning during training
                p_drop = float(training.get('cfg_dropout_prob', 0.1))
                if p_drop > 0:
                    drop_mask = (torch.rand(B, device=device) < p_drop).view(B, 1, 1, 1)
                    cond = torch.where(drop_mask, torch.zeros_like(cond), cond)
                model_in = torch.cat([noisy, cond], dim=1)
            else:
                model_in = noisy
            pred_noise = model(model_in, t.float()/diffusion.timesteps, log_n_model)
            
            # Check for NaN in predictions
            if torch.isnan(pred_noise).any():
                raise RuntimeError(f"NaN in pred_noise at step {step}, skipping batch")
            
            loss_noise = torch.mean((pred_noise - real_noise)**2)
            
            pred_noise = pred_noise.clamp(min=-10, max=10)
            alpha_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            recon_log = (noisy - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
            recon_log = recon_log.clamp(min=-20, max=20)
            recon_density = torch.exp(recon_log).clamp(min=1e-12, max=1e6)
            recon_density = normalize_grid(_sanitize_density(recon_density))

            if torch.isnan(recon_density).any() or torch.isinf(recon_density).any():
                raise RuntimeError(f"NaN/Inf in recon_density at step {step}, aborting batch")

            if transform_to_probit_space and is_log_density:
                recon_log_copula = probit_logdensity_to_copula_logdensity(recon_log, m)
                recon_density_copula = torch.exp(recon_log_copula.clamp(min=-20, max=20)).clamp(min=1e-12, max=1e6)
                recon_density_copula = normalize_grid(_sanitize_density(recon_density_copula))
            else:
                recon_density_copula = recon_density
            recon_density_copula = _sanitize_density(recon_density_copula)

            skip_projection = training.get('skip_projection', False) or projection_iters == 0
            if skip_projection:
                proj_copula = recon_density_copula
                proj_copula_display = recon_density_copula
            else:
                proj_iters_diffusion = projection_iters
                proj_copula_raw = copula_project(recon_density_copula, iters=proj_iters_diffusion)
                detach_projection = training.get('detach_projection', False)
                if detach_projection:
                    proj_copula_display = proj_copula_raw.detach()
                    proj_copula = recon_density_copula
                else:
                    proj_copula = proj_copula_raw
                    proj_copula_display = proj_copula_raw

            proj_copula = normalize_grid(proj_copula)
            proj_copula_display = normalize_grid(proj_copula_display)

            density_loss, density_metrics, density_components = compute_density_losses(
                proj_copula,
                density,
                training,
                geometry_local,
                loss_weights_eff,
                tail_mask,
                model_output=None,
                weight_factor=aux_loss_weights_scalar,
            )
            total = loss_noise + density_loss
            metrics.update(density_metrics)
            metrics['noise'] = float(loss_noise.item())
            metrics['curriculum_weight'] = float(curriculum_weight)
            metrics['tail_weight'] = float(tail_weight_dynamic)
            try:
                row_marg = (proj_copula * col_widths_bc).sum(-1).clamp_min(1e-12)
                col_marg = (proj_copula * row_widths_bc).sum(-2).clamp_min(1e-12)
                row_dev = row_marg - 1.0
                col_dev = col_marg - 1.0
                metrics['row_marg_max_abs'] = float(row_dev.abs().max().item())
                metrics['col_marg_max_abs'] = float(col_dev.abs().max().item())
                metrics['row_marg_rms'] = float(torch.sqrt((row_dev**2).mean()).item())
                metrics['col_marg_rms'] = float(torch.sqrt((col_dev**2).mean()).item())
                metrics['marg_kl_total'] = float(((row_marg * row_marg.log()).mean() + (col_marg * col_marg.log()).mean()).item())
            except Exception:
                pass
        else:
            out = model(x_in, t_scalar) if t_scalar is not None else model(x_in)
            if isinstance(out, dict):
                if 'density' in out:
                    pred_density = out['density']
                elif 'log_density' in out:
                    pred_density = torch.exp(out['log_density'].clamp(min=-15,max=15))
                elif 'residual' in out:
                    pred_density = torch.exp(out['residual'].clamp(min=-15,max=15))
                else:
                    raise ValueError('Unknown output dict keys')
            else:
                pred_density = out
            if not torch.isfinite(pred_density).all():
                bad_frac = float((~torch.isfinite(pred_density)).float().mean().item())
                raise RuntimeError(f"Non-finite pred_density at step {step} (frac={bad_frac:.6f})")
            pred_density = _sanitize_density(pred_density)
            # Independence shortcut: if empirical Kendall tau below threshold, skip expensive projection and use independence density
            indep_thresh = training.get('independence_tau_thresh', None)
            skip_projection = False
            if indep_thresh is not None and samples is not None:
                try:
                    tau_est = kendall_tau(samples)
                    if tau_est.abs() < indep_thresh:
                        # Replace pred_density with uniform (independence) and skip further losses except baseline ISE
                        pred_density = torch.ones_like(pred_density) * pred_density.mean()
                        skip_projection = True
                except Exception:
                    pass
            # Adaptive projection logic
            adaptive_cfg = training.get('adaptive_projection', {})
            if adaptive_cfg.get('enable', False):
                dev_pre = marginal_deviation(pred_density, row_widths=row_widths_vec, col_widths=col_widths_vec).item()
                target_tol = adaptive_cfg.get('marg_tol', 0.01)
                min_iters = adaptive_cfg.get('min_iters', 2)
                max_iters = adaptive_cfg.get('max_iters', projection_iters)
                if dev_pre <= target_tol:
                    projection_iters_step = min_iters
                else:
                    if math.isnan(dev_pre) or math.isinf(dev_pre):
                        print(f"[WARNING] NaN/Inf in dev_pre at step {step}, using min_iters for projection")
                        projection_iters_step = min_iters
                    else:
                        ratio = (dev_pre / target_tol)
                        projection_iters_step = int(min(max_iters, max(min_iters, round(min_iters * ratio))))
            else:
                projection_iters_step = projection_iters
            fast_cfg = adaptive_cfg
            fast_iters = fast_cfg.get('fast_iters', 3)
            tol = fast_cfg.get('fast_tol', 0.01)
            proj_start = time.time()
            if skip_projection:
                proj_copula = pred_density.detach()  # treat as already copula-like (uniform)
                proj_time = 0.0
            else:
                # If in probit log-space, convert to copula space for projection
                if transform_to_probit_space and is_log_density:
                    # Model output is exp(log_c_probit), need density in copula space
                    # First check if model outputs log or regular density
                    if isinstance(out, dict) and 'log_density' in out:
                        pred_log_probit = out['log_density'].clamp(min=-15, max=15)
                    else:
                        pred_log_probit = torch.log(pred_density.clamp(min=1e-10))
                    pred_log_copula = probit_logdensity_to_copula_logdensity(pred_log_probit, m)
                    pred_density_copula = torch.exp(pred_log_copula.clamp(min=-15, max=15))
                else:
                    pred_density_copula = pred_density
                pred_density_copula = _sanitize_density(pred_density_copula)
                
                # SOFT CONSTRAINTS: Skip projection if requested, use marginal KL loss instead
                skip_projection = training.get('skip_projection', False) or projection_iters_step == 0
                projection_method = training.get('projection_method', 'ipfp')  # 'ipfp', 'sinkhorn', or 'stabilized_ipfp'
                if nonuniform and projection_method != 'sinkhorn':
                    if step == 0:
                        print("[INFO] Non-uniform binning detected; forcing projection_method='sinkhorn'")
                    projection_method = 'sinkhorn'
                
                if skip_projection:
                    # No projection - rely on marginal KL regularization
                    proj_copula = pred_density_copula
                    proj_copula_display = pred_density_copula
                else:
                    # Select projection method
                    if projection_method == 'sinkhorn':
                        # Sinkhorn projection (smooth, differentiable)
                        quick = sinkhorn_project_density(
                            pred_density_copula,
                            iters=projection_iters_step if fast_iters<=0 else fast_iters,
                            row_target=row_widths_vec,
                            col_target=col_widths_vec,
                        )
                        dev_quick = marginal_deviation(quick, row_widths=row_widths_vec, col_widths=col_widths_vec)
                        if dev_quick > tol and projection_iters_step > fast_iters:
                            proj_copula = sinkhorn_project_density(
                                pred_density_copula,
                                iters=projection_iters_step,
                                row_target=row_widths_vec,
                                col_target=col_widths_vec,
                            )
                        else:
                            proj_copula = quick
                    elif projection_method == 'stabilized_ipfp':
                        # IPFP with gradient stabilization
                        quick = ipfp_project_log(pred_density_copula, iters=projection_iters_step if fast_iters<=0 else fast_iters, stabilize=True)
                        dev_quick = marginal_deviation(quick, row_widths=row_widths_vec, col_widths=col_widths_vec)
                        if dev_quick > tol and projection_iters_step > fast_iters:
                            proj_copula = ipfp_project_log(pred_density_copula, iters=projection_iters_step, stabilize=True)
                        else:
                            proj_copula = quick
                    else:  # 'ipfp' (original)
                        quick = ipfp_project_log(pred_density_copula, iters=projection_iters_step if fast_iters<=0 else fast_iters)
                        dev_quick = marginal_deviation(quick, row_widths=row_widths_vec, col_widths=col_widths_vec)
                        if dev_quick > tol and projection_iters_step > fast_iters:
                            proj_copula = ipfp_project_log(pred_density_copula, iters=projection_iters_step)
                        else:
                            proj_copula = quick
                    
                    # Optionally detach projection from gradient flow
                    detach_projection = training.get('detach_projection', False)
                    if detach_projection:
                        proj_copula_display = proj_copula.detach()
                        proj_copula = pred_density_copula
                    else:
                        proj_copula_display = proj_copula
                
                proj_time = time.time() - proj_start
            
            if not torch.isfinite(proj_copula).all():
                bad_frac = float((~torch.isfinite(proj_copula)).float().mean().item())
                raise RuntimeError(f"Non-finite proj_copula at step {step} (frac={bad_frac:.6f})")
            # Convert projected density for loss computation
            proj_copula = normalize_grid(proj_copula)
            proj_copula_display = normalize_grid(proj_copula_display)
            model_output = out if isinstance(out, dict) else None
            density_loss, density_metrics, density_components = compute_density_losses(
                proj_copula,
                density,
                training,
                geometry_local,
                loss_weights_eff,
                tail_mask,
                model_output=model_output,
                weight_factor=None,
            )
            uw_cfg = training.get('uncertainty_weighting', {})
            if uw_cfg.get('enable', False) and hasattr(training_step, 'loss_log_vars'):
                log_vars = training_step.loss_log_vars
                total = torch.zeros((), device=device)
                used = []
                for name in ('ce', 'ise', 'tail', 'ms', 'marg_kl'):
                    weight = loss_weights_eff.get(name, 0.0)
                    if weight <= 0:
                        continue
                    comp_loss = density_components.get(name)
                    if comp_loss is None:
                        continue
                    idx = len(used)
                    used.append(name)
                    if idx >= log_vars.shape[0]:
                        total = total + weight * comp_loss
                    else:
                        s = log_vars[idx]
                        total = total + weight * (torch.exp(-s) * comp_loss + s)
            else:
                total = density_loss
            metrics.update(density_metrics)
            metrics['curriculum_weight'] = float(curriculum_weight)
            metrics['tail_weight'] = float(tail_weight_dynamic)
            try:
                row_marg = (proj_copula * col_widths_bc).sum(-1).clamp_min(1e-12)
                col_marg = (proj_copula * row_widths_bc).sum(-2).clamp_min(1e-12)
                row_dev = row_marg - 1.0
                col_dev = col_marg - 1.0
                metrics['row_marg_max_abs'] = float(row_dev.abs().max().item())
                metrics['col_marg_max_abs'] = float(col_dev.abs().max().item())
                metrics['row_marg_rms'] = float(torch.sqrt((row_dev**2).mean()).item())
                metrics['col_marg_rms'] = float(torch.sqrt((col_dev**2).mean()).item())
                metrics['marg_kl_total'] = float(((row_marg * row_marg.log()).mean() + (col_marg * col_marg.log()).mean()).item())
            except Exception:
                pass

    forward_time = time.time() - t_forward_start
    # Backward
    grad_norm = None
    gradient_clip_val = _coerce_float(training.get('gradient_clip', 1.0), 'gradient_clip', default=1.0)
    if gradient_clip_val is None:
        gradient_clip_val = 1.0
    if use_amp:
        scaler_obj.scale(total).backward()
        scaler_obj.unscale_(optimizer)
        # --- Gradient sanitation (NaN/Inf/Exploding) BEFORE norm calc & clipping ---
        gs_cfg = training.get('grad_sanitation', {})
        max_grad_value = _coerce_float(gs_cfg.get('max_grad_value', None), 'grad_sanitation.max_grad_value')
        max_report_norm = _coerce_float(gs_cfg.get('max_report_norm', None), 'grad_sanitation.max_report_norm')
        if gs_cfg.get('enable', False):
            any_bad = False
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.data
                    # Replace NaN/Inf with zero to avoid poisoning optimizer state
                    nan_mask = torch.isnan(g)
                    inf_mask = torch.isinf(g)
                    if nan_mask.any() or inf_mask.any():
                        g[nan_mask | inf_mask] = 0.0
                        any_bad = True
                    # Clamp excessively large absolute values (helps tame outlier spikes)
                    if max_grad_value is not None:
                        g.clamp_(-max_grad_value, max_grad_value)
            if any_bad and gs_cfg.get('verbose', True) and step % 100 == 0:
                print(f"[GradSanitize] Replaced NaN/Inf grads with 0 at step {step}")
                sys.stdout.flush()
        # Gradient norm (pre-clipping) for diagnostics (after sanitation)
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)**2
                total_norm_sq += param_norm.item()
        grad_norm = math.sqrt(total_norm_sq)
        if max_report_norm is not None and grad_norm is not None and grad_norm > max_report_norm:
            grad_norm = max_report_norm
        # Gradient anomaly detection (before clipping, to catch true anomalies)
        grad_threshold = _coerce_float(training.get('grad_anomaly_threshold', None), 'grad_anomaly_threshold')
        grad_skipped = False
        if grad_threshold is not None and grad_norm is not None and (grad_norm > grad_threshold or math.isinf(grad_norm) or math.isnan(grad_norm)):
            grad_skipped = True
            optimizer.zero_grad(set_to_none=True)
            scaler_obj.update()  # CRITICAL: Must call update() to reset scaler state
            # Track consecutive skips & apply LR decay if patience exceeded
            if not hasattr(training_step, 'grad_skip_count'):
                training_step.grad_skip_count = 0
            training_step.grad_skip_count += 1
            patience = _coerce_float(gs_cfg.get('lr_skip_patience', 5), 'grad_sanitation.lr_skip_patience', default=5.0)
            patience = int(max(1, patience if patience is not None else 5.0))
            decay = _coerce_float(gs_cfg.get('lr_decay_factor', 0.5), 'grad_sanitation.lr_decay_factor', default=0.5)
            if decay is None or decay <= 0:
                decay = 0.5
            min_lr = _coerce_float(gs_cfg.get('min_lr', 1e-7), 'grad_sanitation.min_lr', default=1e-7)
            if min_lr is None:
                min_lr = 1e-7
            if gs_cfg.get('enable', False) and training_step.grad_skip_count >= patience:
                for pg in optimizer.param_groups:
                    old_lr = pg.get('lr', 0.0)
                    new_lr = max(min_lr, old_lr * decay)
                    pg['lr'] = new_lr
                if gs_cfg.get('verbose', True):
                    print(f"[GradSanitize] LR decayed after {training_step.grad_skip_count} consecutive skips -> factor {decay}")
                    sys.stdout.flush()
                training_step.grad_skip_count = 0
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            scaler_obj.step(optimizer)
            scaler_obj.update()
            # Reset skip counter on successful update
            if hasattr(training_step, 'grad_skip_count'):
                training_step.grad_skip_count = 0
    else:
        total.backward()
        gs_cfg = training.get('grad_sanitation', {})
        max_grad_value = _coerce_float(gs_cfg.get('max_grad_value', None), 'grad_sanitation.max_grad_value')
        max_report_norm = _coerce_float(gs_cfg.get('max_report_norm', None), 'grad_sanitation.max_report_norm')
        if gs_cfg.get('enable', False):
            any_bad = False
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.data
                    nan_mask = torch.isnan(g)
                    inf_mask = torch.isinf(g)
                    if nan_mask.any() or inf_mask.any():
                        g[nan_mask | inf_mask] = 0.0
                        any_bad = True
                    if max_grad_value is not None:
                        g.clamp_(-max_grad_value, max_grad_value)
            if any_bad and gs_cfg.get('verbose', True) and step % 100 == 0:
                print(f"[GradSanitize] Replaced NaN/Inf grads with 0 at step {step}")
                sys.stdout.flush()
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)**2
                total_norm_sq += param_norm.item()
        grad_norm = math.sqrt(total_norm_sq)
        if max_report_norm is not None and grad_norm is not None and grad_norm > max_report_norm:
            grad_norm = max_report_norm
        grad_threshold = _coerce_float(training.get('grad_anomaly_threshold', None), 'grad_anomaly_threshold')
        grad_skipped = False
        if grad_threshold is not None and grad_norm is not None and (grad_norm > grad_threshold or math.isinf(grad_norm) or math.isnan(grad_norm)):
            grad_skipped = True
            optimizer.zero_grad(set_to_none=True)
            if not hasattr(training_step, 'grad_skip_count'):
                training_step.grad_skip_count = 0
            training_step.grad_skip_count += 1
            patience = _coerce_float(gs_cfg.get('lr_skip_patience', 5), 'grad_sanitation.lr_skip_patience', default=5.0)
            patience = int(max(1, patience if patience is not None else 5.0))
            decay = _coerce_float(gs_cfg.get('lr_decay_factor', 0.5), 'grad_sanitation.lr_decay_factor', default=0.5)
            if decay is None or decay <= 0:
                decay = 0.5
            min_lr = _coerce_float(gs_cfg.get('min_lr', 1e-7), 'grad_sanitation.min_lr', default=1e-7)
            if min_lr is None:
                min_lr = 1e-7
            if gs_cfg.get('enable', False) and training_step.grad_skip_count >= patience:
                for pg in optimizer.param_groups:
                    old_lr = pg.get('lr', 0.0)
                    new_lr = max(min_lr, old_lr * decay)
                    pg['lr'] = new_lr
                if gs_cfg.get('verbose', True):
                    print(f"[GradSanitize] LR decayed after {training_step.grad_skip_count} consecutive skips -> factor {decay}")
                    sys.stdout.flush()
                training_step.grad_skip_count = 0
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()
            if hasattr(training_step, 'grad_skip_count'):
                training_step.grad_skip_count = 0

    metrics = {'loss': float(total.item())}
    if grad_norm is not None:
        metrics['grad_norm'] = float(grad_norm)
    if model_type == 'diffusion_unet':
        # Metrics already set in the diffusion branch above
        if 'noise' not in metrics:
            metrics['noise'] = float(loss_noise.item()) if 'loss_noise' in locals() else 0.0
        if 'ise' not in metrics:
            metrics['ise'] = float(loss_ise.item()) if 'loss_ise' in locals() else 0.0
        if 'tail' not in metrics:
            metrics['tail'] = float(loss_tail.item()) if 'loss_tail' in locals() else 0.0
        # Preserve curriculum weight if computed during warmup
        if training.get('curriculum', {}).get('enable', False) and 'curriculum_weight' not in metrics:
            curriculum = training.get('curriculum', {})
            warmup_steps = curriculum.get('warmup_steps', 5000)
            rampup_steps = curriculum.get('rampup_steps', 10000)
            if step < warmup_steps:
                curriculum_weight = 0.0
            elif step < warmup_steps + rampup_steps:
                curriculum_weight = (step - warmup_steps) / rampup_steps
            else:
                curriculum_weight = 1.0
            metrics['curriculum_weight'] = float(curriculum_weight)
    else:
        if training.get('adaptive_projection', {}).get('enable', False):
            metrics['proj_iters'] = projection_iters_step
        if training.get('use_antialiased_hist', False):
            metrics['hist_sigma'] = training.get('hist_sigma',0.75)
        # Marginal deviation metrics (uniformity diagnostics)
        try:
            dens_for_marg = proj_copula if 'proj_copula' in locals() else proj
            row_marg = (dens_for_marg * col_widths_bc).sum(-1).clamp_min(1e-12)
            col_marg = (dens_for_marg * row_widths_bc).sum(-2).clamp_min(1e-12)
            row_dev = row_marg - 1.0
            col_dev = col_marg - 1.0
            metrics['row_marg_max_abs'] = float(row_dev.abs().max().item())
            metrics['col_marg_max_abs'] = float(col_dev.abs().max().item())
            metrics['row_marg_rms'] = float(torch.sqrt((row_dev**2).mean()).item())
            metrics['col_marg_rms'] = float(torch.sqrt((col_dev**2).mean()).item())
            kl_row = (row_marg * row_marg.log()).mean()
            kl_col = (col_marg * col_marg.log()).mean()
            metrics['marg_kl_total'] = float((kl_row + kl_col).item())
        except Exception:
            pass
    if training.get('uncertainty_weighting', {}).get('enable', False) and hasattr(training_step, 'loss_log_vars'):
        for i in range(training_step.loss_log_vars.shape[0]):
            metrics[f'log_var_{i}'] = float(training_step.loss_log_vars[i].item())
    if 'grad_skipped' in locals() and grad_skipped:
        metrics['grad_skipped'] = 1.0
    # Report current LR & consecutive skip count for monitoring
    try:
        metrics['lr'] = float(optimizer.param_groups[0]['lr'])
    except Exception:
        pass
    if hasattr(training_step, 'grad_skip_count') and training_step.grad_skip_count > 0:
        metrics['grad_skip_count'] = float(training_step.grad_skip_count)
    if profiler is not None:
        if model_type != 'diffusion_unet':
            profiler['forward_times'].append(forward_time - (proj_time if 'proj_time' in locals() else 0))
            profiler['projection_times'].append(proj_time if 'proj_time' in locals() else 0.0)
        else:
            profiler['forward_times'].append(forward_time)
            profiler['projection_times'].append(0.0)
        profiler['loss_times'].append(0.0)
    return metrics


def visualize_copula_comparison(model, model_type, batch, device, config, diffusion, step, output_dir, rank=0):
    """Generate visualization comparing predicted copula vs ground truth."""
    if rank != 0:
        return  # Only rank 0 generates plots
    
    # Save the current training state
    was_training = model.training
    model.eval()
    with torch.no_grad():
        density = batch['density'].to(device)[:1]  # Take first sample only
        is_log_density = batch.get('is_log_density', False)
        if isinstance(is_log_density, torch.Tensor):
            is_log_density = bool(is_log_density.flatten()[0].item())
        B, _, m, _ = density.shape
        samples = batch.get('samples')
        if samples is not None:
            samples = samples.to(device)[:1]
        training_cfg = config.get('training', {})
        input_hist_sigma = training_cfg.get('input_hist_sigma', training_cfg.get('hist_sigma', 0.0))
        if input_hist_sigma is None:
            input_hist_sigma = 0.0
        if samples is not None:
            x_base = anti_aliased_hist(samples, m, sigma=input_hist_sigma, normalize=True)
        else:
            x_base = torch.exp(density.clamp(min=-15, max=15)) if is_log_density else density.clone()
        input_noise_std = training_cfg.get('input_noise_std', 0.0)
        if input_noise_std:
            x_base = (x_base + torch.randn_like(x_base) * input_noise_std).clamp_min(0.0)
        
        # Generate prediction
        
        if model_type == 'diffusion_unet':
            # For diffusion, sample from the model
            if is_log_density:
                target_log = density_log
            else:
                target_log = torch.log(density.clamp(min=1e-10))
            
            # Start from pure noise and denoise
            x_t = torch.randn_like(target_log)
            for t in reversed(range(0, diffusion.timesteps)):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                pred_noise = model(x_t, t_tensor.float()/diffusion.timesteps)
                alpha_t = diffusion.alphas_cumprod[t]
                alpha_t_prev = diffusion.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
                
                # DDPM sampling step
                beta_t = 1 - alpha_t / alpha_t_prev if t > 0 else 1 - alpha_t
                x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                
                if t > 0:
                    noise = torch.randn_like(x_t)
                    x_t = torch.sqrt(alpha_t_prev) * x_0_pred + torch.sqrt(1 - alpha_t_prev) * noise
                else:
                    x_t = x_0_pred
            
            if is_log_density:
                pred_density = torch.exp(x_t.clamp(min=-15, max=15))
            else:
                pred_density = x_t
        else:
            # Direct prediction models
            use_coords = config['model'].get('use_coordinates', False)
            use_probit_coords = config['model'].get('use_probit_coords', False)
            
            if model_type in ['enhanced_cnn','denoiser'] and use_coords:
                coords = build_coordinates(B, m, device, probit=use_probit_coords)
                x_in = torch.cat([x_base, coords], dim=1)
            else:
                x_in = x_base
            
            out = model(x_in)
            if isinstance(out, dict) and 'density' in out:
                pred_density = out['density']
            elif isinstance(out, dict) and 'log_density' in out:
                pred_density = torch.exp(out['log_density'].clamp(min=-15, max=15))
            else:
                pred_density = out
        
        # Convert to numpy for plotting
        if is_log_density:
            gt_density = torch.exp(density.clamp(min=-15, max=15))
        else:
            gt_density = density
        
        pred_np = pred_density[0, 0].cpu().numpy()
        gt_np = gt_density[0, 0].cpu().numpy()
        mse = np.mean((pred_np - gt_np)**2)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Ground truth
        im0 = axes[0].imshow(gt_np, origin='lower', cmap='viridis', aspect='auto')
        axes[0].set_title('Ground Truth Copula Density')
        axes[0].set_xlabel('u')
        axes[0].set_ylabel('v')
        plt.colorbar(im0, ax=axes[0])
        
        # Predicted
        im1 = axes[1].imshow(pred_np, origin='lower', cmap='viridis', aspect='auto')
        axes[1].set_title('Predicted Copula Density')
        axes[1].set_xlabel('u')
        axes[1].set_ylabel('v')
        plt.colorbar(im1, ax=axes[1])
        
        # Difference (error map)
        diff = np.abs(pred_np - gt_np)
        im2 = axes[2].imshow(diff, origin='lower', cmap='hot', aspect='auto')
        axes[2].set_title(f'Absolute Error (MSE={mse:.6f})')
        axes[2].set_xlabel('u')
        axes[2].set_ylabel('v')
        plt.colorbar(im2, ax=axes[2])
        
        plt.suptitle(f'Step {step}: Copula Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'copula_comparison_step_{step}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Restore original training state
    model.train(was_training)


def plot_training_history(loss_history, output_dir, rank=0):
    """Extended training history plots (losses, gradients, projection, marginals).

    Adds:
      - grad_norm curve
      - dynamic tail_weight (if present)
      - marginal deviation metrics (row/col max abs, RMS, KL)
      - projection iterations
      - log_vars (uncertainty weights) if enabled
      - zoomed recent loss panel
    """
    if rank != 0:
        return
    if not loss_history:
        return
    try:
        steps = np.array([e['step'] for e in loss_history])
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))

        # (0,0) Total loss
        axes[0,0].plot(steps, [e['loss'] for e in loss_history], 'b-', lw=1.3)
        axes[0,0].set_title('Total Loss')
        axes[0,0].set_xlabel('Step'); axes[0,0].set_ylabel('Loss'); axes[0,0].grid(alpha=0.3); axes[0,0].set_yscale('log')

        # (0,1) Component losses
        comp_keys = ['ce','ise','tail','ms','marg_kl','noise']
        for k in comp_keys:
            if k in loss_history[0]:
                axes[0,1].plot(steps, [e.get(k,0.0) for e in loss_history], label=k, lw=1.2)
        axes[0,1].set_title('Component Losses')
        axes[0,1].set_xlabel('Step'); axes[0,1].set_ylabel('Loss'); axes[0,1].grid(alpha=0.3); axes[0,1].set_yscale('log'); axes[0,1].legend(fontsize=8, ncol=2)

        # (0,2) Projection iterations
        if 'proj_iters' in loss_history[0]:
            axes[0,2].plot(steps, [e.get('proj_iters',0) for e in loss_history], 'g-', lw=1.2)
            axes[0,2].set_ylabel('Iters')
        if 'tail_weight' in loss_history[0]:
            tw = [e.get('tail_weight', np.nan) for e in loss_history]
            ax2 = axes[0,2].twinx()
            ax2.plot(steps, tw, 'r--', lw=1.0, alpha=0.7)
            ax2.set_ylabel('tail_weight', color='r')
        axes[0,2].set_title('Projection Iters / Tail Weight')
        axes[0,2].set_xlabel('Step'); axes[0,2].grid(alpha=0.3)

        # (1,0) Grad norm & skipped grads
        if 'grad_norm' in loss_history[0]:
            axes[1,0].plot(steps, [e.get('grad_norm',np.nan) for e in loss_history], 'm-', lw=1.1)
        if 'grad_skipped' in loss_history[0]:
            skipped_flags = [e.get('grad_skipped',0) for e in loss_history]
            axes[1,0].scatter(steps, np.array(skipped_flags)*np.nanmax([e.get('grad_norm',1) for e in loss_history]), c='red', s=12, label='grad_skipped')
        axes[1,0].set_title('Gradient Norm')
        axes[1,0].set_xlabel('Step'); axes[1,0].set_ylabel('||grad||'); axes[1,0].grid(alpha=0.3)
        if 'grad_skipped' in loss_history[0]:
            axes[1,0].legend(fontsize=8)

        # (1,1) Marginal deviations (max abs)
        if 'row_marg_max_abs' in loss_history[0]:
            axes[1,1].plot(steps, [e.get('row_marg_max_abs',np.nan) for e in loss_history], label='row_max_abs', lw=1.1)
        if 'col_marg_max_abs' in loss_history[0]:
            axes[1,1].plot(steps, [e.get('col_marg_max_abs',np.nan) for e in loss_history], label='col_max_abs', lw=1.1)
        axes[1,1].set_title('Marginal Max Abs Deviations')
        axes[1,1].set_xlabel('Step'); axes[1,1].set_ylabel('Deviation'); axes[1,1].grid(alpha=0.3)
        handles_11, labels_11 = axes[1,1].get_legend_handles_labels()
        if handles_11:
            axes[1,1].legend(handles_11, labels_11, fontsize=8)

        # (1,2) Marginal RMS & KL
        ax_k = None
        if 'row_marg_rms' in loss_history[0]:
            axes[1,2].plot(steps, [e.get('row_marg_rms',np.nan) for e in loss_history], label='row_rms', lw=1.1)
        if 'col_marg_rms' in loss_history[0]:
            axes[1,2].plot(steps, [e.get('col_marg_rms',np.nan) for e in loss_history], label='col_rms', lw=1.1)
        if 'marg_kl' in loss_history[0]:
            ax_k = axes[1,2].twinx()
            ax_k.plot(steps, [e.get('marg_kl',np.nan) for e in loss_history], 'r--', label='marg_kl', lw=1.0, alpha=0.7)
            ax_k.set_ylabel('marg_kl', color='r')
        axes[1,2].set_title('Marginal RMS / KL')
        axes[1,2].set_xlabel('Step'); axes[1,2].set_ylabel('RMS'); axes[1,2].grid(alpha=0.3)
        handles_12, labels_12 = axes[1,2].get_legend_handles_labels()
        if ax_k is not None:
            handles_k, labels_k = ax_k.get_legend_handles_labels()
            handles_12 += handles_k
            labels_12 += labels_k
        if handles_12:
            axes[1,2].legend(handles_12, labels_12, fontsize=8)

        # (2,0) Uncertainty log vars
        log_var_keys = [k for k in loss_history[0].keys() if k.startswith('log_var_')]
        if log_var_keys:
            for k in log_var_keys:
                axes[2,0].plot(steps, [e.get(k,np.nan) for e in loss_history], label=k, lw=1.1)
            axes[2,0].set_title('Uncertainty Log Vars')
            axes[2,0].set_xlabel('Step'); axes[2,0].set_ylabel('log_var'); axes[2,0].grid(alpha=0.3); axes[2,0].legend(fontsize=8, ncol=2)
        else:
            axes[2,0].text(0.5,0.5,'No uncertainty weighting',ha='center',va='center',transform=axes[2,0].transAxes)
            axes[2,0].set_xticks([]); axes[2,0].set_yticks([])

        # (2,1) Recent loss (last 20%)
        recent_idx = max(0, int(len(steps)*0.8))
        axes[2,1].plot(steps[recent_idx:], [e['loss'] for e in loss_history[recent_idx:]], 'r-', lw=1.2)
        axes[2,1].set_title('Recent Loss (Last 20%)')
        axes[2,1].set_xlabel('Step'); axes[2,1].set_ylabel('Loss'); axes[2,1].grid(alpha=0.3); axes[2,1].set_yscale('log')

        # (2,2) CE vs ISE (overlay for correlation) if both present
        if 'ce' in loss_history[0] and 'ise' in loss_history[0]:
            axes[2,2].plot(steps, [e.get('ce',np.nan) for e in loss_history], 'c-', lw=1.0, label='CE')
            axes[2,2].plot(steps, [e.get('ise',np.nan) for e in loss_history], 'k-', lw=1.0, label='ISE')
            axes[2,2].set_title('CE vs ISE'); axes[2,2].set_xlabel('Step'); axes[2,2].set_ylabel('Loss'); axes[2,2].grid(alpha=0.3); axes[2,2].set_yscale('log'); axes[2,2].legend(fontsize=8)
        else:
            axes[2,2].text(0.5,0.5,'CE/ISE not both present',ha='center',va='center',transform=axes[2,2].transAxes)
            axes[2,2].set_xticks([]); axes[2,2].set_yticks([])

        plt.tight_layout()
        out_path = os.path.join(output_dir, 'training_history_extended.png')
        plt.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close()
        print(f"Extended training history plot saved to: {out_path}")
    except Exception as e:
        print(f"Error in plot_training_history: {e}")
        import traceback; traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-type', required=True, choices=['baseline_cnn','enhanced_cnn','denoiser','diffusion_unet'])
    parser.add_argument('--use-antialiased-hist', action='store_true', help='Override config to enable anti-aliased histogram target construction from samples')
    parser.add_argument('--independence-tau-thresh', type=float, default=None, help='Override config independence tau threshold')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"Unified Trainer | model={args.model_type} world_size={world_size}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    m = config['data']['m']
    binning_mode = config['data'].get('binning', 'uniform')
    geometry = build_geometry(m, binning_mode, device=device, dtype=torch.float32)

    dataset = OnTheFlyCopulaDataset(
        n_samples_per_batch=config['data']['n_samples_per_copula'],
        m=m,
        families=config['data']['copula_families'],
        param_ranges=config['data'].get('param_ranges'),
        rotation_prob=float(config['data'].get('rotation_prob', 0.3)),
        mixture_prob=float(config['data'].get('mixture_prob', 0.0)),
        n_mixture_components=tuple(config['data'].get('n_mixture_components', (2, 3))),
        transform_to_probit_space=config['model'].get('transform_to_probit_space', False),
        seed=42 + rank
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if world_size>1 else None
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], sampler=sampler, num_workers=config['data'].get('num_workers',4), pin_memory=True, drop_last=True)

    model = build_model(args.model_type, config, device)
    if world_size>1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    diffusion = None
    if args.model_type == 'diffusion_unet':
        diff_cfg = config.get('diffusion', {'timesteps':1000,'noise_schedule':'linear'})
        diffusion = CopulaAwareDiffusion(timesteps=diff_cfg.get('timesteps',1000), beta_schedule=diff_cfg.get('noise_schedule','linear')).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training'].get('weight_decay',1e-4))
    scaler = GradScaler(enabled=config['training'].get('use_amp', True))
    training_step.optimizer = optimizer  # attach for access inside function
    # Optional uncertainty weighting parameter tensor (log variances)
    uw_cfg = config['training'].get('uncertainty_weighting', {})
    if uw_cfg.get('enable', False):
        # Reserve slots for ce, ise, tail, ms, marg_kl
        n_slots = 5
        training_step.loss_log_vars = torch.nn.Parameter(torch.zeros(n_slots, device=device))
        optimizer.add_param_group({'params': [training_step.loss_log_vars], 'lr': config['training'].get('learning_rate', 1e-3)})
    else:
        training_step.loss_log_vars = None

    max_steps = config['training']['max_steps']
    # Support both *_interval and *_every naming conventions
    log_interval = config['training'].get('log_interval', config['training'].get('log_every', 100))
    save_interval = config['training'].get('save_interval', config['training'].get('save_every', 10000))
    viz_interval = config['training'].get('viz_interval', config['training'].get('viz_every', 1000))
    checkpoint_dir = config.get('checkpoint_dir', f'checkpoints/{args.model_type}')
    viz_dir = os.path.join(checkpoint_dir, 'visualizations')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    step = 0
    epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        if rank == 0:
            print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', 0)
        epoch = checkpoint.get('epoch', 0)
        if rank == 0:
            print(f"Resumed from step {step}, epoch {epoch}")
    
    pbar = tqdm(total=max_steps, disable=rank!=0, desc='Train', initial=step)
    profiler = {'forward_times': [], 'projection_times': [], 'loss_times': []}
    loss_history = []  # Track loss history for plotting
    viz_batch = None  # Store a batch for consistent visualization

    # Config override from CLI
    if args.use_antialiased_hist:
        config.setdefault('training', {})['use_antialiased_hist'] = True
    if args.independence_tau_thresh is not None:
        config.setdefault('training', {})['independence_tau_thresh'] = args.independence_tau_thresh
    # Early stopping configuration
    es_cfg = config['training'].get('early_stopping', {})
    es_enable = es_cfg.get('enable', False)
    es_metric = es_cfg.get('metric', 'ce')  # 'loss' or component like 'ce'
    es_patience = es_cfg.get('patience', 5000)
    es_min_delta = es_cfg.get('min_delta', 0.0)
    es_best = float('inf')
    es_best_step = 0
    nonfinite_abort = bool(config['training'].get('abort_on_nonfinite', True))
    nonfinite_patience = int(max(1, config['training'].get('nonfinite_patience', 3)))
    nonfinite_count = 0
    force_stop_nonfinite = False
    # Distributed early stop flag
    stop_flag = torch.tensor(0, device=device, dtype=torch.int32)
    try:
        while step < max_steps:
            if sampler: sampler.set_epoch(epoch)
            loader_fetch_start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                now_ts = time.time()
                fetch_latency = now_ts - loader_fetch_start
                step_start_time = now_ts
                if rank == 0 and step == 0 and batch_idx == 0:
                    tensor_shapes = {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    print(f"[TRACE] First batch fetched in {fetch_latency:.2f}s | keys={list(batch.keys())} | tensor_shapes={tensor_shapes}")
                    sys.stdout.flush()
                loader_warn_cfg = config['training'].get('loader_warn_threshold', 120.0)
                try:
                    loader_warn_threshold = float(loader_warn_cfg)
                except (TypeError, ValueError):
                    if rank == 0 and step == 0 and batch_idx == 0:
                        print(f"[WARN] Invalid loader_warn_threshold '{loader_warn_cfg}', defaulting to 120.0s")
                        sys.stdout.flush()
                    loader_warn_threshold = 120.0
                if rank == 0 and fetch_latency > loader_warn_threshold:
                    print(f"[WARN] Data loader wait {fetch_latency:.1f}s exceeded threshold {loader_warn_threshold:.0f}s at global_step={step} epoch={epoch} batch_idx={batch_idx}")
                    sys.stdout.flush()
                if rank == 0 and step < 5:
                    print(f"[TRACE] Entering training step {step} (epoch {epoch}, batch {batch_idx}) | fetch_latency={fetch_latency:.2f}s")
                    sys.stdout.flush()
                if step >= max_steps: break
                # Check distributed stop before next iteration
                if es_enable and dist.is_initialized():
                    dist.broadcast(stop_flag, src=0)
                    if stop_flag.item() == 1:
                        break
                
                # Store first batch for consistent visualization
                if viz_batch is None and rank == 0:
                    viz_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Run training step with NaN handling
                try:
                    metrics = training_step(args.model_type, model.module if hasattr(model,'module') else model, batch, device, config, diffusion=diffusion, scaler=scaler, step=step, profiler=profiler, geometry=geometry)
                    metrics = _filter_metrics_dict(metrics)
                    nonfinite_count = 0
                except RuntimeError as e:
                    if 'NaN in pred_noise' in str(e):
                        if rank == 0:
                            print(f"[WARNING] {e} at step {step}, skipping batch")
                            sys.stdout.flush()
                        loader_fetch_start = time.time()
                        continue
                    if 'Non-finite pred_density' in str(e) or 'Non-finite proj_copula' in str(e):
                        nonfinite_count += 1
                        if rank == 0:
                            print(
                                f"[WARNING] {e}; consecutive non-finite batches={nonfinite_count}/{nonfinite_patience}"
                            )
                            sys.stdout.flush()
                        if nonfinite_abort and nonfinite_count >= nonfinite_patience:
                            force_stop_nonfinite = True
                            if rank == 0:
                                print(
                                    "[ERROR] Stopping training due to repeated non-finite outputs. "
                                    "Use saved checkpoints from before collapse."
                                )
                                sys.stdout.flush()
                            break
                        loader_fetch_start = time.time()
                        continue
                    else:
                        if rank == 0:
                            print(f"[ERROR] RuntimeError at step {step}: {e}")
                            print(traceback.format_exc())
                            sys.stdout.flush()
                        raise
                except Exception as e:
                    if rank == 0:
                        print(f"[WARNING] Exception at step {step}: {e}, skipping batch")
                        print(traceback.format_exc())
                        sys.stdout.flush()
                    loader_fetch_start = time.time()
                    continue
                step += 1
                step_compute_time = time.time() - step_start_time
                metrics['loader_latency'] = float(fetch_latency)
                metrics['step_time'] = float(step_compute_time)
                step_warn_cfg = config['training'].get('step_time_warn_threshold', 300.0)
                try:
                    step_warn_threshold = float(step_warn_cfg)
                except (TypeError, ValueError):
                    if rank == 0 and step <= 1:
                        print(f"[WARN] Invalid step_time_warn_threshold '{step_warn_cfg}', defaulting to 300.0s")
                        sys.stdout.flush()
                    step_warn_threshold = 300.0
                if rank == 0 and step_compute_time > step_warn_threshold:
                    print(f"[WARN] Training step took {step_compute_time:.1f}s (threshold {step_warn_threshold:.0f}s) at global_step={step} epoch={epoch} batch_idx={batch_idx}")
                    sys.stdout.flush()
                
                # Track loss history
                if rank == 0:
                    loss_entry = {'step': step, **metrics}
                    loss_entry = _filter_metrics_dict(loss_entry)
                    loss_history.append(loss_entry)
                
                if rank==0:
                    pbar.update(1)
                    if step % log_interval == 0 or step==1:
                        # append profiling stats
                        if profiler['forward_times']:
                            fwd_ms = 1000 * (sum(profiler['forward_times'])/len(profiler['forward_times']))
                            proj_ms = 1000 * (sum(profiler['projection_times'])/len(profiler['projection_times']))
                        else:
                            fwd_ms = proj_ms = 0.0
                        # Include selected marginal metrics and grad norm in postfix
                        extra_keys = ['row_marg_max_abs','col_marg_max_abs','row_marg_rms','col_marg_rms','grad_norm']
                        prof_subset = {k:metrics[k] for k in extra_keys if k in metrics}
                        prof_info = {**{k:f"{v:.4f}" for k,v in metrics.items()}, **{k:f"{v:.4f}" for k,v in prof_subset.items()}, 'fwd_ms': f"{fwd_ms:.1f}", 'proj_ms': f"{proj_ms:.1f}"}
                        pbar.set_postfix(prof_info)
                        profiler = {'forward_times': [], 'projection_times': [], 'loss_times': []}
                    
                    # Generate visualization every viz_interval steps
                    # DISABLED: Visualization takes >30 min with 1000 diffusion steps, causes NCCL timeout
                    # Run visualization offline after training completes
                    # if step % viz_interval == 0 and viz_batch is not None:
                    #     pbar.write(f"[step {step}] Generating visualization...")
                    #     visualize_copula_comparison(
                    #         model.module if hasattr(model,'module') else model,
                    #         args.model_type,
                    #         viz_batch,
                    #         device,
                    #         config,
                    #         diffusion,
                    #         step,
                    #         viz_dir,
                    #         rank=rank
                    #     )
                    #     # Synchronize all ranks after visualization
                    #     dist.barrier()
                    
                    if step % save_interval == 0:
                        save_model = model.module if hasattr(model,'module') else model
                        torch.save({'step':step,'epoch':epoch,'model_state_dict':save_model.state_dict(),'config':config}, os.path.join(checkpoint_dir,f'model_step_{step}.pt'))
                        pbar.write(f"[step {step}] checkpoint saved")
                    # Early stopping check (rank 0 computes, broadcasts)
                    if es_enable:
                        current_metric = metrics.get(es_metric, metrics['loss'])
                        # Improvement?
                        if current_metric + es_min_delta < es_best:
                            es_best = current_metric
                            es_best_step = step
                        elif step - es_best_step >= es_patience:
                            if rank == 0:
                                print(f"Early stopping triggered at step {step} (best {es_metric}={es_best:.4f} at step {es_best_step})")
                                stop_flag.fill_(1)
                            if dist.is_initialized():
                                dist.broadcast(stop_flag, src=0)
                            else:
                                break
                        if stop_flag.item() == 1:
                            break
                loader_fetch_start = time.time()
            epoch += 1
            if es_enable and stop_flag.item() == 1:
                break
            if force_stop_nonfinite:
                break
    finally:
        if rank==0:
            pbar.close()
            if es_enable and stop_flag.item() == 1:
                print(f"Training finished early at step {step} (best {es_metric}={es_best:.4f} at step {es_best_step})")
            elif force_stop_nonfinite:
                print(f"Training stopped early at step {step} due to repeated non-finite outputs.")
            else:
                print(f"Training finished at step {step}")
            
            # Final visualization - DISABLED (too slow, run offline)
            # if viz_batch is not None:
            #     print("Generating final visualization...")
            #     visualize_copula_comparison(
            #         model.module if hasattr(model,'module') else model,
            #         args.model_type,
            #         viz_batch,
            #         device,
            #         config,
            #         diffusion,
            #         step,
            #         viz_dir,
            #         rank=rank
            #     )
            
            # Plot training history (only on rank 0)
            if loss_history and rank == 0:
                print("Plotting training history...")
                try:
                    plot_training_history(loss_history, viz_dir, rank=rank)
                    
                    # Save loss history as JSON for later analysis
                    import json
                    with open(os.path.join(viz_dir, 'loss_history.json'), 'w') as f:
                        json.dump(loss_history, f, indent=2)
                    # Also save CSV for quick inspection
                    import csv
                    csv_path = os.path.join(viz_dir, 'loss_history.csv')
                    keys = sorted({k for d in loss_history for k in d.keys()})
                    with open(csv_path, 'w', newline='') as cf:
                        writer = csv.DictWriter(cf, fieldnames=keys)
                        writer.writeheader()
                        for row in loss_history:
                            writer.writerow(row)
                    print(f"Loss history CSV saved to: {csv_path}")
                    print("Training history plots saved successfully.")
                except Exception as e:
                    print(f"Warning: Failed to plot training history: {e}")
            
        cleanup_distributed()


if __name__ == '__main__':
    main()
