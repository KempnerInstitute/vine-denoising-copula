"""
Unified training engine for copula density models.

Supports:
  - baseline_cnn      (CopulaDensityCNN)
  - enhanced_cnn      (EnhancedCopulaDensityCNN)
  - denoiser          (CopulaDenoiser)
  - diffusion_unet    (GridUNet + CopulaAwareDiffusion)

This module contains the core training logic, decoupled from CLI parsing.
For the command-line interface, see scripts/train_unified.py.
"""
import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure parent directory is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vdc.data.onthefly import OnTheFlyCopulaDataset
from vdc.models.copula_cnn import CopulaDensityCNN
from vdc.models.copula_cnn_enhanced import EnhancedCopulaDensityCNN
from vdc.models.copula_denoiser import CopulaDenoiser
from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project, sinkhorn_project_density
from vdc.losses import tail_weighted_loss
from vdc.utils.ipfp_log import ipfp_project_log, marginal_deviation
from vdc.utils.histogram import anti_aliased_hist
from vdc.utils.metrics import kendall_tau, tail_dependence_from_grid
from vdc.utils.probit_transform import (
    copula_density_to_probit_density,
    probit_density_to_copula_density,
    copula_logdensity_to_probit_logdensity,
    probit_logdensity_to_copula_logdensity
)


# ============================================================================
# Distributed setup
# ============================================================================
def setup_distributed():
    """Robust distributed setup supporting torchrun and SLURM env vars."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        import datetime
        timeout = datetime.timedelta(seconds=1800)  # 30 minutes
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timeout
        )
        dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Geometry construction
# ============================================================================
def build_binning(m: int, mode: str = 'uniform') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 1D bin edges, centers, and widths.
    
    Returns:
        edges, centers, widths (all numpy arrays of appropriate shapes)
    """
    from scipy.stats import norm
    
    if mode == 'uniform':
        edges = np.linspace(0, 1, m + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
    elif mode == 'probit':
        # Probit transform: more bins near 0 and 1
        quantiles = np.linspace(0, 1, m + 1)
        # Map through normal CDF inverse (probit)
        edges_probit = norm.ppf(np.clip(quantiles, 1e-6, 1 - 1e-6))
        # Bin edges in copula space [0,1]
        edges = np.clip(quantiles, 0, 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
    else:
        raise ValueError(f"Unknown binning mode: {mode}")
    
    return edges, centers, widths


def build_geometry(m: int, binning_mode: str, device: torch.device, dtype=torch.float32) -> Dict[str, torch.Tensor]:
    """
    Build geometric tensors for grid-based losses.
    
    Returns a dict with:
        - row_coords, col_coords: 1D tensors of bin centers
        - row_widths, col_widths: 1D tensors of bin widths
        - row_widths_bc, col_widths_bc: broadcasted for (B,1,m,1) and (B,1,1,m)
        - area: (1,1,m,m) tensor of per-cell areas
        - mode: the binning mode string
    """
    _, row_coords_np, row_widths_np = build_binning(m, mode=binning_mode)
    _, col_coords_np, col_widths_np = build_binning(m, mode=binning_mode)
    
    row_coords = torch.from_numpy(row_coords_np).to(device=device, dtype=dtype)
    col_coords = torch.from_numpy(col_coords_np).to(device=device, dtype=dtype)
    row_widths = torch.from_numpy(row_widths_np).to(device=device, dtype=dtype)
    col_widths = torch.from_numpy(col_widths_np).to(device=device, dtype=dtype)
    
    # Broadcasted versions for marginal computation
    row_widths_bc = row_widths.view(1, 1, m, 1)
    col_widths_bc = col_widths.view(1, 1, 1, m)
    
    # Cell area = Δu_i * Δv_j
    area = row_widths_bc * col_widths_bc  # (1,1,m,m)
    
    return {
        'row_coords': row_coords,
        'col_coords': col_coords,
        'row_widths': row_widths,
        'col_widths': col_widths,
        'row_widths_bc': row_widths_bc,
        'col_widths_bc': col_widths_bc,
        'area': area,
        'mode': binning_mode,
    }


# ============================================================================
# Model construction
# ============================================================================
def build_model(model_type: str, config: Dict, device: torch.device):
    m = config['data']['m']
    mcfg = config.get('model', {})
    
    if model_type == 'baseline_cnn':
        return CopulaDensityCNN(
            m=m,
            base_channels=mcfg.get('base_channels', 128),
            n_blocks=mcfg.get('n_blocks', 3),
            dropout=mcfg.get('dropout', 0.1)
        ).to(device)
    
    if model_type == 'enhanced_cnn':
        return EnhancedCopulaDensityCNN(
            m=m,
            base_channels=mcfg.get('base_channels', 128),
            n_blocks=mcfg.get('n_blocks', 3),
            dropout=mcfg.get('dropout', 0.1),
            input_channels=1 + (2 if mcfg.get('use_coordinates', True) else 0),
            output_mode=mcfg.get('output_mode', 'log'),
            time_conditioning=mcfg.get('time_conditioning', False),
            time_emb_dim=mcfg.get('time_emb_dim', 256),
            multi_scale_aux=mcfg.get('multi_scale_aux', False),
            aux_scales=tuple(mcfg.get('aux_scales', [2, 4]))
        ).to(device)
    
    if model_type == 'denoiser':
        return CopulaDenoiser(
            m=m,
            base_channels=mcfg.get('base_channels', 128),
            n_blocks=mcfg.get('n_blocks', 4),
            dropout=mcfg.get('dropout', 0.1),
            time_emb_dim=mcfg.get('time_emb_dim', 256)
        ).to(device)
    
    if model_type == 'diffusion_unet':
        return GridUNet(
            m=m,
            in_channels=mcfg.get('in_channels', 1),
            base_channels=mcfg.get('base_channels', 128),
            channel_mults=tuple(mcfg.get('channel_mults', [1, 2, 4])),
            num_res_blocks=mcfg.get('num_res_blocks', 2),
            attention_resolutions=tuple(mcfg.get('attention_resolutions', [])),
            dropout=mcfg.get('dropout', 0.1),
            time_emb_dim=mcfg.get('time_emb_dim', 256),
            use_coordinates=mcfg.get('use_coordinates', True)
        ).to(device)
    
    raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# Loss computation
# ============================================================================
def _sanitize_density(density: torch.Tensor, min_val=1e-12, max_val=1e6) -> torch.Tensor:
    """Clamp density to valid range and replace NaN/inf."""
    density = torch.where(torch.isfinite(density), density, torch.zeros_like(density))
    return density.clamp(min=min_val, max=max_val)


def _make_weight_factor(weight_factor, B: int, device):
    """Convert various weight_factor formats to a per-sample weight tensor."""
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
    """
    Compute density-based loss components.
    
    Returns:
        total_loss, metrics_dict, component_losses_dict
    """
    device = pred_density.device
    B = pred_density.shape[0]
    wf = _make_weight_factor(weight_factor, B, device) if weight_factor is not None else None
    
    def apply_weight(per_sample: torch.Tensor) -> torch.Tensor:
        if wf is None:
            return per_sample.mean()
        return (per_sample * wf).mean()
    
    metrics = {}
    total_loss = pred_density.new_zeros(())
    area = geometry['area']
    row_widths_bc = geometry['row_widths_bc']
    col_widths_bc = geometry['col_widths_bc']
    component_losses: Dict[str, torch.Tensor] = {}
    
    # ISE (Integrated Squared Error)
    use_log_ise = training_cfg.get('use_log_ise', False)
    if use_log_ise:
        pred_log = torch.log(pred_density.clamp(min=1e-12))
        target_log = torch.log(target_density.clamp(min=1e-12))
        diff = (pred_log - target_log) ** 2
    else:
        diff = (pred_density - target_density) ** 2
    
    # Correct integration: ISE = ∫∫ (p-q)^2 du dv ≈ Σ (p_ij - q_ij)^2 Δu_i Δv_j
    ise_tensor = (diff * area).view(B, -1).sum(-1)
    metrics['ise'] = float(ise_tensor.mean().item())
    component_losses['ise'] = apply_weight(ise_tensor)
    if loss_weights.get('ise', 0.0) > 0.0:
        total_loss = total_loss + loss_weights['ise'] * component_losses['ise']
    
    # Cross-entropy
    T_mass = (target_density * area).clamp_min(1e-12)
    D_mass = (pred_density * area).clamp_min(1e-12)
    ce_tensor = -(T_mass * D_mass.log()).sum(dim=(-2, -1)).view(B)
    metrics['ce'] = float(ce_tensor.mean().item())
    component_losses['ce'] = apply_weight(ce_tensor)
    if loss_weights.get('ce', 0.0) > 0.0:
        total_loss = total_loss + loss_weights['ce'] * component_losses['ce']
    
    # Tail loss
    if tail_mask is not None:
        log_ratio = torch.log(D_mass) - torch.log(T_mass)
        tail_tensor = (tail_mask * (log_ratio ** 2)).view(B, -1).mean(-1)
        metrics['tail'] = float(tail_tensor.mean().item())
        component_losses['tail'] = apply_weight(tail_tensor)
        if loss_weights.get('tail', 0.0) > 0.0:
            total_loss = total_loss + loss_weights['tail'] * component_losses['tail']
    else:
        metrics['tail'] = 0.0
        component_losses['tail'] = pred_density.new_zeros(())
    
    # Multi-scale loss (downsampled CE + ISE)
    ms_cfg = training_cfg.get('multi_scale', {'enable': True})
    if ms_cfg.get('enable', True) and model_output is not None and loss_weights.get('ms', 0.0) > 0.0:
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
            ise_s = ((D_ds - T_ds) ** 2).view(B, -1).mean(-1)
            loss_ms_per = loss_ms_per + ce_w * ce_s + ise_w * ise_s
        
        metrics['ms'] = float(loss_ms_per.mean().item())
        component_losses['ms'] = apply_weight(loss_ms_per)
        total_loss = total_loss + loss_weights['ms'] * component_losses['ms']
    else:
        metrics['ms'] = 0.0
        component_losses['ms'] = pred_density.new_zeros(())
    
    # Marginal KL divergence (negative for nearly uniform marginals)
    row_marg = (pred_density * col_widths_bc).sum(-1).clamp_min(1e-12)
    col_marg = (pred_density * row_widths_bc).sum(-2).clamp_min(1e-12)
    kl_row = (row_marg * row_marg.log()).sum(dim=-1)
    kl_col = (col_marg * col_marg.log()).sum(dim=-1)
    kl_per = kl_row + kl_col
    metrics['marg_kl'] = float(kl_per.mean().item())
    component_losses['marg_kl'] = apply_weight(kl_per)
    if loss_weights.get('marg_kl', 0.0) > 0.0:
        total_loss = total_loss + loss_weights['marg_kl'] * component_losses['marg_kl']
    
    return total_loss, metrics, component_losses


def _build_tail_mask(
    m: int,
    tau: float,
    device,
    row_coords: Optional[torch.Tensor] = None,
    col_coords: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a binary mask for tail regions (near boundaries)."""
    if row_coords is None:
        u = torch.linspace(0.5 / m, 1 - 0.5 / m, m, device=device)
    else:
        u = row_coords.to(device)
    if col_coords is None:
        v = torch.linspace(0.5 / m, 1 - 0.5 / m, m, device=device)
    else:
        v = col_coords.to(device)
    
    U, V = torch.meshgrid(u, v, indexing='ij')
    mask = ((U < tau) | (V < tau) | (U > 1 - tau) | (V > 1 - tau)).float()
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,m,m)


# ============================================================================
# Training step
# ============================================================================
def training_step(
    model_type: str,
    model,
    batch,
    device,
    config: Dict,
    diffusion=None,
    scaler=None,
    step: int = 0,
    profiler: Optional[Dict] = None,
    geometry: Optional[Dict] = None,
):
    """
    Single training step for all model types.
    
    Returns:
        loss (scalar), metrics (dict), model_output (dict or None)
    """
    density_raw = batch['density'].to(device)
    is_log_density = batch.get('is_log_density', False)
    if isinstance(is_log_density, torch.Tensor):
        is_log_density = bool(is_log_density.flatten()[0].item())
    elif not isinstance(is_log_density, bool):
        is_log_density = bool(is_log_density)
    
    samples = batch.get('samples')
    if samples is not None:
        samples = samples.to(device)
    
    B, _, m, _ = density_raw.shape
    dtype = density_raw.dtype
    
    # Build geometry if not provided
    if geometry is not None:
        area = geometry['area'].to(device=device, dtype=dtype)
        row_widths_bc = geometry['row_widths_bc'].to(device=device, dtype=dtype)
        col_widths_bc = geometry['col_widths_bc'].to(device=device, dtype=dtype)
        row_widths_vec = geometry['row_widths'].to(device=device, dtype=dtype)
        col_widths_vec = geometry['col_widths'].to(device=device, dtype=dtype)
        row_coords_geom = geometry['row_coords'].to(device=device, dtype=dtype)
        col_coords_geom = geometry['col_coords'].to(device=device, dtype=dtype)
        nonuniform = geometry.get('mode', 'uniform') != 'uniform'
    else:
        width = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
        row_widths_vec = col_widths_vec = width
        row_widths_bc = width.view(1, 1, m, 1)
        col_widths_bc = width.view(1, 1, 1, m)
        area = row_widths_bc * col_widths_bc
        row_coords_geom = torch.linspace(0.5 / m, 1 - 0.5 / m, m, device=device, dtype=dtype)
        col_coords_geom = row_coords_geom
        nonuniform = False
    
    def normalize_grid(grid: torch.Tensor) -> torch.Tensor:
        if grid is None:
            return None
        grid = _sanitize_density(grid)
        mass = (grid * area).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        grid = grid / mass
        return _sanitize_density(grid)
    
    # Prepare target density
    if is_log_density:
        density = torch.exp(density_raw).clamp(min=1e-12, max=1e6)
    else:
        density = _sanitize_density(density_raw)
    density = normalize_grid(density)
    density_log = torch.log(density.clamp(min=1e-12))
    
    training = config['training']
    loss_weights = training['loss_weights'].copy()
    tail_tau = training.get('tail_tau', 0.15)
    tail_mask = geometry.get('tail_mask') if geometry is not None else None
    if tail_mask is None or tail_mask.device != device:
        tail_mask = _build_tail_mask(m, tail_tau, device, row_coords=row_coords_geom, col_coords=col_coords_geom)
        if geometry is not None:
            geometry['tail_mask'] = tail_mask
    
    # Dynamic weights (curriculum, tail ramp)
    curriculum_weight = 1.0
    curriculum_cfg = training.get('curriculum', {})
    if curriculum_cfg.get('enable', False):
        warmup_steps = curriculum_cfg.get('warmup_steps', 1000)
        curriculum_weight = min(1.0, step / max(1, warmup_steps))
    
    for key in ('ce', 'ise', 'tail', 'ms', 'marg_kl'):
        loss_weights.setdefault(key, 0.0)
    
    # Tail weight ramp
    tail_curriculum_cfg = training.get('tail_curriculum', {})
    if tail_curriculum_cfg.get('enable', False):
        start_step = tail_curriculum_cfg.get('start_step', 0)
        end_step = tail_curriculum_cfg.get('end_step', 5000)
        start_weight = tail_curriculum_cfg.get('start_weight', 0.0)
        end_weight = tail_curriculum_cfg.get('end_weight', loss_weights.get('tail', 0.0))
        if step < start_step:
            tail_weight_dynamic = start_weight
        elif step >= end_step:
            tail_weight_dynamic = end_weight
        else:
            frac = (step - start_step) / (end_step - start_step)
            tail_weight_dynamic = start_weight + frac * (end_weight - start_weight)
        loss_weights['tail'] = tail_weight_dynamic
    
    loss_weights_eff = {k: v * curriculum_weight for k, v in loss_weights.items()}
    
    # Projection iterations (with optional ramp)
    projection_iters = training.get('projection_iters', 10)
    proj_ramp_cfg = training.get('projection_ramp', {})
    if proj_ramp_cfg.get('enable', False):
        ramp_steps = proj_ramp_cfg.get('ramp_steps', 5000)
        max_iters = proj_ramp_cfg.get('max_iters', projection_iters)
        min_iters = proj_ramp_cfg.get('min_iters', 2)
        frac = min(1.0, step / max(1, ramp_steps))
        projection_iters = int(min_iters + frac * (max_iters - min_iters))
    
    use_coords = config['model'].get('use_coordinates', False)
    use_amp = training.get('use_amp', True)
    detach_projection = training.get('detach_projection', True)
    
    # Forward pass
    model_output = None
    
    if model_type == 'diffusion_unet':
        # Diffusion model: denoise from noisy log-density
        t = torch.randint(0, diffusion.timesteps, (B,), device=device).long()
        noise = torch.randn_like(density_log)
        noisy_log_density = diffusion.q_sample(density_log, t, noise=noise)
        
        with autocast(device_type='cuda', enabled=use_amp):
            if use_coords:
                coords = torch.stack(torch.meshgrid(row_coords_geom, col_coords_geom, indexing='ij'), dim=0)
                coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
                model_input = torch.cat([noisy_log_density, coords], dim=1)
            else:
                model_input = noisy_log_density
            
            predicted_noise = model(model_input, t)
            
            # Predicted clean log-density
            pred_log_density = diffusion.predict_start_from_noise(noisy_log_density, t, predicted_noise)
            pred_density = torch.exp(pred_log_density).clamp(min=1e-12, max=1e6)
            pred_density = normalize_grid(pred_density)
            
            # Projection to copula constraints
            if projection_iters > 0:
                if detach_projection:
                    pred_density_proj = copula_project(
                        pred_density.detach(),
                        iters=projection_iters,
                        row_target=row_widths_vec,
                        col_target=col_widths_vec
                    )
                    pred_density = pred_density + (pred_density_proj - pred_density.detach())
                else:
                    pred_density = copula_project(
                        pred_density,
                        iters=projection_iters,
                        row_target=row_widths_vec,
                        col_target=col_widths_vec
                    )
                pred_density = normalize_grid(pred_density)
            
            # Loss on denoising task
            noise_loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            
            # Density-based losses
            density_loss, density_metrics, density_components = compute_density_losses(
                pred_density,
                density,
                training,
                geometry,
                loss_weights_eff,
                tail_mask,
                model_output=None,
                weight_factor=None
            )
            
            total_loss = noise_loss + density_loss
        
        metrics = {
            'loss': total_loss.item(),
            'noise_loss': noise_loss.item(),
            **density_metrics
        }
    
    else:
        # Direct density models (CNN, denoiser)
        with autocast(device_type='cuda', enabled=use_amp):
            if model_type == 'denoiser':
                t = torch.rand(B, device=device)
                model_output = model(density_log, t)
            else:
                if use_coords:
                    coords = torch.stack(torch.meshgrid(row_coords_geom, col_coords_geom, indexing='ij'), dim=0)
                    coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
                    model_input = torch.cat([density_log, coords], dim=1)
                else:
                    model_input = density_log
                model_output = model(model_input)
            
            pred_log_density = model_output
            pred_density = torch.exp(pred_log_density).clamp(min=1e-12, max=1e6)
            pred_density = normalize_grid(pred_density)
            
            if projection_iters > 0:
                if detach_projection:
                    pred_density_proj = copula_project(
                        pred_density.detach(),
                        iters=projection_iters,
                        row_target=row_widths_vec,
                        col_target=col_widths_vec
                    )
                    pred_density = pred_density + (pred_density_proj - pred_density.detach())
                else:
                    pred_density = copula_project(
                        pred_density,
                        iters=projection_iters,
                        row_target=row_widths_vec,
                        col_target=col_widths_vec
                    )
                pred_density = normalize_grid(pred_density)
            
            total_loss, metrics, components = compute_density_losses(
                pred_density,
                density,
                training,
                geometry,
                loss_weights_eff,
                tail_mask,
                model_output=None,
                weight_factor=None
            )
            
            metrics['loss'] = total_loss.item()
    
    return total_loss, metrics, model_output


# ============================================================================
# Main training loop
# ============================================================================
def train(
    model_type: str,
    config: Dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    resume_checkpoint: Optional[str] = None,
):
    """
    Main training loop.
    
    Args:
        model_type: One of 'baseline_cnn', 'enhanced_cnn', 'denoiser', 'diffusion_unet'
        config: Full training configuration dict
        rank: Process rank for distributed training
        world_size: Total number of processes
        local_rank: Local GPU index
        resume_checkpoint: Path to checkpoint to resume from
    """
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"Unified Trainer | model={model_type} world_size={world_size}")
    
    # Build geometry
    m = config['data']['m']
    binning_mode = config['data'].get('binning', 'uniform')
    geometry = build_geometry(m, binning_mode, device=device, dtype=torch.float32)
    
    # Dataset and dataloader
    dataset = OnTheFlyCopulaDataset(
        n_samples_per_batch=config['data']['n_samples_per_copula'],
        m=m,
        families=config['data']['copula_families'],
        transform_to_probit_space=config['model'].get('transform_to_probit_space', False),
        seed=42 + rank
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    # Model
    model = build_model(model_type, config, device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Diffusion (for diffusion_unet only)
    diffusion = None
    if model_type == 'diffusion_unet':
        diff_cfg = config.get('diffusion', {'timesteps': 1000, 'noise_schedule': 'linear'})
        diffusion = CopulaAwareDiffusion(
            timesteps=diff_cfg.get('timesteps', 1000),
            beta_schedule=diff_cfg.get('noise_schedule', 'linear')
        ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training'].get('use_amp', True))
    
    # Uncertainty weighting (optional learned loss weights)
    uw_cfg = config['training'].get('uncertainty_weighting', {})
    loss_log_vars = None
    if uw_cfg.get('enable', False):
        n_slots = 5  # ce, ise, tail, ms, marg_kl
        loss_log_vars = torch.nn.Parameter(torch.zeros(n_slots, device=device))
        optimizer.add_param_group({
            'params': [loss_log_vars],
            'lr': config['training'].get('learning_rate', 1e-3)
        })
    
    # Training config
    max_steps = config['training']['max_steps']
    log_interval = config['training'].get('log_interval', config['training'].get('log_every', 100))
    save_interval = config['training'].get('save_interval', config['training'].get('save_every', 10000))
    checkpoint_dir = config.get('checkpoint_dir', f'checkpoints/{model_type}')
    viz_dir = os.path.join(checkpoint_dir, 'visualizations')
    
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
    
    # Resume from checkpoint
    step = 0
    epoch = 0
    if resume_checkpoint:
        if rank == 0:
            print(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', 0)
        epoch = checkpoint.get('epoch', 0)
        if rank == 0:
            print(f"Resumed from step {step}, epoch {epoch}")
    
    # Early stopping
    es_cfg = config['training'].get('early_stopping', {})
    es_enable = es_cfg.get('enable', False)
    es_patience = es_cfg.get('patience', 2000)
    es_min_delta = es_cfg.get('min_delta', 0.0)
    es_metric = es_cfg.get('metric', 'loss')
    es_best = float('inf')
    es_best_step = 0
    stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
    
    # Training loop
    model.train()
    loss_history = []
    profiler = {'forward_times': [], 'projection_times': [], 'loss_times': []}
    
    pbar = tqdm(total=max_steps, desc="Training", disable=(rank != 0))
    
    try:
        while step < max_steps:
            if world_size > 1:
                sampler.set_epoch(epoch)
            
            for batch in dataloader:
                if step >= max_steps:
                    break
                
                optimizer.zero_grad()
                
                loss, metrics, model_output = training_step(
                    model_type,
                    model,
                    batch,
                    device,
                    config,
                    diffusion=diffusion,
                    scaler=scaler,
                    step=step,
                    profiler=profiler,
                    geometry=geometry
                )
                
                # Backward
                scaler.scale(loss).backward()
                
                # Gradient clipping
                max_grad_norm = config['training'].get('max_grad_norm', 1.0)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
                step += 1
                
                # Logging
                if step % log_interval == 0:
                    if rank == 0:
                        pbar.update(log_interval)
                        pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
                        loss_history.append({'step': step, **metrics})
                
                # Checkpointing
                if step % save_interval == 0 and rank == 0:
                    save_model = model.module if hasattr(model, 'module') else model
                    torch.save({
                        'step': step,
                        'epoch': epoch,
                        'model_state_dict': save_model.state_dict(),
                        'config': config
                    }, os.path.join(checkpoint_dir, f'model_step_{step}.pt'))
                    pbar.write(f"[step {step}] checkpoint saved")
                
                # Early stopping
                if es_enable:
                    current_metric = metrics.get(es_metric, metrics['loss'])
                    if current_metric + es_min_delta < es_best:
                        es_best = current_metric
                        es_best_step = step
                    elif step - es_best_step >= es_patience:
                        if rank == 0:
                            print(f"Early stopping triggered at step {step} (best {es_metric}={es_best:.4f} at step {es_best_step})")
                            stop_flag.fill_(1)
                        if dist.is_initialized():
                            dist.broadcast(stop_flag, src=0)
                        break
                    if stop_flag.item() == 1:
                        break
            
            epoch += 1
            if es_enable and stop_flag.item() == 1:
                break
    
    finally:
        if rank == 0:
            pbar.close()
            if es_enable and stop_flag.item() == 1:
                print(f"Training finished early at step {step} (best {es_metric}={es_best:.4f} at step {es_best_step})")
            else:
                print(f"Training finished at step {step}")
            
            # Save loss history
            if loss_history:
                import json
                import csv
                
                with open(os.path.join(viz_dir, 'loss_history.json'), 'w') as f:
                    json.dump(loss_history, f, indent=2)
                
                csv_path = os.path.join(viz_dir, 'loss_history.csv')
                keys = sorted({k for d in loss_history for k in d.keys()})
                with open(csv_path, 'w', newline='') as cf:
                    writer = csv.DictWriter(cf, fieldnames=keys)
                    writer.writeheader()
                    for row in loss_history:
                        writer.writerow(row)
                print(f"Loss history saved to: {csv_path}")
        
        cleanup_distributed()

