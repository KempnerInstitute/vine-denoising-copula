#!/usr/bin/env python
"""
Large-scale training script for diffusion copula networks.

Optimized for:
- Very large datasets (millions of copula pairs)
- Multi-GPU/multi-node training
- Efficient checkpointing and resumption
- Advanced monitoring and logging
- Memory-efficient data loading

Usage:
    # Single GPU
    python scripts/train_large_scale.py --config configs/train_large.yaml
    
    # Multi-GPU (single node)
    torchrun --nproc_per_node=4 scripts/train_large_scale.py --config configs/train_large.yaml
    
    # Multi-node (SLURM)
    sbatch scripts/slurm/train_large_ddp.sh
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vdc.models.unet_grid import GridUNet
from vdc.models.projection import copula_project
from vdc.losses import nll_points, ise_logdensity, marginal_penalty, tail_weighted_loss
from vdc.data.streaming import StreamingCopulaDataset, InfiniteStreamingDataset, collate_streaming_batch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be limited")


class LargeScaleTrainer:
    """
    Enhanced trainer for large-scale copula network training.
    
    Features:
    - Efficient memory usage with streaming data
    - Robust checkpointing with automatic resumption
    - Learning rate scheduling
    - Gradient accumulation for large effective batch sizes
    - Mixed precision training
    - Comprehensive logging
    """
    
    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        # Build model
        self.model = self._build_model()
        
        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.log_dir = Path(config.get('log_dir', 'logs'))
        
        if self.is_main:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Data loaders (built on-demand to avoid worker issues before resumption)
        self.train_loader = None
        self.val_loader = None
    
    def _build_model(self) -> nn.Module:
        """Build the model."""
        model_cfg = self.config['model']
        
        model = GridUNet(
            m=model_cfg['m'],
            in_channels=model_cfg.get('in_channels', 1),
            base_channels=model_cfg.get('base_channels', 64),
            num_res_blocks=model_cfg.get('num_res_blocks', 2),
            attention_resolutions=tuple(model_cfg.get('attention_resolutions', [16])),
            dropout=model_cfg.get('dropout', 0.0),
        )
        
        model = model.to(self.device)
        
        # Wrap with DDP for multi-GPU
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )
        
        n_params = sum(p.numel() for p in model.parameters())
        if self.is_main:
            print(f"Model built: {n_params:,} parameters")
        
        return model
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer."""
        opt_cfg = self.config['optimizer']
        
        if opt_cfg['type'] == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg['lr'],
                betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
                weight_decay=opt_cfg.get('weight_decay', 0.01),
                eps=opt_cfg.get('eps', 1e-8),
            )
        elif opt_cfg['type'] == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_cfg['lr'],
                betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
                eps=opt_cfg.get('eps', 1e-8),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg['type']}")
        
        return optimizer
    
    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler."""
        sched_cfg = self.config.get('scheduler', None)
        
        if sched_cfg is None:
            return None
        
        if sched_cfg['type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg['T_max'],
                eta_min=sched_cfg.get('eta_min', 0),
            )
        elif sched_cfg['type'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg['step_size'],
                gamma=sched_cfg['gamma'],
            )
        elif sched_cfg['type'] == 'warmup_cosine':
            from torch.optim.lr_scheduler import LambdaLR
            
            warmup_steps = sched_cfg['warmup_steps']
            total_steps = sched_cfg['total_steps']
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler type: {sched_cfg['type']}")
        
        return scheduler
    
    def _setup_logging(self):
        """Setup logging (W&B, tensorboard, etc.)."""
        self.use_wandb = self.config.get('use_wandb', False) and WANDB_AVAILABLE and self.is_main
        
        if self.use_wandb:
            wandb_cfg = self.config.get('wandb', {})
            
            wandb.init(
                project=wandb_cfg.get('project', 'vine-copula'),
                name=wandb_cfg.get('run_name', None),
                config=self.config,
                dir=str(self.log_dir),
                resume='allow',
                id=wandb_cfg.get('run_id', None),
            )
            
            # Watch model (log gradients, parameters)
            if wandb_cfg.get('watch_model', False):
                wandb.watch(self.model, log='all', log_freq=1000)
    
    def _build_dataloaders(self):
        """Build data loaders."""
        data_cfg = self.config['data']
        
        # Training dataset
        train_dataset = StreamingCopulaDataset(
            data_root=data_cfg['train_root'],
            m=self.config['model']['m'],
            split='train',
            train_frac=data_cfg.get('train_frac', 0.8),
            val_frac=data_cfg.get('val_frac', 0.1),
            seed=data_cfg.get('seed', 42),
            reflect=data_cfg.get('reflect', True),
            smooth_sigma=data_cfg.get('smooth_sigma', None),
            cache_size=data_cfg.get('cache_size', 100),
            augment=data_cfg.get('augment', True),
            shuffle=True,
            shuffle_buffer_size=data_cfg.get('shuffle_buffer_size', 1000),
        )
        
        # Wrap in infinite iterator for step-based training
        if data_cfg.get('infinite', True):
            train_dataset = InfiniteStreamingDataset(train_dataset)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_cfg['batch_size'],
            num_workers=data_cfg.get('num_workers', 4),
            pin_memory=True,
            collate_fn=collate_streaming_batch,
        )
        
        # Validation dataset
        if data_cfg.get('val_root'):
            val_dataset = StreamingCopulaDataset(
                data_root=data_cfg['val_root'],
                m=self.config['model']['m'],
                split='val',
                train_frac=data_cfg.get('train_frac', 0.8),
                val_frac=data_cfg.get('val_frac', 0.1),
                seed=data_cfg.get('seed', 42),
                reflect=data_cfg.get('reflect', True),
                smooth_sigma=data_cfg.get('smooth_sigma', None),
                cache_size=0,  # No cache for validation
                augment=False,
                shuffle=False,
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=data_cfg.get('val_batch_size', data_cfg['batch_size']),
                num_workers=data_cfg.get('num_workers', 4),
                pin_memory=True,
                collate_fn=collate_streaming_batch,
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # Move to device
        hist = batch['hist'].to(self.device, non_blocking=True)
        points = batch['points'].to(self.device, non_blocking=True)
        points_mask = batch['points_mask'].to(self.device, non_blocking=True)
        log_pdf_grid = batch['log_pdf_grid'].to(self.device, non_blocking=True)
        
        B, _, m, _ = hist.shape
        
        # Sample diffusion timestep
        t = torch.rand(B, 1, 1, 1, device=self.device)
        
        # Forward pass
        with autocast(enabled=self.use_amp):
            # Predict log-density
            logD_raw = self.model(hist, t)
            
            # Convert to density
            D_pos = torch.exp(logD_raw)
            
            # Project to valid copula
            projection_iters = self.config['training'].get('projection_iters', 20)
            D_hat = copula_project(D_pos, iters=projection_iters)
            
            # Compute losses
            loss_nll = nll_points(D_hat, points, points_mask)
            
            # ISE with teacher
            logD_hat = torch.log(D_hat + 1e-12)
            loss_ise = ise_logdensity(logD_hat, log_pdf_grid)
            
            # Regularization
            loss_marginal = marginal_penalty(D_hat)
            loss_tail = tail_weighted_loss(D_hat, points, points_mask)
            
            # Weighted combination
            loss_cfg = self.config['training']['loss_weights']
            loss = (
                loss_cfg.get('nll', 1.0) * loss_nll +
                loss_cfg.get('ise', 1.0) * loss_ise +
                loss_cfg.get('marginal', 0.1) * loss_marginal +
                loss_cfg.get('tail', 0.5) * loss_tail
            )
            
            # Scale for gradient accumulation
            grad_accum = self.config['training'].get('grad_accum_steps', 1)
            loss = loss / grad_accum
        
        # Backward
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Metrics
        metrics = {
            'loss': loss.item() * grad_accum,
            'loss_nll': loss_nll.item(),
            'loss_ise': loss_ise.item(),
            'loss_marginal': loss_marginal.item(),
            'loss_tail': loss_tail.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return metrics
    
    def optimizer_step(self):
        """Perform optimizer step."""
        # Gradient clipping
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.optimizer.zero_grad()
        
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        val_metrics = []
        num_val_batches = self.config['training'].get('num_val_batches', 100)
        
        for i, batch in enumerate(self.val_loader):
            if i >= num_val_batches:
                break
            
            hist = batch['hist'].to(self.device, non_blocking=True)
            points = batch['points'].to(self.device, non_blocking=True)
            points_mask = batch['points_mask'].to(self.device, non_blocking=True)
            log_pdf_grid = batch['log_pdf_grid'].to(self.device, non_blocking=True)
            
            B, _, m, _ = hist.shape
            t = torch.ones(B, 1, 1, 1, device=self.device) * 0.5
            
            with autocast(enabled=self.use_amp):
                logD_raw = self.model(hist, t)
                D_pos = torch.exp(logD_raw)
                D_hat = copula_project(D_pos)
                
                loss_nll = nll_points(D_hat, points, points_mask)
                
                logD_hat = torch.log(D_hat + 1e-12)
                loss_ise = ise_logdensity(logD_hat, log_pdf_grid)
                
                loss_marginal = marginal_penalty(D_hat)
            
            val_metrics.append({
                'val_nll': loss_nll.item(),
                'val_ise': loss_ise.item(),
                'val_marginal': loss_marginal.item(),
            })
        
        self.model.train()
        
        # Average across batches and GPUs
        avg_metrics = {}
        for key in val_metrics[0].keys():
            values = [m[key] for m in val_metrics]
            avg_metrics[key] = np.mean(values)
        
        # Synchronize across GPUs
        if self.world_size > 1:
            for key in avg_metrics:
                tensor = torch.tensor(avg_metrics[key], device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                avg_metrics[key] = tensor.item()
        
        return avg_metrics
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save checkpoint."""
        if not self.is_main:
            return
        
        # Unwrap DDP
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if self.is_main:
            print(f"[Step {self.global_step}] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load checkpoint."""
        if not path.exists():
            print(f"Checkpoint {path} not found, starting from scratch")
            return
        
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load state
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        # Build dataloaders
        self._build_dataloaders()
        
        # Training config
        train_cfg = self.config['training']
        max_steps = train_cfg['max_steps']
        grad_accum_steps = train_cfg.get('grad_accum_steps', 1)
        log_every = train_cfg.get('log_every', 100)
        val_every = train_cfg.get('val_every', 5000)
        save_every = train_cfg.get('save_every', 10000)
        
        # Progress bar
        if self.is_main:
            pbar = tqdm(total=max_steps, initial=self.global_step, desc='Training')
        
        self.model.train()
        
        # Training loop
        train_iter = iter(self.train_loader)
        
        while self.global_step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart iterator (shouldn't happen with infinite dataset)
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
                self.epoch += 1
            
            # Training step
            metrics = self.train_step(batch)
            
            # Optimizer step (with gradient accumulation)
            if (self.global_step + 1) % grad_accum_steps == 0:
                grad_norm = self.optimizer_step()
                metrics['grad_norm'] = grad_norm
            
            self.global_step += 1
            
            # Logging
            if self.global_step % log_every == 0 and self.is_main:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
                pbar.update(log_every)
                
                if self.use_wandb:
                    wandb.log(metrics, step=self.global_step)
            
            # Validation
            if self.global_step % val_every == 0:
                val_metrics = self.validate()
                
                if self.is_main and val_metrics:
                    print(f"\n[Step {self.global_step}] Validation: {val_metrics}")
                    
                    if self.use_wandb:
                        wandb.log(val_metrics, step=self.global_step)
                    
                    # Save best model
                    val_loss = val_metrics.get('val_nll', float('inf'))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best.pt')
                        print(f"  → New best model! (val_nll={val_loss:.4f})")
            
            # Checkpointing
            if self.global_step % save_every == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
                self.save_checkpoint('latest.pt')
        
        if self.is_main:
            pbar.close()
            self.save_checkpoint('final.pt')
            print(f"\n✓ Training complete! Best val_nll: {self.best_val_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
    
    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Create trainer
    trainer = LargeScaleTrainer(config, rank=rank, world_size=world_size)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    elif config.get('auto_resume', True):
        # Try to resume from latest checkpoint
        latest_ckpt = trainer.checkpoint_dir / 'latest.pt'
        if latest_ckpt.exists():
            trainer.load_checkpoint(latest_ckpt)
    
    # Train
    trainer.train()
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
