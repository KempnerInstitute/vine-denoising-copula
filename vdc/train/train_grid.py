"""
Training script for grid-based diffusion copula estimator.

Supports:
- Single and multi-GPU training with DDP
- Mixed precision training
- Gradient accumulation
- Checkpointing and logging
- Hydra configuration
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import wandb
from typing import Optional, Dict, Any
import argparse
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vdc.models.unet_grid import GridUNet
from vdc.models.projection import copula_project
from vdc.losses import nll_points, ise_logdensity, marginal_penalty, tail_weighted_loss
from vdc.data.loaders import CopulaPairsDataset


class DiffusionCopulaTrainer:
    """Main trainer class for diffusion copula model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = 'cuda',
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        
        # Config
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 400000)
        self.grad_accum_steps = self.config.get('grad_accum_steps', 1)
        self.val_every = self.config.get('val_every', 5000)
        self.save_every = self.config.get('save_every', 10000)
        self.log_every = self.config.get('log_every', 100)
        
        # Loss weights
        self.lambda_ise = self.config.get('lambda_ise', 1.0)
        self.lambda_marginal = self.config.get('lambda_marginal', 0.1)
        self.lambda_tail = self.config.get('lambda_tail', 0.5)
        
        # Optimizer
        if optimizer is None:
            lr = self.config.get('lr', 3e-4)
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        else:
            self.optimizer = optimizer
        
        # Mixed precision
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.use_wandb = self.config.get('use_wandb', False) and self.is_main_process
        if self.use_wandb:
            wandb.init(
                project=self.config.get('wandb_project', 'vine-copula'),
                name=self.config.get('wandb_run_name', None),
                config=self.config
            )
        
        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # Extract batch
        hist = batch['hist'].to(self.device)  # (B, 1, m, m)
        points = batch['points'].to(self.device)  # (B, n, 2)
        teacher_logpdf = batch.get('teacher_logpdf', None)
        if teacher_logpdf is not None:
            teacher_logpdf = teacher_logpdf.to(self.device)
        
        B, _, m, _ = hist.shape
        
        # Random noise time
        t = torch.rand(B, 1, 1, 1, device=self.device)
        
        # Forward pass
        with autocast(enabled=self.use_amp):
            # Predict log-density
            logD_raw = self.model(hist, t)
            
            # Convert to positive density
            D_pos = torch.exp(logD_raw)
            
            # Project to copula (IPFP/Sinkhorn)
            D_hat = copula_project(
                D_pos,
                iters=self.config.get('projection_iters', 20)
            )
            
            # NLL loss on points
            loss_nll = nll_points(D_hat, points)
            
            # ISE loss vs teacher (if available)
            loss_ise = 0.0
            if teacher_logpdf is not None:
                logD_hat = torch.log(D_hat + 1e-12)
                loss_ise = ise_logdensity(logD_hat, teacher_logpdf)
            
            # Marginal penalty
            loss_marginal = marginal_penalty(D_hat)
            
            # Tail weighting
            loss_tail = tail_weighted_loss(D_hat, points)
            
            # Total loss
            loss = (
                loss_nll +
                self.lambda_ise * loss_ise +
                self.lambda_marginal * loss_marginal +
                self.lambda_tail * loss_tail
            )
            
            # Scale for gradient accumulation
            loss = loss / self.grad_accum_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Metrics
        metrics = {
            'loss': loss.item() * self.grad_accum_steps,
            'loss_nll': loss_nll.item(),
            'loss_ise': loss_ise.item() if isinstance(loss_ise, torch.Tensor) else loss_ise,
            'loss_marginal': loss_marginal.item(),
            'loss_tail': loss_tail.item(),
        }
        
        return metrics
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.use_amp:
            # Unscale before clipping
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_metrics = []
        
        for batch in self.val_loader:
            hist = batch['hist'].to(self.device)
            points = batch['points'].to(self.device)
            teacher_logpdf = batch.get('teacher_logpdf', None)
            if teacher_logpdf is not None:
                teacher_logpdf = teacher_logpdf.to(self.device)
            
            B, _, m, _ = hist.shape
            t = torch.ones(B, 1, 1, 1, device=self.device) * 0.5  # Fixed time for eval
            
            with autocast(enabled=self.use_amp):
                logD_raw = self.model(hist, t)
                D_pos = torch.exp(logD_raw)
                D_hat = copula_project(D_pos)
                
                loss_nll = nll_points(D_hat, points)
                
                loss_ise = 0.0
                if teacher_logpdf is not None:
                    logD_hat = torch.log(D_hat + 1e-12)
                    loss_ise = ise_logdensity(logD_hat, teacher_logpdf)
                
                loss_marginal = marginal_penalty(D_hat)
            
            val_metrics.append({
                'val_nll': loss_nll.item(),
                'val_ise': loss_ise.item() if isinstance(loss_ise, torch.Tensor) else loss_ise,
                'val_marginal': loss_marginal.item(),
            })
        
        self.model.train()
        
        # Average metrics
        avg_metrics = {}
        for key in val_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in val_metrics])
        
        return avg_metrics
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save checkpoint."""
        if not self.is_main_process:
            return
        
        # Unwrap DDP if necessary
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load checkpoint."""
        path = self.checkpoint_dir / filename
        if not path.exists():
            print(f"Checkpoint {path} not found")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load state
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
    
    def train(self):
        """Main training loop."""
        self.model.train()
        
        pbar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            disable=not self.is_main_process,
            desc='Training'
        )
        
        while self.global_step < self.max_steps:
            for batch in self.train_loader:
                # Training step with gradient accumulation
                metrics = self.train_step(batch)
                
                # Optimizer step every grad_accum_steps
                if (self.global_step + 1) % self.grad_accum_steps == 0:
                    self.optimizer_step()
                
                self.global_step += 1
                pbar.update(1)
                
                # Logging
                if self.global_step % self.log_every == 0 and self.is_main_process:
                    pbar.set_postfix(metrics)
                    if self.use_wandb:
                        wandb.log(metrics, step=self.global_step)
                
                # Validation
                if self.global_step % self.val_every == 0:
                    val_metrics = self.validate()
                    if self.is_main_process and val_metrics:
                        print(f"\nValidation at step {self.global_step}: {val_metrics}")
                        if self.use_wandb:
                            wandb.log(val_metrics, step=self.global_step)
                        
                        # Save best model
                        val_loss = val_metrics.get('val_nll', float('inf'))
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint('best.pt')
                
                # Checkpointing
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
                    self.save_checkpoint('latest.pt')
                
                if self.global_step >= self.max_steps:
                    break
            
            self.epoch += 1
        
        pbar.close()
        
        # Final checkpoint
        if self.is_main_process:
            self.save_checkpoint('final.pt')
        
        if self.use_wandb:
            wandb.finish()


def setup_ddp(rank: int, world_size: int):
    """Initialize DDP."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--m', type=int, default=64, help='Grid resolution')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=400000, help='Max training steps')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use W&B logging')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # DDP setup
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        setup_ddp(local_rank, world_size)
    
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    
    # Config
    config = {
        'm': args.m,
        'lr': args.lr,
        'max_steps': args.max_steps,
        'checkpoint_dir': args.checkpoint_dir,
        'use_wandb': args.use_wandb,
        'batch_size': args.batch_size,
        'grad_accum_steps': 1,
        'val_every': 5000,
        'save_every': 10000,
        'log_every': 100,
        'lambda_ise': 1.0,
        'lambda_marginal': 0.1,
        'lambda_tail': 0.5,
        'use_amp': True,
        'max_grad_norm': 1.0,
        'projection_iters': 20,
    }
    
    # Model
    model = GridUNet(m=args.m, in_channels=1, base_channels=64)
    model = model.to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Dataset
    train_dataset = CopulaPairsDataset(
        root=args.data_root,
        split='train',
        m=args.m
    )
    
    val_dataset = CopulaPairsDataset(
        root=args.data_root,
        split='val',
        m=args.m
    )
    
    # DataLoader
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Trainer
    trainer = DiffusionCopulaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        rank=local_rank,
        world_size=world_size
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    # Cleanup
    if world_size > 1:
        cleanup_ddp()


if __name__ == '__main__':
    main()
