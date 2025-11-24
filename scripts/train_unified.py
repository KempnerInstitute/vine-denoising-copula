#!/usr/bin/env python3
"""
Thin CLI wrapper for the unified copula density training engine.

Usage:
    torchrun --nproc_per_node=4 scripts/train_unified.py \\
        --config configs/train_diffusion_uniform_m128.yaml \\
        --model-type diffusion_unet

    python scripts/train_unified.py \\
        --config configs/train_diffusion_probit_m128.yaml \\
        --model-type diffusion_unet \\
        --resume checkpoints/diffusion_probit_m128/model_step_10000.pt

All core training logic lives in vdc.train.unified_trainer.
This script handles only CLI argument parsing and distributed setup.
"""
import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vdc.train.unified_trainer import setup_distributed, train


def main():
    parser = argparse.ArgumentParser(description="Unified copula density model trainer")
    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help='Path to training config YAML file'
    )
    parser.add_argument(
        '--model-type',
        required=True,
        choices=['baseline_cnn', 'enhanced_cnn', 'denoiser', 'diffusion_unet'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--use-antialiased-hist',
        action='store_true',
        help='Override config to enable anti-aliased histogram target construction from samples'
    )
    parser.add_argument(
        '--independence-tau-thresh',
        type=float,
        default=None,
        help='Override config independence tau threshold'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply CLI overrides
    if args.use_antialiased_hist:
        config['training']['use_antialiased_hist'] = True
    if args.independence_tau_thresh is not None:
        config['data']['independence_tau_thresh'] = args.independence_tau_thresh
    
    # Distributed setup
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"=" * 80)
        print(f"Unified Copula Density Trainer")
        print(f"=" * 80)
        print(f"Model type:     {args.model_type}")
        print(f"Config:         {args.config}")
        print(f"World size:     {world_size}")
        print(f"Grid size:      {config['data']['m']}")
        print(f"Binning mode:   {config['data'].get('binning', 'uniform')}")
        print(f"Max steps:      {config['training']['max_steps']}")
        print(f"Checkpoint dir: {config.get('checkpoint_dir', f'checkpoints/{args.model_type}')}")
        if args.resume:
            print(f"Resuming from:  {args.resume}")
        print(f"=" * 80)
    
    # Launch training
    train(
        model_type=args.model_type,
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        resume_checkpoint=args.resume
    )
    
    if rank == 0:
        print("Training complete!")


if __name__ == '__main__':
    main()

