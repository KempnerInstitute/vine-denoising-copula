#!/usr/bin/env python
"""
Unified training script for Vine Denoising Copula.

This is the main entry point for training copula density estimation models.
It wraps the unified trainer with a clean interface.

Usage:
    # Train with default config
    python scripts/train.py --config configs/train/default.yaml
    
    # Train with custom model type
    python scripts/train.py --config configs/train/default.yaml --model-type diffusion_unet
    
    # Override specific values
    python scripts/train.py --config configs/train/default.yaml \
        training.learning_rate=0.0001 model.base_channels=128

    # Resume from checkpoint
    python scripts/train.py --config configs/train/default.yaml --resume checkpoints/model_step_50000.pt
    
    # Multi-GPU training with torchrun
    torchrun --nproc_per_node=4 scripts/train.py --config configs/train/default.yaml
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.config import Config, get_run_dir, save_run_config


def main():
    parser = argparse.ArgumentParser(
        description="Train Vine Denoising Copula models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/train.py --config configs/train/default.yaml
  
  # With specific model type
  python scripts/train.py --config configs/train/default.yaml --model-type diffusion_unet
  
  # Multi-GPU with torchrun
  torchrun --nproc_per_node=4 scripts/train.py --config configs/train/default.yaml
  
  # Resume training
  python scripts/train.py --config configs/train/default.yaml --resume checkpoints/model.pt
        """
    )
    parser.add_argument(
        '--config', type=Path, 
        default=REPO_ROOT / 'configs' / 'train' / 'default.yaml',
        help='Path to training config YAML file'
    )
    parser.add_argument(
        '--model-type', type=str, 
        choices=['diffusion_unet', 'enhanced_cnn', 'denoiser', 'baseline_cnn'],
        default='diffusion_unet',
        help='Model architecture type'
    )
    parser.add_argument(
        '--resume', type=Path, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Override checkpoint directory'
    )
    parser.add_argument(
        'overrides', nargs='*',
        help='Config overrides in format key=value or key.subkey=value'
    )
    
    args = parser.parse_args()
    
    # Load config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        print(f"Available configs:")
        config_dir = REPO_ROOT / 'configs' / 'train'
        if config_dir.exists():
            for f in config_dir.glob('*.yaml'):
                print(f"  {f}")
        sys.exit(1)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Apply command-line overrides
    if args.overrides:
        from vdc.config import parse_overrides, merge_configs
        override_dict = parse_overrides(args.overrides)
        config = merge_configs(config, override_dict)
    
    # Set checkpoint directory
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    else:
        # Create timestamped checkpoint dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = os.environ.get('SLURM_JOB_ID', '')
        dir_name = f"{args.model_type}_{timestamp}"
        if job_id:
            dir_name = f"{dir_name}_job{job_id}"
        config['checkpoint_dir'] = str(REPO_ROOT / 'checkpoints' / dir_name)
    
    # Import and run the unified trainer
    from vdc.train.unified_trainer import train, setup_distributed, cleanup_distributed
    
    # Setup distributed if needed
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("Vine Denoising Copula - Training")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"Model type: {args.model_type}")
        print(f"Checkpoint dir: {config['checkpoint_dir']}")
        print(f"World size: {world_size}")
        if args.resume:
            print(f"Resuming from: {args.resume}")
        print("=" * 60)
    
    try:
        train(
            model_type=args.model_type,
            config=config,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            resume_checkpoint=str(args.resume) if args.resume else None,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
