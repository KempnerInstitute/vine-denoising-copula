#!/usr/bin/env python
"""
Quick start script for training diffusion copula networks.

This script provides an easy interface to launch training with sensible defaults.

Usage:
    # Quick test (1 GPU, small model, short training)
    python scripts/quick_train.py --mode test
    
    # Local multi-GPU (4 GPUs, medium scale)
    python scripts/quick_train.py --mode local --gpus 4
    
    # Production (SLURM cluster, large scale)
    python scripts/quick_train.py --mode cluster --nodes 4 --gpus 4
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def update_config(config_path: Path, updates: dict) -> Path:
    """Update config file with new values."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply updates
    for key_path, value in updates.items():
        keys = key_path.split('.')
        obj = config
        for key in keys[:-1]:
            obj = obj[key]
        obj[keys[-1]] = value
    
    # Save to temporary file
    temp_config = Path('configs/.temp_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    return temp_config


def launch_test(args):
    """Launch quick test training."""
    print("=" * 60)
    print("QUICK TEST MODE")
    print("=" * 60)
    print("Configuration:")
    print(f"  Model: m=32, base_channels=64")
    print(f"  Data: {args.data_dir}")
    print(f"  Steps: 5000 (quick test)")
    print(f"  GPU: 1")
    print("=" * 60)
    
    # Update config
    updates = {
        'data.train_root': str(args.data_dir / 'train'),
        'data.val_root': str(args.data_dir / 'val'),
        'training.max_steps': 5000,
        'training.log_every': 50,
        'training.val_every': 1000,
        'use_wandb': False,
    }
    
    config = update_config(Path('configs/train_quick.yaml'), updates)
    
    # Launch
    cmd = [
        sys.executable,
        'scripts/train_large_scale.py',
        '--config', str(config),
    ]
    
    print(f"\nLaunching: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def launch_local(args):
    """Launch local multi-GPU training."""
    print("=" * 60)
    print("LOCAL MULTI-GPU MODE")
    print("=" * 60)
    print("Configuration:")
    print(f"  Model: m=64, base_channels=128")
    print(f"  Data: {args.data_dir}")
    print(f"  Steps: {args.steps}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Batch size: {args.batch_size} per GPU")
    print("=" * 60)
    
    # Update config
    updates = {
        'data.train_root': str(args.data_dir / 'train'),
        'data.val_root': str(args.data_dir / 'val'),
        'data.batch_size': args.batch_size,
        'training.max_steps': args.steps,
        'checkpoint_dir': f'checkpoints/local_{args.gpus}gpu',
        'use_wandb': args.wandb,
    }
    
    config = update_config(Path('configs/train_large.yaml'), updates)
    
    # Launch with torchrun
    cmd = [
        'torchrun',
        f'--nproc_per_node={args.gpus}',
        '--master_addr=localhost',
        '--master_port=29500',
        'scripts/train_large_scale.py',
        '--config', str(config),
    ]
    
    print(f"\nLaunching: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def launch_cluster(args):
    """Launch cluster training via SLURM."""
    print("=" * 60)
    print("CLUSTER MODE (SLURM)")
    print("=" * 60)
    print("Configuration:")
    print(f"  Nodes: {args.nodes}")
    print(f"  GPUs per node: {args.gpus}")
    print(f"  Total GPUs: {args.nodes * args.gpus}")
    print(f"  Batch size: {args.batch_size} per GPU")
    print(f"  Steps: {args.steps}")
    print(f"  Time limit: {args.time}h")
    print("=" * 60)
    
    # Update config
    updates = {
        'data.train_root': str(args.data_dir / 'train'),
        'data.val_root': str(args.data_dir / 'val'),
        'data.batch_size': args.batch_size,
        'training.max_steps': args.steps,
        'checkpoint_dir': f'checkpoints/cluster_{args.nodes}n_{args.gpus}g',
        'use_wandb': args.wandb,
    }
    
    config_path = update_config(Path('configs/train_large.yaml'), updates)
    
    # Create SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=copula-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --mem={args.mem}G
#SBATCH --time={args.time}:00:00
#SBATCH --partition={args.partition}

module purge
module load python/3.10 cuda/12.1 cudnn/8.9
source activate vine-copula

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun torchrun \\
    --nnodes=$SLURM_JOB_NUM_NODES \\
    --nproc_per_node=$SLURM_GPUS_ON_NODE \\
    --rdzv_id=$SLURM_JOB_ID \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
    scripts/train_large_scale.py \\
    --config {config_path}
"""
    
    slurm_file = Path('scripts/.temp_slurm.sh')
    with open(slurm_file, 'w') as f:
        f.write(slurm_script)
    
    # Submit job
    cmd = ['sbatch', str(slurm_file)]
    
    print(f"\nSubmitting job: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode == 0:
        print("✓ Job submitted successfully!")
        print("\nMonitor with: squeue -u $USER")
        print(f"View logs: tail -f logs/train_*.out")


def main():
    parser = argparse.ArgumentParser(description='Quick training launcher')
    parser.add_argument(
        '--mode',
        choices=['test', 'local', 'cluster'],
        required=True,
        help='Training mode: test (quick), local (multi-GPU), cluster (SLURM)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Root data directory (should contain train/ and val/)'
    )
    parser.add_argument('--gpus', type=int, default=4, help='GPUs per node')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes (cluster only)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--steps', type=int, default=100000, help='Training steps')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    
    # Cluster-specific
    parser.add_argument('--time', type=int, default=48, help='Time limit (hours, cluster only)')
    parser.add_argument('--partition', type=str, default='gpu', help='SLURM partition')
    parser.add_argument('--mem', type=int, default=128, help='Memory per node (GB)')
    parser.add_argument('--cpus-per-task', type=int, default=16, help='CPUs per task')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory {args.data_dir} does not exist")
        print("\nGenerate data first:")
        print(f"  python -m vdc.data.generators --output {args.data_dir}/train --n-samples 100000")
        sys.exit(1)
    
    # Launch appropriate mode
    if args.mode == 'test':
        launch_test(args)
    elif args.mode == 'local':
        launch_local(args)
    elif args.mode == 'cluster':
        launch_cluster(args)


if __name__ == '__main__':
    main()
