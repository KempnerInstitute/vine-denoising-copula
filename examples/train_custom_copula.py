#!/usr/bin/env python3
"""
Example: Training a custom diffusion copula model.

This shows how to:
1. Create a custom training configuration
2. Train a model using the unified trainer
3. Monitor training progress
4. Use the trained model

Usage:
    python examples/train_custom_copula.py
    
This will create a config file and show training commands.
"""
import os
import sys
from pathlib import Path

import yaml

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def create_config(output_path: str = 'configs/train/custom_experiment.yaml'):
    """Create a custom training configuration."""
    
    config = {
        # Experiment metadata
        'experiment': {
            'name': 'custom_copula_experiment',
            'description': 'Custom diffusion copula training',
            'seed': 42,
        },
        
        # Model architecture
        'model': {
            'type': 'diffusion_unet',
            'grid_size': 64,           # Output density grid resolution
            'in_channels': 1,          # Input channels (log-density only)
            'base_channels': 64,       # Base feature channels
            'channel_mults': [1, 2, 3, 4],  # Channel multipliers per level
            'num_res_blocks': 2,       # Residual blocks per level
            'attention_resolutions': [16, 8],  # Add attention at these resolutions
            'dropout': 0.1,
            'time_emb_dim': 256,
            'use_coordinates': False,  # Don't concatenate coordinates
        },
        
        # Diffusion process
        'diffusion': {
            'timesteps': 1000,
            'noise_schedule': 'cosine',  # 'linear' or 'cosine'
        },
        
        # Training data generation
        'data': {
            'm': 64,                    # Grid resolution
            'n_samples_per_copula': 1000,  # Samples for histogram
            'copula_families': [        # Families to train on
                'gaussian',
                'student',
                'clayton',
                'gumbel',
                'frank',
                'joe',
            ],
            'binning': 'uniform',       # 'uniform' or 'probit'
            'num_workers': 4,
        },
        
        # Training parameters
        'training': {
            'max_steps': 100000,        # Total training steps
            'batch_size': 32,           # Batch size per GPU
            'learning_rate': 1.0e-4,
            'weight_decay': 0.01,
            'gradient_clip': 1.0,
            
            # Loss weights
            'loss_weights': {
                'ce': 1.0,              # Cross-entropy (primary loss)
                'ise': 0.1,             # Integrated squared error
                'tail': 0.05,           # Tail region loss
                'ms': 0.1,              # Multi-scale loss
                'marg_kl': 0.01,        # Marginal uniformity
            },
            
            # Copula projection
            'projection_iters': 10,     # IPFP iterations
            'detach_projection': True,  # Detach from gradient
            
            # Mixed precision
            'use_amp': True,
            
            # Logging
            'log_every': 100,
            'save_every': 10000,
            'viz_every': 5000,
            
            # Early stopping (optional)
            'early_stopping': {
                'enable': False,
                'metric': 'loss',
                'patience': 5000,
            },
        },
        
        # Output paths
        'output': {
            'base_dir': 'results',
            'checkpoint_dir': 'checkpoints',
            'include_timestamp': True,
            'include_job_id': True,
        },
        
        # Hardware
        'hardware': {
            'device': 'cuda',
            'num_workers': 4,
            'mixed_precision': True,
        },
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return output_path, config


def main():
    print("=" * 70)
    print("Vine Diffusion Copula - Custom Training Example")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Create custom configuration
    # =========================================================================
    print("\n[Step 1] Creating custom training configuration...")
    
    config_path, config = create_config()
    print(f"  ✓ Saved configuration to: {config_path}")
    print(f"\n  Key settings:")
    print(f"    - Grid resolution: {config['data']['m']}×{config['data']['m']}")
    print(f"    - Training steps: {config['training']['max_steps']:,}")
    print(f"    - Batch size: {config['training']['batch_size']}")
    print(f"    - Learning rate: {config['training']['learning_rate']}")
    print(f"    - Copula families: {', '.join(config['data']['copula_families'])}")
    
    # =========================================================================
    # Step 2: Training commands
    # =========================================================================
    print("\n[Step 2] Training Commands")
    print("-" * 50)
    
    print("""
  Single GPU training:
  
    python scripts/train.py --config {config}
  
  Multi-GPU training (4 GPUs):
  
    torchrun --nproc_per_node=4 scripts/train.py --config {config}
  
  Override parameters on command line:
  
    python scripts/train.py --config {config} \\
        training.max_steps=200000 \\
        training.learning_rate=5e-5 \\
        model.base_channels=128
  
  Resume from checkpoint:
  
    python scripts/train.py --config {config} \\
        --resume checkpoints/model_step_50000.pt
""".format(config=config_path))
    
    # =========================================================================
    # Step 3: SLURM submission
    # =========================================================================
    print("[Step 3] SLURM Cluster Submission")
    print("-" * 50)
    
    print(f"""
  Submit training job:
  
    CONFIG={config_path} sbatch slurm/train.sh
  
  Or edit slurm/train.sh to set your config path, then:
  
    sbatch slurm/train.sh
  
  Monitor progress:
  
    tail -f slurm_logs/vdc_train_*.out
""")
    
    # =========================================================================
    # Step 4: Using trained model
    # =========================================================================
    print("[Step 4] Using the Trained Model")
    print("-" * 50)
    
    print("""
  After training completes, use your model:
  
    import torch
    from vdc.models.unet_grid import GridUNet
    from vdc.models.copula_diffusion import CopulaAwareDiffusion
    from vdc.vine.api import VineCopulaModel
    
    # Load checkpoint
    checkpoint = torch.load('checkpoints/your_model/model_step_100000.pt')
    config = checkpoint['config']
    
    # Rebuild model
    model = GridUNet(
        m=config['data']['m'],
        in_channels=config['model'].get('in_channels', 1),
        base_channels=config['model'].get('base_channels', 64),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Fit vine copula to your data
    vine = VineCopulaModel(vine_type='dvine')
    vine.fit(your_data, model)
    
    # Evaluate and sample
    logpdf = vine.logpdf(test_data)
    samples = vine.simulate(n=1000)
""")
    
    # =========================================================================
    # Step 5: Configuration tips
    # =========================================================================
    print("[Step 5] Configuration Tips")
    print("-" * 50)
    
    print("""
  Adjust for your needs:
  
  ▶ More capacity (slower but potentially better):
    model.base_channels: 128
    model.num_res_blocks: 3
    
  ▶ Faster training (less capacity):
    model.base_channels: 32
    training.max_steps: 50000
    
  ▶ Better tail estimation:
    data.binning: 'probit'
    training.loss_weights.tail: 0.1
    
  ▶ More copula families:
    data.copula_families: ['gaussian', 'student', 'clayton', 
                           'gumbel', 'frank', 'joe']
    
  ▶ Larger batches (if memory allows):
    training.batch_size: 64
""")
    
    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print(f"\nConfiguration saved to: {config_path}")
    print("Edit this file to customize your experiment, then run training.")


if __name__ == '__main__':
    main()

