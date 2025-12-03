#!/usr/bin/env python3
"""
Example: Training a custom diffusion copula model.

This shows how external users can train their own diffusion copula model
with custom configurations.
"""
import yaml


def main():
    """Demonstrate training a custom model."""
    
    print("=" * 80)
    print("Training a Custom Diffusion Copula Model")
    print("=" * 80)
    
    # Step 1: Create a custom configuration
    print("\n1. Creating custom configuration...")
    
    config = {
        'data': {
            'm': 128,  # Grid resolution
            'binning': 'uniform',  # or 'probit' for boundary-focused
            'n_samples_per_copula': 5000,
            'copula_families': {
                'gaussian': 0.3,
                'clayton': 0.3,
                'gumbel': 0.2,
                'frank': 0.2,
            }
        },
        'model': {
            'type': 'diffusion_unet',
            'base_channels': 64,
            'channel_mults': [1, 2, 3, 4],
            'num_res_blocks': 2,
            'attention_resolutions': [32, 16, 8],
            'dropout': 0.1,
        },
        'diffusion': {
            'timesteps': 1000,
            'noise_schedule': 'cosine',
        },
        'training': {
            'batch_size': 24,
            'max_steps': 5000,
            'learning_rate': 5e-5,
            'projection_iters': 15,
            'detach_projection': True,
            'use_amp': False,
            'gradient_clip': 2.0,
            'loss_weights': {
                'ise': 0.25,
                'tail': 0.05,
                'marg_kl': 0.01,
            },
        },
        'checkpoint_dir': 'checkpoints/my_custom_model',
    }
    
    # Save config
    config_path = 'configs/my_custom_model.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Saved configuration to: {config_path}")
    
    # Step 2: Training command
    print("\n2. Training command:")
    print(f"""
    To train on a single GPU:
      python scripts/train_unified.py \\
          --config {config_path} \\
          --model-type diffusion_unet
    
    To train on multiple GPUs (e.g., 4 GPUs):
      torchrun --nproc_per_node=4 scripts/train_unified.py \\
          --config {config_path} \\
          --model-type diffusion_unet
    
    Or submit as a SLURM job:
      # Create a job script similar to slurm_jobs/validate_diffusion_no_probit_m128.sh
      # Then: sbatch my_job.sh
    """)
    
    # Step 3: Using the trained model
    print("\n3. After training, use the model:")
    print("""
    from vdc.vine.copula_diffusion import DiffusionCopulaModel
    
    # Load your trained model
    model = DiffusionCopulaModel.from_checkpoint(
        'checkpoints/my_custom_model/model_step_5000.pt'
    )
    
    # Use for density estimation
    density, u, v = model.estimate_density_from_samples(your_data)
    
    # Get h-functions for vine copulas
    h1, h2 = model.h_functions_from_grid(density)
    
    # Sample from the estimated copula
    samples = model.sample_from_density(density, n_samples=5000)
    """)
    
    print("\n" + "=" * 80)
    print("✓ Example complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the generated config: configs/my_custom_model.yaml")
    print("  2. Modify parameters as needed")
    print("  3. Run training with the command above")
    print("  4. Use the trained model with DiffusionCopulaModel.from_checkpoint()")


if __name__ == '__main__':
    main()

