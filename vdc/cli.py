"""
Command-line interface for vine diffusion copula operations.

Provides easy access to all major functionality:
- Data generation
- Model training
- Vine fitting
- Evaluation
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Vine Diffusion Copula CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic training data
  vdc generate --output data/synthetic --n-samples 2000000 --m 64

  # Train model (single GPU)
  vdc train --data-root data/synthetic --m 64 --batch-size 32

  # Train model (multi-GPU)
  torchrun --nproc_per_node=4 -m vdc.cli train --data-root data/synthetic

  # Fit vine to data
  vdc fit --data my_data.csv --model checkpoints/best.pt --output my_vine.pkl

  # Evaluate vine
  vdc evaluate --vine my_vine.pkl --test my_test.csv

For more information, see: https://github.com/KempnerInstitute/vine-diffusion-copula
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ========== Generate Data ==========
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic training data')
    gen_parser.add_argument('--output', type=str, required=True, help='Output directory')
    gen_parser.add_argument('--n-samples', type=int, default=2000000, help='Number of samples')
    gen_parser.add_argument('--m', type=int, default=64, help='Grid resolution')
    gen_parser.add_argument('--families', type=str, default='all', help='Copula families (comma-separated)')
    gen_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # ========== Train Model ==========
    train_parser = subparsers.add_parser('train', help='Train diffusion copula model')
    train_parser.add_argument('--data-root', type=str, required=True, help='Training data directory')
    train_parser.add_argument('--m', type=int, default=64, help='Grid resolution')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    train_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--max-steps', type=int, default=400000, help='Max training steps')
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--use-wandb', action='store_true', help='Use W&B logging')
    train_parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # ========== Fit Vine ==========
    fit_parser = subparsers.add_parser('fit', help='Fit vine copula to data')
    fit_parser.add_argument('--data', type=str, required=True, help='Data file (CSV/NPY)')
    fit_parser.add_argument('--model', type=str, required=True, help='Trained model checkpoint')
    fit_parser.add_argument('--output', type=str, required=True, help='Output vine file (PKL)')
    fit_parser.add_argument('--m', type=int, default=64, help='Grid resolution')
    fit_parser.add_argument('--max-trees', type=int, default=None, help='Maximum vine trees')
    fit_parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # ========== Evaluate Vine ==========
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate fitted vine')
    eval_parser.add_argument('--vine', type=str, required=True, help='Fitted vine file (PKL)')
    eval_parser.add_argument('--test', type=str, required=True, help='Test data (CSV/NPY)')
    eval_parser.add_argument('--output', type=str, default=None, help='Output metrics file (JSON)')
    eval_parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    # ========== Sample from Vine ==========
    sample_parser = subparsers.add_parser('sample', help='Generate samples from vine')
    sample_parser.add_argument('--vine', type=str, required=True, help='Fitted vine file (PKL)')
    sample_parser.add_argument('--n', type=int, default=1000, help='Number of samples')
    sample_parser.add_argument('--output', type=str, required=True, help='Output file (CSV/NPY)')
    sample_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    if args.command == 'generate':
        from vdc.data.generators import generate_synthetic_dataset
        generate_synthetic_dataset(args)
    
    elif args.command == 'train':
        from vdc.train.train_grid import main as train_main
        train_main()
    
    elif args.command == 'fit':
        from vdc.vine.api import fit_vine_from_cli
        fit_vine_from_cli(args)
    
    elif args.command == 'evaluate':
        from vdc.eval.vine import evaluate_vine_from_cli
        evaluate_vine_from_cli(args)
    
    elif args.command == 'sample':
        from vdc.vine.api import sample_from_vine_cli
        sample_from_vine_cli(args)
    
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
