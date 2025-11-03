#!/usr/bin/env python3
"""
Master data generation script for comprehensive copula training.

Generates 42M samples covering:
- Single-family parametric copulas (720k)
- 2-component mixtures (5M)
- 3-component mixtures (10M)
- 4-5 component mixtures (5M)
- Vine conditionals 3D (5M)
- Vine conditionals 5D (3M)
- Hard examples (1M)

Total: ~42M samples, ~4.2 TB storage
"""

import argparse
import os
import sys
from pathlib import Path
import multiprocessing as mp
from datetime import datetime


def run_stage(stage_name, command, dry_run=False):
    """Run a data generation stage."""
    print(f"\n{'='*80}")
    print(f"STAGE: {stage_name}")
    print(f"{'='*80}")
    print(f"Command: {command}\n")
    
    if dry_run:
        print("[DRY RUN] Skipping actual execution\n")
        return True
    
    start_time = datetime.now()
    ret = os.system(command)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600  # hours
    
    if ret == 0:
        print(f"✓ {stage_name} completed in {duration:.1f} hours\n")
        return True
    else:
        print(f"✗ {stage_name} FAILED with exit code {ret}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive copula training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset generation
  python generate_comprehensive_data.py --output-dir data/train --n-jobs 32
  
  # Quick test (1% of data)
  python generate_comprehensive_data.py --output-dir data/test --quick
  
  # Dry run (show commands without executing)
  python generate_comprehensive_data.py --output-dir data/train --dry-run
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/train',
        help='Root output directory for all datasets'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=mp.cpu_count(),
        help='Number of parallel workers (default: all CPUs)'
    )
    
    parser.add_argument(
        '--m',
        type=int,
        default=64,
        help='Grid resolution (default: 64)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Generate 1%% of data for quick testing (~420k samples)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show commands without executing'
    )
    
    parser.add_argument(
        '--stages',
        type=str,
        nargs='+',
        choices=['single', 'mix2', 'mix3', 'mix45', 'vine3d', 'vine5d', 'hard', 'all'],
        default=['all'],
        help='Which stages to run (default: all)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Scale factor for quick mode
    scale = 0.01 if args.quick else 1.0
    
    # Sample counts
    n_single = int(720_000 * scale)
    n_mix2 = int(5_000_000 * scale)
    n_mix3 = int(10_000_000 * scale)
    n_mix45 = int(5_000_000 * scale)
    n_vine3d = int(5_000_000 * scale)
    n_vine5d = int(3_000_000 * scale)
    n_hard = int(1_000_000 * scale)
    
    total_samples = n_single + n_mix2 + n_mix3 + n_mix45 + n_vine3d + n_vine5d + n_hard
    total_storage = total_samples * 100 / 1024**3  # ~100KB per sample
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║         COMPREHENSIVE COPULA DATASET GENERATION                       ║
╚═══════════════════════════════════════════════════════════════════════╝

Configuration:
  Output directory:  {args.output_dir}
  Parallel workers:  {args.n_jobs}
  Grid resolution:   {args.m}×{args.m}
  Random seed:       {args.seed}
  Mode:              {'QUICK TEST (1%)' if args.quick else 'FULL PRODUCTION'}
  Dry run:           {args.dry_run}

Dataset Composition:
  Single-family:     {n_single:>10,} samples  ({n_single/total_samples*100:>5.1f}%)
  2-comp mixtures:   {n_mix2:>10,} samples  ({n_mix2/total_samples*100:>5.1f}%)
  3-comp mixtures:   {n_mix3:>10,} samples  ({n_mix3/total_samples*100:>5.1f}%)
  4-5 comp mixtures: {n_mix45:>10,} samples  ({n_mix45/total_samples*100:>5.1f}%)
  Vine cond 3D:      {n_vine3d:>10,} samples  ({n_vine3d/total_samples*100:>5.1f}%)
  Vine cond 5D:      {n_vine5d:>10,} samples  ({n_vine5d/total_samples*100:>5.1f}%)
  Hard examples:     {n_hard:>10,} samples  ({n_hard/total_samples*100:>5.1f}%)
  {'─'*71}
  TOTAL:             {total_samples:>10,} samples  (100.0%)
  
Estimated Storage:   ~{total_storage:.1f} GB
    """)
    
    if not args.dry_run:
        response = input("Proceed with data generation? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which stages to run
    run_all = 'all' in args.stages
    stages_to_run = {
        'single': run_all or 'single' in args.stages,
        'mix2': run_all or 'mix2' in args.stages,
        'mix3': run_all or 'mix3' in args.stages,
        'mix45': run_all or 'mix45' in args.stages,
        'vine3d': run_all or 'vine3d' in args.stages,
        'vine5d': run_all or 'vine5d' in args.stages,
        'hard': run_all or 'hard' in args.stages,
    }
    
    success = True
    start_time = datetime.now()
    
    # ========================================================================
    # STAGE 1: Single-family parametric copulas
    # ========================================================================
    if stages_to_run['single']:
        cmd = f"""python -m vdc.data.generators \\
    --output {output_dir}/single_family \\
    --n-samples {n_single} \\
    --m {args.m} \\
    --families gaussian student clayton gumbel frank joe bb1 bb6 bb7 bb8 \\
    --include-rotations \\
    --include-edge-cases \\
    --tau-range -0.95 0.95 \\
    --n-jobs {args.n_jobs} \\
    --seed {args.seed}"""
        
        success = success and run_stage(
            "Stage 1: Single-family copulas",
            cmd,
            args.dry_run
        )
    
    # ========================================================================
    # STAGE 2: 2-component mixtures
    # ========================================================================
    if stages_to_run['mix2'] and success:
        cmd = f"""python -m vdc.data.mixtures \\
    --output {output_dir}/mixtures_2comp \\
    --n-samples {n_mix2} \\
    --n-components-min 2 \\
    --n-components-max 2 \\
    --m {args.m} \\
    --families gaussian student clayton gumbel frank joe bb1 bb7 \\
    --tau-range -0.7 0.7 \\
    --include-intra-family \\
    --include-asymmetric \\
    --n-jobs {args.n_jobs} \\
    --seed {args.seed + 1}"""
        
        success = success and run_stage(
            "Stage 2: 2-component mixtures",
            cmd,
            args.dry_run
        )
    
    # ========================================================================
    # STAGE 3: 3-component mixtures
    # ========================================================================
    if stages_to_run['mix3'] and success:
        cmd = f"""python -m vdc.data.mixtures \\
    --output {output_dir}/mixtures_3comp \\
    --n-samples {n_mix3} \\
    --n-components-min 3 \\
    --n-components-max 3 \\
    --m {args.m} \\
    --families gaussian student clayton gumbel frank joe bb1 bb7 \\
    --tau-range -0.7 0.7 \\
    --n-jobs {args.n_jobs} \\
    --seed {args.seed + 2}"""
        
        success = success and run_stage(
            "Stage 3: 3-component mixtures",
            cmd,
            args.dry_run
        )
    
    # ========================================================================
    # STAGE 4: 4-5 component mixtures
    # ========================================================================
    if stages_to_run['mix45'] and success:
        # Split into 4-comp and 5-comp
        n_mix4 = int(n_mix45 * 0.6)
        n_mix5 = n_mix45 - n_mix4
        
        cmd4 = f"""python -m vdc.data.mixtures \\
    --output {output_dir}/mixtures_4_5comp/4comp \\
    --n-samples {n_mix4} \\
    --n-components-min 4 \\
    --n-components-max 4 \\
    --m {args.m} \\
    --families gaussian student clayton gumbel frank joe bb1 bb7 \\
    --tau-range -0.6 0.6 \\
    --n-jobs {args.n_jobs} \\
    --seed {args.seed + 3}"""
        
        cmd5 = f"""python -m vdc.data.mixtures \\
    --output {output_dir}/mixtures_4_5comp/5comp \\
    --n-samples {n_mix5} \\
    --n-components-min 5 \\
    --n-components-max 5 \\
    --m {args.m} \\
    --families gaussian clayton gumbel frank joe bb1 \\
    --tau-range -0.5 0.5 \\
    --n-jobs {args.n_jobs} \\
    --seed {args.seed + 4}"""
        
        success = success and run_stage(
            "Stage 4a: 4-component mixtures",
            cmd4,
            args.dry_run
        )
        
        success = success and run_stage(
            "Stage 4b: 5-component mixtures",
            cmd5,
            args.dry_run
        )
    
    # ========================================================================
    # STAGE 5: Vine conditionals (3D)
    # ========================================================================
    if stages_to_run['vine3d'] and success:
        cmd = f"""python -m vdc.data.vine_conditionals \\
    --output {output_dir}/vine_cond_3d \\
    --n-samples {n_vine3d} \\
    --vine-dim 3 \\
    --vine-types rvine cvine dvine \\
    --m {args.m} \\
    --n-jobs {args.n_jobs} \\
    --seed {args.seed + 5}"""
        
        success = success and run_stage(
            "Stage 5: 3D vine conditionals",
            cmd,
            args.dry_run
        )
    
    # ========================================================================
    # STAGE 6: Vine conditionals (5D)
    # ========================================================================
    if stages_to_run['vine5d'] and success:
        cmd = f"""python -m vdc.data.vine_conditionals \\
    --output {output_dir}/vine_cond_5d \\
    --n-samples {n_vine5d} \\
    --vine-dim 5 \\
    --vine-types rvine cvine \\
    --m {args.m} \\
    --n-jobs {args.n_jobs // 2} \\
    --seed {args.seed + 6}"""
        
        success = success and run_stage(
            "Stage 6: 5D vine conditionals",
            cmd,
            args.dry_run
        )
    
    # ========================================================================
    # STAGE 7: Hard examples
    # ========================================================================
    if stages_to_run['hard'] and success:
        cmd = f"""python -m vdc.data.adversarial \\
    --output {output_dir}/hard_examples \\
    --n-samples {n_hard} \\
    --m {args.m} \\
    --include-near-singular \\
    --include-extreme-tails \\
    --include-oscillatory \\
    --include-sparse-data \\
    --n-jobs {args.n_jobs} \\
    --seed {args.seed + 7}"""
        
        success = success and run_stage(
            "Stage 7: Hard examples",
            cmd,
            args.dry_run
        )
    
    # ========================================================================
    # Summary
    # ========================================================================
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 3600
    
    print(f"\n{'='*80}")
    if success:
        print("✓ ALL STAGES COMPLETED SUCCESSFULLY")
    else:
        print("✗ SOME STAGES FAILED")
    print(f"{'='*80}")
    print(f"Total time: {total_duration:.1f} hours")
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {total_samples:,}")
    print(f"Storage used: ~{total_storage:.1f} GB")
    
    if success and not args.dry_run:
        print("\nNext steps:")
        print("1. Validate dataset:")
        print(f"   python scripts/validate_dataset.py {output_dir}")
        print("2. Generate validation split:")
        print(f"   python scripts/split_train_val.py {output_dir}")
        print("3. Start training:")
        print("   sbatch scripts/slurm/train_comprehensive.sh")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
