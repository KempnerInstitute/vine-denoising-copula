#!/usr/bin/env python
"""
Generate Publication-Quality Figures for Vine Copula Diffusion Paper

Creates:
1. Main comparison heatmap
2. Scaling analysis plots  
3. Per-scenario performance curves
4. Method comparison bar charts
"""
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_all_results():
    """Load all available results."""
    results = {}
    
    # Extended test results
    ext_dir = REPO_ROOT / 'results' / 'high_dimensional_extended'
    for run_dir in sorted(ext_dir.glob('*')):
        if (run_dir / 'results.json').exists():
            with open(run_dir / 'results.json') as f:
                results['extended'] = json.load(f)
    
    # Basic test results
    basic_dir = REPO_ROOT / 'results' / 'high_dimensional_test'
    for run_dir in sorted(basic_dir.glob('*')):
        if (run_dir / 'results.json').exists():
            with open(run_dir / 'results.json') as f:
                results['basic'] = json.load(f)
    
    return results


def create_main_heatmap(data, output_dir):
    """Create main performance heatmap for paper."""
    scenarios = sorted(set(r['scenario'] for r in data))
    dimensions = sorted(set(r['dimension'] for r in data))
    
    # Prepare matrices
    ks_matrix = np.zeros((len(scenarios), len(dimensions)))
    time_matrix = np.zeros((len(scenarios), len(dimensions)))
    
    for i, scenario in enumerate(scenarios):
        for j, d in enumerate(dimensions):
            result = next((r for r in data if r['scenario'] == scenario and r['dimension'] == d), None)
            if result:
                ks_matrix[i, j] = result.get('diffusion_ks_pvalue', result.get('mean_ks_pvalue', np.nan))
                time_matrix[i, j] = result.get('diffusion_fit_time', result.get('fit_time_sec', np.nan))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # KS p-value heatmap
    ax = axes[0]
    im = ax.imshow(ks_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.15)
    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels(dimensions)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s.replace('_', ' ').title() for s in scenarios])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Data Generating Process')
    ax.set_title('Rosenblatt Uniformity (KS p-value)')
    
    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(dimensions)):
            val = ks_matrix[i, j]
            color = 'white' if val < 0.05 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('KS p-value')
    
    # Time heatmap
    ax = axes[1]
    im = ax.imshow(time_matrix, aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels(dimensions)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s.replace('_', ' ').title() for s in scenarios])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Data Generating Process')
    ax.set_title('Computation Time (seconds)')
    
    for i in range(len(scenarios)):
        for j in range(len(dimensions)):
            val = time_matrix[i, j]
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                   color='white' if val > time_matrix.max()/2 else 'black', fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Time (s)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_performance_heatmap.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_performance_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Created: fig1_performance_heatmap")


def create_scaling_plot(data, output_dir):
    """Create computational scaling analysis plot."""
    dimensions = sorted(set(r['dimension'] for r in data))
    
    # Aggregate by dimension
    dim_stats = {}
    for d in dimensions:
        dim_data = [r for r in data if r['dimension'] == d]
        times = [r.get('diffusion_fit_time', r.get('fit_time_sec', np.nan)) for r in dim_data]
        ks_vals = [r.get('diffusion_ks_pvalue', r.get('mean_ks_pvalue', np.nan)) for r in dim_data]
        dim_stats[d] = {
            'time_mean': np.nanmean(times),
            'time_std': np.nanstd(times),
            'ks_mean': np.nanmean(ks_vals),
            'ks_std': np.nanstd(ks_vals),
            'pairs': d * (d - 1) // 2
        }
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Time vs Dimension
    ax = axes[0]
    dims = np.array(dimensions)
    times = np.array([dim_stats[d]['time_mean'] for d in dimensions])
    time_stds = np.array([dim_stats[d]['time_std'] for d in dimensions])
    ax.errorbar(dims, times, yerr=time_stds, fmt='o-', capsize=4, color='C0', markersize=8)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computational Cost')
    ax.grid(True, alpha=0.3)
    
    # Time vs Pairs (should be linear)
    ax = axes[1]
    pairs = np.array([dim_stats[d]['pairs'] for d in dimensions])
    ax.scatter(pairs, times, s=80, c='C0', edgecolors='black', linewidth=1)
    coeffs = np.polyfit(pairs, times, 1)
    x_fit = np.linspace(0, max(pairs)*1.1, 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r--', 
           label=f'Linear: {coeffs[0]:.2f}s/pair + {coeffs[1]:.1f}s')
    ax.set_xlabel('Number of Pair Copulas')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # KS p-value vs Dimension
    ax = axes[2]
    ks_means = np.array([dim_stats[d]['ks_mean'] for d in dimensions])
    ks_stds = np.array([dim_stats[d]['ks_std'] for d in dimensions])
    ax.errorbar(dims, ks_means, yerr=ks_stds, fmt='s-', capsize=4, color='C2', markersize=8)
    ax.axhline(y=0.05, color='r', linestyle='--', label='α = 0.05')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Mean KS p-value')
    ax.set_title('Uniformity Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(ks_means) * 1.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_scaling_analysis.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_scaling_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Created: fig2_scaling_analysis")


def create_method_comparison(data, output_dir):
    """Compare diffusion vs Gaussian baseline."""
    scenarios = sorted(set(r['scenario'] for r in data))
    
    # Filter to results with Gaussian baseline
    gauss_data = [r for r in data if 'gaussian_mean_logpdf' in r]
    
    if not gauss_data:
        print("No Gaussian baseline data found, skipping method comparison")
        return
    
    # Aggregate by scenario
    scenario_diff = {}
    for scenario in scenarios:
        s_data = [r for r in gauss_data if r['scenario'] == scenario]
        if s_data:
            diff_logpdfs = [r['diffusion_mean_logpdf'] for r in s_data]
            gauss_logpdfs = [r['gaussian_mean_logpdf'] for r in s_data]
            diff_ks = [r['diffusion_ks_pvalue'] for r in s_data]
            gauss_ks = [r.get('gaussian_ks_pvalue', np.nan) for r in s_data]
            
            scenario_diff[scenario] = {
                'diff_logpdf': np.mean(diff_logpdfs),
                'gauss_logpdf': np.mean(gauss_logpdfs),
                'diff_ks': np.mean(diff_ks),
                'gauss_ks': np.nanmean(gauss_ks),
            }
    
    if not scenario_diff:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(scenario_diff))
    width = 0.35
    
    # Log-PDF comparison
    ax = axes[0]
    scenarios_ordered = list(scenario_diff.keys())
    diff_vals = [scenario_diff[s]['diff_logpdf'] for s in scenarios_ordered]
    gauss_vals = [scenario_diff[s]['gauss_logpdf'] for s in scenarios_ordered]
    
    bars1 = ax.bar(x - width/2, diff_vals, width, label='Diffusion D-Vine', color='C0', edgecolor='black')
    bars2 = ax.bar(x + width/2, gauss_vals, width, label='Gaussian Copula', color='C1', edgecolor='black')
    
    ax.set_ylabel('Mean Log-PDF')
    ax.set_title('Log-Likelihood Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in scenarios_ordered], fontsize=9)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # KS p-value comparison
    ax = axes[1]
    diff_ks = [scenario_diff[s]['diff_ks'] for s in scenarios_ordered]
    gauss_ks = [scenario_diff[s]['gauss_ks'] for s in scenarios_ordered]
    
    ax.bar(x - width/2, diff_ks, width, label='Diffusion D-Vine', color='C0', edgecolor='black')
    ax.bar(x + width/2, gauss_ks, width, label='Gaussian Copula', color='C1', edgecolor='black')
    
    ax.axhline(y=0.05, color='r', linestyle='--', label='α = 0.05')
    ax.set_ylabel('Mean KS p-value')
    ax.set_title('Rosenblatt Uniformity Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in scenarios_ordered], fontsize=9)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_method_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_method_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Created: fig3_method_comparison")


def create_per_scenario_curves(data, output_dir):
    """Create per-scenario performance curves."""
    scenarios = sorted(set(r['scenario'] for r in data))
    dimensions = sorted(set(r['dimension'] for r in data))
    
    n_scenarios = len(scenarios)
    n_cols = 4
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        scenario_data = sorted([r for r in data if r['scenario'] == scenario], key=lambda x: x['dimension'])
        
        dims = [r['dimension'] for r in scenario_data]
        ks = [r.get('diffusion_ks_pvalue', r.get('mean_ks_pvalue', 0)) for r in scenario_data]
        
        ax.plot(dims, ks, 'o-', markersize=8, linewidth=2, color='C0')
        ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='α=0.05')
        ax.fill_between(dims, 0, 0.05, alpha=0.1, color='red')
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('KS p-value')
        ax.set_title(scenario.replace('_', ' ').title())
        ax.set_ylim(0, max(0.25, max(ks)*1.2))
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(scenarios), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_per_scenario.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_per_scenario.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Created: fig4_per_scenario")


def main():
    print("Generating Publication-Quality Figures")
    print("=" * 50)
    
    results = load_all_results()
    
    if not results:
        print("No results found!")
        return
    
    # Use extended results if available
    data = results.get('extended', results.get('basic', []))
    
    output_dir = REPO_ROOT / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nUsing {len(data)} experiments")
    print(f"Output: {output_dir}\n")
    
    create_main_heatmap(data, output_dir)
    create_scaling_plot(data, output_dir)
    create_method_comparison(data, output_dir)
    create_per_scenario_curves(data, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}")
    print("Generated:")
    print("  - fig1_performance_heatmap.{pdf,png}")
    print("  - fig2_scaling_analysis.{pdf,png}")
    print("  - fig3_method_comparison.{pdf,png}")
    print("  - fig4_per_scenario.{pdf,png}")


if __name__ == '__main__':
    main()
