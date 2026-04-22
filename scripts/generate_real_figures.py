#!/usr/bin/env python3
"""
Generate figures from real experiment results.

This script reads results from:
  1. Scaling experiments (timing vs dimension)
  2. Model selection results (ISE, MI, tau errors)
  3. MI estimation benchmarks
  4. Probit vs non-probit comparison

And generates publication-ready figures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color palette
COLORS = {
    'ours': '#2E86AB',          # Blue
    'ours_batched': '#3498db',  # Lighter blue
    'ours_probit': '#E76F51',   # Coral
    'denoiser': '#28A745',      # Green
    'pyvine_param': '#F4A261',  # Orange
    'pyvine_nonpar': '#E9C46A', # Yellow
    'histogram': '#6C757D',     # Gray
    'kde': '#9B59B6',           # Purple
    'ksg': '#17A2B8',           # Cyan
    'infonce': '#DC3545',       # Red
}

MARKERS = {
    'ours': 'o',
    'ours_batched': 's',
    'ours_probit': '^',
    'denoiser': 'D',
    'pyvine_param': 'v',
    'pyvine_nonpar': '<',
    'histogram': 'x',
    'kde': '+',
    'ksg': '*',
}


def load_scaling_results(output_base: Path) -> Optional[Dict]:
    """Load scaling experiment results (timing only, not accuracy)."""
    # Find latest timing-only scaling results (exclude accuracy experiments)
    scaling_dirs = sorted([
        d for d in output_base.glob("vdc_paper_scaling_*")
        if "accuracy" not in d.name
    ])
    if not scaling_dirs:
        return None
    
    # Search from latest to earliest for valid results
    for latest_dir in reversed(scaling_dirs):
        results_path = latest_dir / "results" / "e2_scaling_results.json"
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
    
    return None


def load_scaling_accuracy_results(output_base: Path) -> Optional[Dict]:
    """Load scaling accuracy experiment results."""
    scaling_dirs = sorted(output_base.glob("vdc_paper_scaling_accuracy_*"))
    if not scaling_dirs:
        return None
    
    latest_dir = scaling_dirs[-1]
    
    results = {}
    for subdir in ["bilinear_vs_probit", "denoiser"]:
        results_path = latest_dir / subdir / "scaling_accuracy_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results[subdir] = json.load(f)
    
    return results if results else None


def load_probit_comparison_results(output_base: Path) -> Dict[str, pd.DataFrame]:
    """Load probit vs bilinear comparison results."""
    results = {}
    
    # m=64 bilinear
    m64_bilinear = output_base / "vdc_paper_diffusion_cond_m64_tuned_bilinear_20260117_020404_55563036"
    if (m64_bilinear / "results" / "model_selection.csv").exists():
        results["m64_bilinear"] = pd.read_csv(m64_bilinear / "results" / "model_selection.csv")
    
    # m=64 probit
    m64_probit = output_base / "vdc_paper_diffusion_cond_m64_tuned_probit_bilinear_20260117_021859_55563037"
    if (m64_probit / "results" / "model_selection.csv").exists():
        results["m64_probit"] = pd.read_csv(m64_probit / "results" / "model_selection.csv")
    
    # m=128 bilinear
    m128_bilinear = output_base / "vdc_paper_diffusion_cond_m128_tuned_bilinear_20260117_034629_55563038"
    if (m128_bilinear / "results" / "model_selection.csv").exists():
        results["m128_bilinear"] = pd.read_csv(m128_bilinear / "results" / "model_selection.csv")
    
    # m=128 probit
    m128_probit = output_base / "vdc_paper_diffusion_cond_m128_tuned_probit_bilinear_20260117_035057_55563039"
    if (m128_probit / "results" / "model_selection.csv").exists():
        results["m128_probit"] = pd.read_csv(m128_probit / "results" / "model_selection.csv")
    
    return results


def load_mi_results(output_base: Path) -> Dict[str, Dict]:
    """Load MI estimation benchmark results."""
    results = {}
    
    mi_dirs = {
        "ksg": "vdc_paper_mi_ksg_20260113_233916_55240330",
        "gaussian": "vdc_paper_mi_gaussian_20260114_131546_55295484",
        "infonce": "vdc_paper_mi_infonce_20260114_131547_55295487",
        "mine": "vdc_paper_mi_mine_20260114_120819_55290873",
        "nwj": "vdc_paper_mi_nwj_20260114_131552_55295489",
        "minde": "vdc_paper_mi_minde_20260113_233952_55240331",
    }
    
    for name, dirname in mi_dirs.items():
        results_path = output_base / dirname / "results" / "mi_estimation.json"
        if results_path.exists():
            with open(results_path) as f:
                results[name] = json.load(f)
    
    return results


def fig_scaling_time_vs_dim(scaling_data: Dict, output_path: Path):
    """Generate scaling time vs dimension figure."""
    records = scaling_data.get("records", [])
    if not records:
        print("No scaling records found")
        return
    
    # Aggregate by method and dimension
    methods = sorted(set(r["method"] for r in records))
    dims = sorted(set(r["d"] for r in records))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for method in methods:
        method_recs = [r for r in records if r["method"] == method]
        if not method_recs:
            continue
        
        x = []
        y = []
        for d in dims:
            d_recs = [r for r in method_recs if r["d"] == d]
            if d_recs:
                x.append(d)
                y.append(np.mean([r["fit_s"] for r in d_recs]))
        
        color = COLORS.get(method, '#333333')
        marker = MARKERS.get(method, 'o')
        label = {
            'ours': 'DCD (ours)',
            'ours_batched': 'DCD (batched)',
            'pyvine_param': 'pyvine (parametric)',
            'pyvine_nonpar': 'pyvine (TLL)',
        }.get(method, method)
        
        ax.loglog(x, y, marker=marker, color=color, label=label, markersize=6, linewidth=2)
    
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Fit time (seconds)")
    ax.set_title("Vine Copula Fitting Time vs. Dimension")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_tc_vs_dim_real(accuracy_data: Dict, output_path: Path):
    """Generate TC error vs dimension figure from real data."""
    if not accuracy_data:
        raise FileNotFoundError("No scaling-accuracy results were found under the selected output base.")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for exp_name, exp_data in accuracy_data.items():
        records = exp_data.get("records", [])
        if not records:
            continue
        
        methods = sorted(set(r["method"] for r in records))
        dims = sorted(set(r["d"] for r in records))
        
        for method in methods:
            method_recs = [r for r in records if r["method"] == method]
            if not method_recs:
                continue
            
            x = []
            y = []
            y_std = []
            for d in dims:
                d_recs = [r for r in method_recs if r["d"] == d]
                if d_recs:
                    x.append(d)
                    tc_errs = [r["tc_err"] for r in d_recs]
                    y.append(np.mean(tc_errs))
                    y_std.append(np.std(tc_errs))
            
            color = COLORS.get(method, '#333333')
            marker = MARKERS.get(method, 'o')
            label = method.replace('_', ' ').title()
            
            ax.errorbar(x, y, yerr=y_std, marker=marker, color=color, 
                       label=label, markersize=6, linewidth=2, capsize=3)
    
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("TC Error (nats)")
    ax.set_title("Total Correlation Estimation Error vs. Dimension")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_nll_vs_dim_real(accuracy_data: Dict, output_path: Path):
    """Generate NLL error vs dimension figure from real data."""
    if not accuracy_data:
        raise FileNotFoundError("No scaling-accuracy results were found under the selected output base.")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for exp_name, exp_data in accuracy_data.items():
        records = exp_data.get("records", [])
        if not records:
            continue
        
        methods = sorted(set(r["method"] for r in records))
        dims = sorted(set(r["d"] for r in records))
        
        for method in methods:
            method_recs = [r for r in records if r["method"] == method]
            if not method_recs:
                continue
            
            x = []
            y = []
            y_std = []
            for d in dims:
                d_recs = [r for r in method_recs if r["d"] == d]
                if d_recs:
                    x.append(d)
                    nll_errs = [r["nll_err"] for r in d_recs]
                    y.append(np.mean(nll_errs))
                    y_std.append(np.std(nll_errs))
            
            color = COLORS.get(method, '#333333')
            marker = MARKERS.get(method, 'o')
            label = method.replace('_', ' ').title()
            
            ax.errorbar(x, y, yerr=y_std, marker=marker, color=color,
                       label=label, markersize=6, linewidth=2, capsize=3)
    
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("NLL Error (nats)")
    ax.set_title("Negative Log-Likelihood Error vs. Dimension")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_probit_comparison(probit_data: Dict[str, pd.DataFrame], output_path: Path):
    """Generate probit vs non-probit comparison figure."""
    if not probit_data:
        print("No probit comparison data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: m=64 comparison
    ax = axes[0]
    if "m64_bilinear" in probit_data and "m64_probit" in probit_data:
        df_bi = probit_data["m64_bilinear"]
        df_pr = probit_data["m64_probit"]
        
        copulas = df_bi["name"].tolist()
        ise_bi = df_bi["ise"].values
        ise_pr = df_pr["ise"].values
        
        x = np.arange(len(copulas))
        width = 0.35
        
        ax.bar(x - width/2, ise_bi, width, label='Bilinear (no probit)', color=COLORS['ours'])
        ax.bar(x + width/2, ise_pr, width, label='Probit', color=COLORS['ours_probit'])
        
        ax.set_ylabel('ISE')
        ax.set_title('Grid Size m=64')
        ax.set_xticks(x)
        ax.set_xticklabels([c[:8] for c in copulas], rotation=45, ha='right')
        ax.legend()
        ax.set_yscale('log')
    
    # Right: m=128 comparison
    ax = axes[1]
    if "m128_bilinear" in probit_data and "m128_probit" in probit_data:
        df_bi = probit_data["m128_bilinear"]
        df_pr = probit_data["m128_probit"]
        
        copulas = df_bi["name"].tolist()
        ise_bi = df_bi["ise"].values
        ise_pr = df_pr["ise"].values
        
        x = np.arange(len(copulas))
        width = 0.35
        
        ax.bar(x - width/2, ise_bi, width, label='Bilinear (no probit)', color=COLORS['ours'])
        ax.bar(x + width/2, ise_pr, width, label='Probit', color=COLORS['ours_probit'])
        
        ax.set_ylabel('ISE')
        ax.set_title('Grid Size m=128')
        ax.set_xticks(x)
        ax.set_xticklabels([c[:8] for c in copulas], rotation=45, ha='right')
        ax.legend()
        ax.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_probit_summary(probit_data: Dict[str, pd.DataFrame], output_path: Path):
    """Generate summary probit comparison bar chart."""
    if not probit_data:
        print("No probit comparison data found")
        return
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    configs = []
    mean_ise_bi = []
    mean_ise_pr = []
    
    for (m, name_bi, name_pr) in [
        (64, "m64_bilinear", "m64_probit"),
        (128, "m128_bilinear", "m128_probit"),
    ]:
        if name_bi in probit_data and name_pr in probit_data:
            configs.append(f"m={m}")
            mean_ise_bi.append(probit_data[name_bi]["ise"].mean())
            mean_ise_pr.append(probit_data[name_pr]["ise"].mean())
    
    if not configs:
        return
    
    x = np.arange(len(configs))
    width = 0.35
    
    ax.bar(x - width/2, mean_ise_bi, width, label='Bilinear (no probit)', color=COLORS['ours'])
    ax.bar(x + width/2, mean_ise_pr, width, label='Probit', color=COLORS['ours_probit'])
    
    ax.set_ylabel('Mean ISE')
    ax.set_title('Probit vs. Bilinear Transformation')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bi, pr) in enumerate(zip(mean_ise_bi, mean_ise_pr)):
        ax.annotate(f'{bi:.1e}', xy=(i - width/2, bi), ha='center', va='bottom', fontsize=8)
        ax.annotate(f'{pr:.1e}', xy=(i + width/2, pr), ha='center', va='bottom', fontsize=8)
    
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def fig_mi_comparison(mi_data: Dict[str, Dict], output_path: Path):
    """Generate MI estimation comparison figure."""
    if not mi_data:
        print("No MI data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Aggregate by method
    methods = list(mi_data.keys())
    mean_errs = []
    mean_times = []
    
    for method in methods:
        records = mi_data[method].get("records", [])
        if records:
            errs = [r["mi_err"] for r in records]
            times = [r["time_s"] for r in records]
            mean_errs.append(np.mean(errs))
            mean_times.append(np.mean(times))
        else:
            mean_errs.append(np.nan)
            mean_times.append(np.nan)
    
    # Left: MI error
    ax = axes[0]
    colors = [COLORS.get(m, '#333333') for m in methods]
    ax.bar(methods, mean_errs, color=colors)
    ax.set_ylabel('Mean MI Error (nats)')
    ax.set_title('MI Estimation Error by Method')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    
    # Right: Time
    ax = axes[1]
    ax.bar(methods, mean_times, color=colors)
    ax.set_ylabel('Mean Time (seconds)')
    ax.set_title('MI Estimation Time by Method')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from real experiment results")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output_base",
        type=str,
        default=os.environ.get("OUTPUT_BASE", str(repo_root / "results")),
        help="Base output directory with experiment results",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default=None,
        help="Directory to save figures (defaults to <output_base>/figures)",
    )
    args = parser.parse_args()
    
    output_base = Path(args.output_base)
    figure_dir = Path(args.figure_dir) if args.figure_dir else (output_base / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating figures from real experiment results")
    print("="*60)
    print(f"Output base: {output_base}")
    print(f"Figure dir: {figure_dir}")
    print()
    
    # Load results
    print("Loading results...")
    scaling_data = load_scaling_results(output_base)
    accuracy_data = load_scaling_accuracy_results(output_base)
    probit_data = load_probit_comparison_results(output_base)
    mi_data = load_mi_results(output_base)
    
    print(f"  Scaling data: {'found' if scaling_data else 'not found'}")
    print(f"  Accuracy data: {'found' if accuracy_data else 'not found'}")
    print(f"  Probit data: {len(probit_data)} configs found")
    print(f"  MI data: {len(mi_data)} methods found")
    print()
    
    # Generate figures
    print("Generating figures...")
    
    if scaling_data:
        fig_scaling_time_vs_dim(scaling_data, figure_dir / "scaling_time_vs_d.pdf")
    
    fig_tc_vs_dim_real(accuracy_data, figure_dir / "fig_tc_vs_dim.pdf")
    fig_nll_vs_dim_real(accuracy_data, figure_dir / "fig_nll_vs_dim.pdf")
    
    if probit_data:
        fig_probit_comparison(probit_data, figure_dir / "fig_probit_comparison.pdf")
        fig_probit_summary(probit_data, figure_dir / "fig_probit_summary.pdf")
    
    if mi_data:
        fig_mi_comparison(mi_data, figure_dir / "fig_mi_comparison.pdf")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
