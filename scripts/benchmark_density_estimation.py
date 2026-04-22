#!/usr/bin/env python
"""
Benchmark for copula density estimation.

Compares diffusion-based approach with:
1. Kernel Density Estimation (KDE) with various kernels
2. Parametric copula fitting (if pyvinecopulib available)
3. Beta-mixture transformation approach
4. Our model with different robustness settings

Supports both bivariate and higher-dimensional (via vines) evaluation.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
import argparse
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.models.unet_grid import GridUNet
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.models.projection import copula_project
from vdc.data.generators import sample_bicop, analytic_logpdf_grid
from vdc.data.hist import scatter_to_hist


# ============================================================================
# TEST COPULA SUITES
# ============================================================================

# Standard bivariate test suite
BIVARIATE_TEST_SUITE = [
    # Elliptical copulas
    {'family': 'gaussian', 'params': {'rho': 0.9}, 'name': 'Gaussian(ρ=0.9)'},
    {'family': 'gaussian', 'params': {'rho': 0.5}, 'name': 'Gaussian(ρ=0.5)'},
    {'family': 'gaussian', 'params': {'rho': -0.7}, 'name': 'Gaussian(ρ=-0.7)'},
    {'family': 'student', 'params': {'rho': 0.7, 'nu': 5}, 'name': 'Student-t(ρ=0.7, ν=5)'},
    {'family': 'student', 'params': {'rho': 0.7, 'nu': 3}, 'name': 'Student-t(ρ=0.7, ν=3)'},
    
    # Archimedean copulas (corner dependence)
    {'family': 'clayton', 'params': {'theta': 2.0}, 'name': 'Clayton(θ=2.0)'},
    {'family': 'clayton', 'params': {'theta': 5.0}, 'name': 'Clayton(θ=5.0)'},
    {'family': 'gumbel', 'params': {'theta': 2.5}, 'name': 'Gumbel(θ=2.5)'},
    {'family': 'gumbel', 'params': {'theta': 4.0}, 'name': 'Gumbel(θ=4.0)'},
    {'family': 'frank', 'params': {'theta': 5.0}, 'name': 'Frank(θ=5.0)'},
    {'family': 'frank', 'params': {'theta': -5.0}, 'name': 'Frank(θ=-5.0)'},
    {'family': 'joe', 'params': {'theta': 3.0}, 'name': 'Joe(θ=3.0)'},
    
    # Rotated versions (test rotation invariance)
    {'family': 'clayton', 'params': {'theta': 3.0}, 'name': 'Clayton(θ=3.0, 90°)', 'rotation': 90},
    {'family': 'clayton', 'params': {'theta': 3.0}, 'name': 'Clayton(θ=3.0, 180°)', 'rotation': 180},
    {'family': 'gumbel', 'params': {'theta': 2.5}, 'name': 'Gumbel(θ=2.5, 270°)', 'rotation': 270},
]

# Challenging cases (edge cases, mixtures)
CHALLENGING_TEST_SUITE = [
    # Near independence
    {'family': 'gaussian', 'params': {'rho': 0.1}, 'name': 'Gaussian(ρ=0.1)'},
    {'family': 'frank', 'params': {'theta': 0.5}, 'name': 'Frank(θ=0.5)'},
    
    # Very strong dependence
    {'family': 'gaussian', 'params': {'rho': 0.95}, 'name': 'Gaussian(ρ=0.95)'},
    {'family': 'clayton', 'params': {'theta': 10.0}, 'name': 'Clayton(θ=10.0)'},
    {'family': 'gumbel', 'params': {'theta': 6.0}, 'name': 'Gumbel(θ=6.0)'},
    
    # Heavy tails (low df student-t)
    {'family': 'student', 'params': {'rho': 0.5, 'nu': 2}, 'name': 'Student-t(ρ=0.5, ν=2)'},
]


# ============================================================================
# BASELINE METHODS
# ============================================================================

class BaseDensityEstimator:
    """Base class for density estimators."""
    
    def __init__(self, name: str):
        self.name = name
    
    def fit(self, samples: np.ndarray) -> None:
        """Fit the estimator to samples."""
        raise NotImplementedError
    
    def evaluate(self, m: int) -> np.ndarray:
        """Evaluate density on m×m grid."""
        raise NotImplementedError


class KDEEstimator(BaseDensityEstimator):
    """Kernel Density Estimation with various options."""
    
    def __init__(self, bandwidth: str = 'scott', transform: str = 'none'):
        """
        Args:
            bandwidth: 'scott', 'silverman', or float
            transform: 'none', 'logit', 'probit' (apply before KDE)
        """
        super().__init__(f'KDE-{bandwidth}-{transform}')
        self.bandwidth = bandwidth
        self.transform = transform
        self._kde = None
        self._samples = None
    
    def _transform(self, samples: np.ndarray) -> np.ndarray:
        """Transform samples before KDE."""
        eps = 1e-6
        samples = np.clip(samples, eps, 1 - eps)
        
        if self.transform == 'none':
            return samples
        elif self.transform == 'logit':
            return np.log(samples / (1 - samples))
        elif self.transform == 'probit':
            return stats.norm.ppf(samples)
        else:
            return samples
    
    def _inverse_transform_density(self, density: np.ndarray, grid_u: np.ndarray, 
                                    grid_v: np.ndarray) -> np.ndarray:
        """Transform density back to copula space."""
        if self.transform == 'none':
            return density
        elif self.transform == 'logit':
            # Jacobian: d(logit(u))/du = 1/(u(1-u))
            jacobian = 1.0 / (grid_u * (1 - grid_u) * grid_v * (1 - grid_v))
            return density * jacobian
        elif self.transform == 'probit':
            # Jacobian: d(Phi^-1(u))/du = 1/phi(Phi^-1(u))
            z_u = stats.norm.ppf(grid_u)
            z_v = stats.norm.ppf(grid_v)
            jacobian = stats.norm.pdf(z_u) * stats.norm.pdf(z_v)
            return density / (jacobian + 1e-12)
        return density
    
    def fit(self, samples: np.ndarray) -> None:
        """Fit KDE to samples."""
        self._samples = self._transform(samples)
        try:
            self._kde = stats.gaussian_kde(self._samples.T, bw_method=self.bandwidth)
        except Exception as e:
            print(f"KDE fit failed: {e}")
            self._kde = None
    
    def evaluate(self, m: int) -> np.ndarray:
        """Evaluate on grid."""
        if self._kde is None:
            return np.ones((m, m)) / m**2
        
        eps = 1e-6
        u = np.linspace(0.5/m, 1 - 0.5/m, m)
        U, V = np.meshgrid(u, u, indexing='ij')
        
        if self.transform == 'none':
            positions = np.vstack([U.ravel(), V.ravel()])
            density = self._kde(positions).reshape(m, m)
        else:
            # Transform grid points
            U_t = self._transform(np.stack([U.ravel(), V.ravel()], axis=1))
            positions = U_t.T
            density_transformed = self._kde(positions).reshape(m, m)
            density = self._inverse_transform_density(density_transformed, U, V)
        
        # Normalize to integrate to 1
        du = 1.0 / m
        density = np.maximum(density, 0)
        mass = density.sum() * du * du
        if mass > 0:
            density = density / mass
        
        return density


class TransformKDEEstimator(BaseDensityEstimator):
    """KDE with beta-kernel transformation (better for bounded support)."""
    
    def __init__(self, bandwidth: float = 0.1):
        super().__init__(f'BetaKDE-{bandwidth}')
        self.bandwidth = bandwidth
        self._samples = None
    
    def fit(self, samples: np.ndarray) -> None:
        self._samples = samples
    
    def _beta_kernel(self, x: np.ndarray, xi: float, h: float) -> np.ndarray:
        """Beta kernel that respects [0,1] boundary."""
        # Use local polynomial bandwidth
        a = xi / h + 1
        b = (1 - xi) / h + 1
        return stats.beta.pdf(x, a, b)
    
    def evaluate(self, m: int) -> np.ndarray:
        """Evaluate using product beta kernels."""
        u = np.linspace(0.5/m, 1 - 0.5/m, m)
        density = np.zeros((m, m))
        
        for sample in self._samples:
            for i, ui in enumerate(u):
                for j, uj in enumerate(u):
                    k_u = self._beta_kernel(sample[0], ui, self.bandwidth)
                    k_v = self._beta_kernel(sample[1], uj, self.bandwidth)
                    density[i, j] += k_u * k_v
        
        density /= len(self._samples)
        
        # Normalize
        du = 1.0 / m
        mass = density.sum() * du * du
        if mass > 0:
            density = density / mass
        
        return density


class HistogramEstimator(BaseDensityEstimator):
    """Simple histogram estimator with optional smoothing."""
    
    def __init__(self, smooth_sigma: float = 0.0):
        super().__init__(f'Histogram-σ{smooth_sigma}')
        self.smooth_sigma = smooth_sigma
        self._samples = None
    
    def fit(self, samples: np.ndarray) -> None:
        self._samples = samples
    
    def evaluate(self, m: int) -> np.ndarray:
        """Create histogram density."""
        hist = scatter_to_hist(self._samples, m, reflect=True)
        
        if self.smooth_sigma > 0:
            hist = gaussian_filter(hist, sigma=self.smooth_sigma)
        
        # Normalize
        du = 1.0 / m
        mass = hist.sum() * du * du
        if mass > 0:
            hist = hist / mass
        
        return hist


class ParametricCopulaEstimator(BaseDensityEstimator):
    """Parametric copula fitting (requires pyvinecopulib)."""
    
    def __init__(self, families: List[str] = None):
        super().__init__('Parametric')
        self.families = families or ['gaussian', 'student', 'clayton', 'gumbel', 'frank', 'joe']
        self._fitted_copula = None
        self._has_pvc = False
        
        try:
            import pyvinecopulib as pvc
            self._has_pvc = True
            self._pvc = pvc
        except ImportError:
            pass
    
    def fit(self, samples: np.ndarray) -> None:
        if not self._has_pvc:
            return
        
        try:
            # Fit best copula from allowed families
            controls = self._pvc.FitControlsBicop(family_set=self._pvc.all_families())
            self._fitted_copula = self._pvc.Bicop(samples, controls=controls)
        except Exception as e:
            print(f"Parametric fit failed: {e}")
            self._fitted_copula = None
    
    def evaluate(self, m: int) -> np.ndarray:
        if not self._has_pvc or self._fitted_copula is None:
            return np.ones((m, m)) / m**2
        
        eps = 1e-6
        u = np.linspace(0.5/m, 1 - 0.5/m, m)
        U, V = np.meshgrid(u, u, indexing='ij')
        grid = np.stack([U.ravel(), V.ravel()], axis=1)
        grid = np.clip(grid, eps, 1 - eps)
        
        try:
            log_pdf = self._fitted_copula.loglik(grid)
            density = np.exp(log_pdf / len(grid)).reshape(m, m)
        except:
            # Fallback to PDF
            density = np.exp(self._fitted_copula.pdf(grid).reshape(m, m))
        
        # Normalize
        du = 1.0 / m
        mass = density.sum() * du * du
        if mass > 0:
            density = density / mass
        
        return density


# ============================================================================
# DIFFUSION MODEL ESTIMATOR
# ============================================================================

class DiffusionCopulaEstimator(BaseDensityEstimator):
    """Our diffusion-based copula density estimator with robustness options."""
    
    def __init__(self, checkpoint_path: str, device: torch.device,
                 num_ensemble: int = 1, 
                 ensemble_mode: str = 'geometric',
                 cfg_scale: float = 2.0,
                 num_steps: int = 50,
                 adaptive_cfg: bool = False):
        name = f'Diffusion(ens={num_ensemble},{ensemble_mode})'
        super().__init__(name)
        
        self.device = device
        self.num_ensemble = num_ensemble
        self.ensemble_mode = ensemble_mode
        self.cfg_scale = cfg_scale
        self.num_steps = num_steps
        self.adaptive_cfg = adaptive_cfg
        
        # Load model
        self.model, self.diffusion, self.config = self._load_model(checkpoint_path)
        self._samples = None
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        model = GridUNet(
            m=config['data']['m'],
            in_channels=2,
            base_channels=config['model'].get('base_channels', 64),
            channel_mults=tuple(config['model'].get('channel_mults', (1, 2, 3, 4))),
            num_res_blocks=config['model'].get('num_res_blocks', 2),
            attention_resolutions=tuple(config['model'].get('attention_resolutions', (16, 8))),
            dropout=config['model'].get('dropout', 0.1),
        ).to(self.device)
        
        # Handle state dict prefixes
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('unet.'):
                new_state_dict[k[5:]] = v
            elif k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        
        diffusion = CopulaAwareDiffusion(
            timesteps=config.get('diffusion', {}).get('timesteps', 1000),
            beta_schedule=config.get('diffusion', {}).get('noise_schedule', 'cosine'),
        ).to(self.device)
        
        return model, diffusion, config
    
    def fit(self, samples: np.ndarray) -> None:
        """Store samples for conditioning."""
        self._samples = samples
    
    def _compute_adaptive_cfg(self, histogram: torch.Tensor) -> float:
        """Adaptively choose CFG based on histogram properties."""
        hist_np = histogram[0, 0].cpu().numpy()
        
        # Check corner concentration
        m = hist_np.shape[0]
        corner_size = max(2, m // 8)
        
        corners = [
            hist_np[:corner_size, :corner_size].sum(),  # bottom-left
            hist_np[-corner_size:, -corner_size:].sum(),  # top-right
            hist_np[:corner_size, -corner_size:].sum(),  # bottom-right
            hist_np[-corner_size:, :corner_size].sum(),  # top-left
        ]
        
        max_corner = max(corners)
        total_mass = hist_np.sum()
        corner_ratio = max_corner / (total_mass + 1e-12)
        
        # High corner concentration → lower CFG (avoid artifacts)
        # Smooth distribution → higher CFG (need stronger guidance)
        if corner_ratio > 0.3:
            return 1.5  # Strong corner peak
        elif corner_ratio > 0.15:
            return 2.0  # Moderate
        else:
            return 3.0  # Smooth, need more guidance
    
    @torch.no_grad()
    def _sample_single(self, histogram: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        """Single diffusion sampling."""
        m = histogram.shape[-1]
        T = self.diffusion.timesteps
        
        log_histogram = torch.log(histogram.clamp(min=1e-12))
        x_t = torch.randn(1, 1, m, m, device=self.device)
        
        step_size = max(1, T // self.num_steps)
        timesteps = list(range(T - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)
        
        alphas_cumprod = self.diffusion.alphas_cumprod.to(self.device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
            t_normalized = t_tensor.float() / T
            
            # CFG sampling
            model_input_cond = torch.cat([x_t, log_histogram], dim=1)
            pred_noise_cond = self.model(model_input_cond, t_normalized)
            
            model_input_uncond = torch.cat([x_t, torch.zeros_like(log_histogram)], dim=1)
            pred_noise_uncond = self.model(model_input_uncond, t_normalized)
            
            pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
            
            alpha_t = alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
            pred_x0 = pred_x0.clamp(-20, 20)
            
            if t == 0:
                x_t = pred_x0
            else:
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                alpha_t_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)
                dir_xt = torch.sqrt(1 - alpha_t_prev) * pred_noise
                x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        
        return x_t
    
    def evaluate(self, m: int) -> np.ndarray:
        """Evaluate with optional ensemble averaging."""
        # Create histogram conditioning
        hist = scatter_to_hist(self._samples, m, reflect=True)
        du = 1.0 / m
        hist = hist / (hist.sum() * du * du + 1e-12)
        histogram = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Determine CFG scale
        cfg_scale = self._compute_adaptive_cfg(histogram) if self.adaptive_cfg else self.cfg_scale
        
        # Ensemble sampling
        if self.num_ensemble == 1:
            log_density = self._sample_single(histogram, cfg_scale)
            density = torch.exp(log_density).clamp(1e-12, 1e6)
        else:
            densities = []
            for _ in range(self.num_ensemble):
                log_density = self._sample_single(histogram, cfg_scale)
                densities.append(log_density)
            
            if self.ensemble_mode == 'geometric':
                # Average in log space
                log_density_avg = torch.stack(densities).mean(dim=0)
                density = torch.exp(log_density_avg).clamp(1e-12, 1e6)
            elif self.ensemble_mode == 'arithmetic':
                # Average in linear space
                density = torch.stack([torch.exp(d) for d in densities]).mean(dim=0).clamp(1e-12, 1e6)
            elif self.ensemble_mode == 'median':
                # Median in linear space
                density = torch.stack([torch.exp(d) for d in densities]).median(dim=0)[0].clamp(1e-12, 1e6)
            else:
                density = torch.exp(densities[0]).clamp(1e-12, 1e6)
        
        # Normalize
        mass = (density * du * du).sum()
        density = density / mass
        
        # Apply copula projection
        density = copula_project(density, iters=50)
        
        return density[0, 0].cpu().numpy()


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_ise(pred: np.ndarray, true: np.ndarray, m: int) -> float:
    """Integrated Squared Error."""
    du = 1.0 / m
    return np.mean((pred - true) ** 2) * du * du


def compute_mae(pred: np.ndarray, true: np.ndarray, m: int) -> float:
    """Mean Absolute Error."""
    du = 1.0 / m
    return np.mean(np.abs(pred - true)) * du * du


def compute_kl_divergence(pred: np.ndarray, true: np.ndarray, m: int) -> float:
    """KL divergence D_KL(true || pred)."""
    eps = 1e-12
    du = 1.0 / m
    pred_safe = np.maximum(pred, eps)
    true_safe = np.maximum(true, eps)
    return np.sum(true_safe * np.log(true_safe / pred_safe)) * du * du


def compute_hellinger(pred: np.ndarray, true: np.ndarray, m: int) -> float:
    """Hellinger distance."""
    du = 1.0 / m
    return np.sqrt(0.5 * np.sum((np.sqrt(pred) - np.sqrt(true)) ** 2) * du * du)


def compute_marginal_error(pred: np.ndarray, m: int) -> Tuple[float, float]:
    """Check if marginals are uniform."""
    du = 1.0 / m
    marginal_u = pred.sum(axis=1) * du
    marginal_v = pred.sum(axis=0) * du
    
    uniform = np.ones(m) / m
    err_u = np.mean((marginal_u - 1.0) ** 2)
    err_v = np.mean((marginal_v - 1.0) ** 2)
    
    return err_u, err_v


def compute_tail_dependence_error(pred: np.ndarray, true: np.ndarray, m: int, 
                                   tail_region: float = 0.1) -> float:
    """Error in tail regions."""
    tail_idx = int(tail_region * m)
    
    # Lower tail (near origin)
    lower_pred = pred[:tail_idx, :tail_idx]
    lower_true = true[:tail_idx, :tail_idx]
    
    # Upper tail (near (1,1))
    upper_pred = pred[-tail_idx:, -tail_idx:]
    upper_true = true[-tail_idx:, -tail_idx:]
    
    lower_err = np.mean((lower_pred - lower_true) ** 2)
    upper_err = np.mean((upper_pred - upper_true) ** 2)
    
    return 0.5 * (lower_err + upper_err)


def compute_correlation(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson correlation between predicted and true densities."""
    return np.corrcoef(pred.ravel(), true.ravel())[0, 1]


def evaluate_estimator(estimator: BaseDensityEstimator, 
                       samples: np.ndarray,
                       true_density: np.ndarray,
                       m: int) -> Dict:
    """Evaluate one estimator on the configured benchmark suite."""
    estimator.fit(samples)
    pred_density = estimator.evaluate(m)
    
    # Compute all metrics
    ise = compute_ise(pred_density, true_density, m)
    mae = compute_mae(pred_density, true_density, m)
    kl = compute_kl_divergence(pred_density, true_density, m)
    hellinger = compute_hellinger(pred_density, true_density, m)
    corr = compute_correlation(pred_density, true_density)
    marg_u, marg_v = compute_marginal_error(pred_density, m)
    tail_err = compute_tail_dependence_error(pred_density, true_density, m)
    
    return {
        'ise': float(ise),
        'mae': float(mae),
        'kl_divergence': float(kl),
        'hellinger': float(hellinger),
        'correlation': float(corr),
        'marginal_u_error': float(marg_u),
        'marginal_v_error': float(marg_v),
        'tail_error': float(tail_err),
        'pred_density': pred_density,
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(estimators: List[BaseDensityEstimator],
                  test_suite: List[Dict],
                  n_samples: int = 2000,
                  m: int = 64,
                  n_trials: int = 1,
                  output_dir: Optional[Path] = None) -> Dict:
    """Run full benchmark suite."""
    
    results = {est.name: [] for est in estimators}
    
    for copula_spec in test_suite:
        family = copula_spec['family']
        params = copula_spec['params']
        name = copula_spec['name']
        rotation = copula_spec.get('rotation', 0)
        
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        # Get true density
        true_log_density = analytic_logpdf_grid(family, params, m=m, rotation=rotation)
        true_density = np.exp(np.clip(true_log_density, -20, 20))
        du = 1.0 / m
        true_density = true_density / (true_density.sum() * du * du)
        
        # Apply projection to true density for fair comparison
        true_density_t = torch.from_numpy(true_density).float().unsqueeze(0).unsqueeze(0)
        true_density_t = copula_project(true_density_t, iters=50)
        true_density = true_density_t[0, 0].numpy()
        
        for trial in range(n_trials):
            # Generate samples
            samples = sample_bicop(family, params, n_samples, rotation=rotation)
            
            # Evaluate each estimator
            for estimator in estimators:
                try:
                    metrics = evaluate_estimator(estimator, samples, true_density, m)
                    metrics['copula'] = name
                    metrics['trial'] = trial
                    results[estimator.name].append(metrics)
                    
                    print(f"  {estimator.name}: ISE={metrics['ise']:.6f}, Corr={metrics['correlation']:.4f}")
                    
                except Exception as e:
                    print(f"  {estimator.name}: FAILED - {e}")
    
    return results


def summarize_results(results: Dict) -> None:
    """Print summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Aggregate by method
    summary = {}
    for method, method_results in results.items():
        if not method_results:
            continue
        
        ise_values = [r['ise'] for r in method_results]
        corr_values = [r['correlation'] for r in method_results]
        
        summary[method] = {
            'mean_ise': np.mean(ise_values),
            'std_ise': np.std(ise_values),
            'median_ise': np.median(ise_values),
            'mean_corr': np.mean(corr_values),
        }
    
    # Sort by mean ISE
    sorted_methods = sorted(summary.items(), key=lambda x: x[1]['mean_ise'])
    
    print(f"\n{'Method':<35} {'Mean ISE':<12} {'Std ISE':<12} {'Median ISE':<12} {'Mean Corr':<12}")
    print("-" * 85)
    
    for method, stats in sorted_methods:
        print(f"{method:<35} {stats['mean_ise']:<12.6f} {stats['std_ise']:<12.6f} "
              f"{stats['median_ise']:<12.6f} {stats['mean_corr']:<12.4f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark copula density estimation methods')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to diffusion model checkpoint')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of samples per copula')
    parser.add_argument('--m', type=int, default=64,
                        help='Grid resolution')
    parser.add_argument('--n-trials', type=int, default=1,
                        help='Number of trials per copula')
    parser.add_argument('--output', type=str, default='results/benchmark',
                        help='Output directory')
    parser.add_argument('--suite', type=str, choices=['standard', 'challenging', 'all'],
                        default='standard', help='Test suite to run')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Select test suite
    if args.suite == 'standard':
        test_suite = BIVARIATE_TEST_SUITE
    elif args.suite == 'challenging':
        test_suite = CHALLENGING_TEST_SUITE
    else:
        test_suite = BIVARIATE_TEST_SUITE + CHALLENGING_TEST_SUITE
    
    # Create estimators
    estimators = [
        # Baselines
        KDEEstimator(bandwidth='scott', transform='none'),
        KDEEstimator(bandwidth='scott', transform='probit'),
        KDEEstimator(bandwidth='silverman', transform='none'),
        HistogramEstimator(smooth_sigma=0.0),
        HistogramEstimator(smooth_sigma=1.0),
        HistogramEstimator(smooth_sigma=2.0),
        ParametricCopulaEstimator(),
    ]
    
    # Add diffusion model if checkpoint provided
    if args.checkpoint:
        checkpoint_path = REPO_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
        if checkpoint_path.exists():
            # Single sample
            estimators.append(DiffusionCopulaEstimator(
                str(checkpoint_path), device,
                num_ensemble=1, cfg_scale=2.0
            ))
            
            # Ensemble variants
            estimators.append(DiffusionCopulaEstimator(
                str(checkpoint_path), device,
                num_ensemble=5, ensemble_mode='geometric', cfg_scale=2.0
            ))
            
            # Adaptive CFG
            estimators.append(DiffusionCopulaEstimator(
                str(checkpoint_path), device,
                num_ensemble=1, adaptive_cfg=True
            ))
            
            # Ensemble + adaptive
            estimators.append(DiffusionCopulaEstimator(
                str(checkpoint_path), device,
                num_ensemble=3, ensemble_mode='geometric', adaptive_cfg=True
            ))
    
    # Run benchmark
    results = run_benchmark(
        estimators=estimators,
        test_suite=test_suite,
        n_samples=args.n_samples,
        m=args.m,
        n_trials=args.n_trials,
    )
    
    # Summarize
    summarize_results(results)
    
    # Save results (without density arrays)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_serializable = {}
    for method, method_results in results.items():
        results_serializable[method] = [
            {k: v for k, v in r.items() if k != 'pred_density'}
            for r in method_results
        ]
    
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
