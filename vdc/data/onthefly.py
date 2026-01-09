"""On-the-fly copula dataset extracted from legacy training script.

Generates synthetic bivariate copula samples and analytic density grids without pre-saving data.
Simplified: mixture support retained; excludes unused legacy fallback comments.

Supports probit/Gaussian copula representation when transform_to_probit_space=True:
- Returns densities in probit space with proper Jacobian correction
- Samples remain in copula space [0,1]²

Configuration:
- transform_to_probit_space: If True, apply Jacobian correction to transform densities to probit space
"""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from vdc.data.generators import sample_bicop, analytic_logpdf_grid
from vdc.data.hist import scatter_to_hist
from vdc.utils.probit_transform import copula_logdensity_to_probit_logdensity


class OnTheFlyCopulaDataset(Dataset):
    def __init__(self,
                 n_samples_per_batch: int = 10000,
                 m: int = 256,
                 families: Optional[Union[Dict[str, float], List[str]]] = None,
                 param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                 rotation_prob: float = 0.3,
                 mixture_prob: float = 0.0,
                 n_mixture_components: Tuple[int, int] = (2, 3),
                 transform_to_probit_space: bool = False,
                 seed: Optional[int] = None):
        self.n_samples = n_samples_per_batch
        self.m = m
        self.rotation_prob = rotation_prob
        self.mixture_prob = mixture_prob
        self.n_mixture_components = n_mixture_components
        self.transform_to_probit_space = transform_to_probit_space

        # Families may be passed as:
        #  - dict {family: weight}
        #  - list ["gaussian", "clayton", ...] (uniform weights)
        if families is None:
            self.families: Dict[str, float] = {
            # Elliptical (symmetric)
            'gaussian': 0.20,
            'student': 0.10,
            # Archimedean (asymmetric tails)
            'clayton': 0.15,
            'gumbel': 0.15,
            'frank': 0.10,
            'joe': 0.10,
            # Two-parameter families
            'bb1': 0.05,
            'bb7': 0.05,
            # Independence
            'independence': 0.05,
            }
        elif isinstance(families, dict):
            self.families = {str(k): float(v) for k, v in families.items()}
        else:
            fam_list = [str(f) for f in families]
            if len(fam_list) == 0:
                raise ValueError("families list cannot be empty")
            self.families = {f: 1.0 for f in fam_list}

        # Parameter ranges: accept both legacy and config-style keys.
        # Notes:
        # - We internally sample student degrees of freedom as `nu` but accept `student_df` too.
        self.param_ranges = param_ranges or {
            # Elliptical
            'gaussian_rho': [-0.95, 0.95],
            'student_rho': [-0.95, 0.95],
            'student_nu': [2.5, 30.0],  # legacy key
            'student_df': [2.5, 30.0],  # config key alias
            # Archimedean
            'clayton_theta': [0.2, 12.0],
            'gumbel_theta': [1.0, 10.0],
            'frank_theta': [-12.0, 12.0],
            'joe_theta': [1.0, 10.0],
            # BB families (two parameters)
            'bb1_theta': [0.1, 5.0],
            'bb1_delta': [1.0, 5.0],
            'bb7_theta': [1.0, 5.0],
            'bb7_delta': [0.1, 5.0],
        }
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return 10**9  # effectively infinite

    def _sample_family(self):
        fams = list(self.families.keys())
        probs = np.array(list(self.families.values()))
        probs = probs / probs.sum()  # Normalize in case weights don't sum to 1
        family = self.rng.choice(fams, p=probs)
        params = {}
        
        if family == 'gaussian':
            lo, hi = self.param_ranges.get('gaussian_rho', [-0.95, 0.95])
            params['rho'] = self.rng.uniform(lo, hi)
        elif family == 'student':
            lo, hi = self.param_ranges.get('student_rho', [-0.95, 0.95])
            params['rho'] = self.rng.uniform(lo, hi)
            # Support either student_nu or student_df
            if 'student_nu' in self.param_ranges:
                lo, hi = self.param_ranges.get('student_nu', [2.5, 30.0])
            else:
                lo, hi = self.param_ranges.get('student_df', [2.5, 30.0])
            params['nu'] = float(self.rng.uniform(lo, hi))
        elif family in ['clayton', 'gumbel', 'frank', 'joe']:
            key = f"{family}_theta"
            lo, hi = self.param_ranges.get(key, [1.0, 5.0])
            params['theta'] = self.rng.uniform(lo, hi)
        elif family in ['bb1', 'bb6', 'bb7', 'bb8']:
            # Two-parameter families
            theta_key = f"{family}_theta"
            delta_key = f"{family}_delta"
            lo, hi = self.param_ranges.get(theta_key, [1.0, 5.0])
            params['theta'] = self.rng.uniform(lo, hi)
            lo, hi = self.param_ranges.get(delta_key, [1.0, 5.0])
            params['delta'] = self.rng.uniform(lo, hi)
        elif family == 'independence':
            pass  # No parameters needed
        
        # Apply rotation for asymmetric copulas
        rotation = 0
        rotatable_families = ['clayton', 'gumbel', 'joe', 'bb1', 'bb6', 'bb7', 'bb8']
        if family in rotatable_families and self.rng.random() < self.rotation_prob:
            rotation = self.rng.choice([0, 90, 180, 270])
        
        return family, params, rotation

    def _generate_mixture(self):
        n_min, n_max = self.n_mixture_components
        k = self.rng.randint(n_min, n_max+1)
        weights = self.rng.dirichlet(np.ones(k))
        all_samples = []
        all_density = []
        for i, w in enumerate(weights):
            family, params, rotation = self._sample_family()
            n_k = int(w * self.n_samples)
            if i == k - 1:
                n_k = self.n_samples - sum(len(s) for s in all_samples)
            n_k = max(1, n_k)
            samples_k = sample_bicop(family, params, n_k, rotation=rotation)
            all_samples.append(samples_k)
            try:
                lg = analytic_logpdf_grid(family if family!='student' else 'student_t', params, self.m, rotation=rotation)
                # Clip log-density BEFORE exponentiation to prevent overflow
                lg = np.clip(lg, -15, 15)
                dens = np.exp(lg)
                dens = np.clip(dens, 1e-10, 1e3)
            except Exception:
                dens = np.ones((self.m,self.m))
            all_density.append(dens)
        samples = np.vstack(all_samples)
        perm = self.rng.permutation(len(samples))
        samples = samples[perm]
        density_grid = sum(w*d for w,d in zip(weights, all_density))
        # Normalize mixture density to integrate to 1
        du = dv = 1.0 / self.m
        density_grid = density_grid / (density_grid.sum() * du * dv)
        # Clip mixture density safely (wider range to preserve peaks)
        density_grid = np.clip(density_grid, 1e-12, 1e6)
        return samples, density_grid

    def __getitem__(self, idx):
        if self.rng.random() < self.mixture_prob:
            samples, density_grid = self._generate_mixture()
        else:
            family, params, rotation = self._sample_family()
            samples = sample_bicop(family, params, self.n_samples, rotation=rotation)
            try:
                lg = analytic_logpdf_grid(family if family!='student' else 'student_t', params, self.m, rotation=rotation)
                # Clip log-density BEFORE exponentiation to prevent overflow
                # log(1e6) ≈ 13.8, so clamp to [-20, 20] gives density range [2e-9, 4.8e8]
                # Use wider range to preserve natural peak structure
                lg = np.clip(lg, -20, 20)
                density_grid = np.exp(lg)
                # Additional safety clip (should rarely trigger after log clipping)
                density_grid = np.clip(density_grid, 1e-12, 1e6)
            except Exception:
                density_grid = scatter_to_hist(samples, self.m, reflect=True)
        
        # CRITICAL: Normalize so integral = 1 (proper copula density)
        du = dv = 1.0 / self.m
        density_grid = density_grid / (density_grid.sum() * du * dv)
        
        # Convert to torch tensors
        density_tensor = torch.from_numpy(density_grid).float().unsqueeze(0)  # (1, m, m)
        
        # Safety check for NaN/Inf before returning
        if torch.isnan(density_tensor).any() or torch.isinf(density_tensor).any():
            # Fallback to uniform density if preprocessing fails
            print(f"Warning: NaN/Inf detected in generated density, using uniform fallback")
            density_tensor = torch.ones(1, self.m, self.m)
        
        # Transform to probit space if requested (using log-space for numerical stability)
        if self.transform_to_probit_space:
            # Convert to log-density
            log_density_copula = torch.log(density_tensor.clamp(min=1e-10))
            # Transform to probit space in log-domain
            log_density_probit = copula_logdensity_to_probit_logdensity(log_density_copula, self.m)
            # Return log-density for stable training
            density_tensor = log_density_probit
        
        return {
            'samples': torch.from_numpy(samples).float(),
            'density': density_tensor,
            'is_log_density': bool(self.transform_to_probit_space)  # Python bool, not tensor
        }


