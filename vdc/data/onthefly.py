"""On-the-fly copula dataset extracted from legacy training script.

Generates synthetic bivariate copula samples and analytic density grids without pre-saving data.
Simplified: mixture support retained; excludes unused legacy fallback comments.

Supports probit/Gaussian copula representation when transform_to_probit_space=True:
- Returns densities in probit space with proper Jacobian correction
- Samples remain in copula space [0,1]²

Configuration:
- transform_to_probit_space: If True, apply Jacobian correction to transform densities to probit space
"""
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from vdc.data.generators import sample_bicop, analytic_logpdf_grid
from vdc.data.complex_copulas import complex_copula_density_grid
from vdc.inference.density import scatter_to_hist
from vdc.utils.probit_transform import copula_logdensity_to_probit_logdensity
from vdc.vine.copula_diffusion import DiffusionCopulaModel


class OnTheFlyCopulaDataset(Dataset):
    def __init__(self,
                 n_samples_per_batch: Union[int, List[int], Dict[str, Any]] = 10000,
                 m: int = 256,
                 families: Optional[Union[Dict[str, float], List[str]]] = None,
                 param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                 rotation_prob: float = 0.3,
                 mixture_prob: float = 0.0,
                 n_mixture_components: Tuple[int, int] = (2, 3),
                 transform_to_probit_space: bool = False,
                 seed: Optional[int] = None):
        # Sample size spec:
        # - int: fixed n
        # - list[int]: choose uniformly from the list each example
        # - dict: {mode: "log_uniform"|"uniform"|"choices", min/max/choices}
        self._n_spec: Union[int, List[int], Dict[str, Any]] = n_samples_per_batch
        self._n_mode: str = "fixed"
        self._n_fixed: Optional[int] = None
        self._n_choices: Optional[List[int]] = None
        self._n_min: Optional[int] = None
        self._n_max: Optional[int] = None
        self._parse_n_spec(self._n_spec)

        # Only return raw samples when n is fixed; otherwise DataLoader collation
        # will fail (variable-length tensors).
        self._return_samples: bool = (self._n_mode == "fixed")
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
        # IMPORTANT: DataLoader with num_workers>0 will copy the Dataset object into
        # multiple worker processes. If we keep a fixed RandomState here, workers
        # can end up generating highly correlated / duplicated batches.
        #
        # We therefore keep only a base seed and lazily construct a per-worker RNG.
        self._base_seed = None if seed is None else int(seed)
        self._rng: Optional[np.random.RandomState] = None

    def _get_rng(self) -> np.random.RandomState:
        """Return a per-worker RNG to avoid duplicated synthetic batches across DataLoader workers."""
        if self._rng is not None:
            return self._rng
        # If DataLoader uses workers, each worker process has a unique worker id.
        try:
            worker_info = torch.utils.data.get_worker_info()
        except Exception:
            worker_info = None
        worker_id = int(worker_info.id) if worker_info is not None else 0
        # If no explicit seed was provided, fall back to a random base seed.
        # (This path is typically not used in training, since we pass a seed.)
        base = self._base_seed
        if base is None:
            base = int(np.random.randint(0, 2**31 - 1))
        # Large stride so different workers don't overlap early.
        seed = base + 100000 * worker_id
        self._rng = np.random.RandomState(seed)
        return self._rng

    def _parse_n_spec(self, spec: Union[int, List[int], Dict[str, Any]]) -> None:
        if isinstance(spec, (int, np.integer)):
            n = int(spec)
            if n <= 0:
                raise ValueError(f"n_samples_per_batch must be positive, got {spec}")
            self._n_mode = "fixed"
            self._n_fixed = n
            return
        if isinstance(spec, (list, tuple)):
            choices = [int(x) for x in spec]
            if not choices:
                raise ValueError("n_samples_per_batch list cannot be empty")
            if any(n <= 0 for n in choices):
                raise ValueError(f"n_samples_per_batch choices must be positive, got {choices}")
            self._n_mode = "choices"
            self._n_choices = choices
            return
        if isinstance(spec, dict):
            mode = str(spec.get("mode", "log_uniform"))
            self._n_mode = mode
            if mode == "choices":
                choices_raw = spec.get("choices", None)
                if not isinstance(choices_raw, (list, tuple)) or len(choices_raw) == 0:
                    raise ValueError("n_samples_per_batch spec mode=choices requires non-empty 'choices' list")
                self._n_choices = [int(x) for x in choices_raw]
                if any(n <= 0 for n in self._n_choices):
                    raise ValueError(f"n_samples_per_batch choices must be positive, got {self._n_choices}")
                return
            # Range-based modes
            n_min = int(spec.get("min", spec.get("lo", 200)))
            n_max = int(spec.get("max", spec.get("hi", 5000)))
            if n_min <= 0 or n_max <= 0 or n_min > n_max:
                raise ValueError(f"Invalid n_samples_per_batch range: min={n_min}, max={n_max}")
            self._n_min = n_min
            self._n_max = n_max
            if mode not in ("log_uniform", "uniform"):
                raise ValueError(f"Unknown n_samples_per_batch mode '{mode}' (expected log_uniform|uniform|choices)")
            return
        raise TypeError(f"Unsupported type for n_samples_per_batch: {type(spec)}")

    def _sample_n(self) -> int:
        rng = self._get_rng()
        if self._n_mode == "fixed":
            assert self._n_fixed is not None
            return int(self._n_fixed)
        if self._n_mode == "choices":
            assert self._n_choices is not None
            return int(rng.choice(self._n_choices))
        assert self._n_min is not None and self._n_max is not None
        if self._n_mode == "uniform":
            return int(rng.randint(self._n_min, self._n_max + 1))
        # log-uniform: sample in log-space to cover many decades fairly
        lo = float(np.log(self._n_min))
        hi = float(np.log(self._n_max))
        n = int(np.round(np.exp(rng.uniform(lo, hi))))
        return int(max(self._n_min, min(self._n_max, n)))

    def __len__(self):
        return 10**9  # effectively infinite

    def _sample_family(self):
        rng = self._get_rng()
        fams = list(self.families.keys())
        probs = np.array(list(self.families.values()))
        probs = probs / probs.sum()  # Normalize in case weights don't sum to 1
        family = rng.choice(fams, p=probs)
        params = {}

        # ---------------------------------------------------------------------
        # Complex synthetic families (optional; only used if included in config)
        #
        # Supported naming:
        #   - "complex:x", "complex:ring", "complex:double_banana"
        #   - "complex_x", "complex_ring", "complex_double_banana"
        #   - "complex" (randomly chooses one of the above kinds)
        # ---------------------------------------------------------------------
        if str(family).startswith("complex"):
            fam_str = str(family)
            if fam_str == "complex":
                kind = str(rng.choice(["x", "ring", "double_banana"]))
            elif fam_str.startswith("complex:"):
                kind = fam_str.split("complex:", 1)[1]
            elif fam_str.startswith("complex_"):
                kind = fam_str.split("complex_", 1)[1]
            else:
                kind = fam_str
            kind = str(kind).lower().strip()

            # Sample a small range of shape parameters.
            if kind in {"x", "xshape", "x_shape"}:
                params = {
                    "sigma": float(rng.uniform(0.02, 0.06)),
                    "w2": float(rng.uniform(0.5, 2.0)),
                }
                family = "complex:x"
            elif kind in {"ring", "o", "circle"}:
                params = {
                    "r0": float(rng.uniform(0.20, 0.36)),
                    "sigma": float(rng.uniform(0.02, 0.06)),
                }
                family = "complex:ring"
            elif kind in {"double_banana", "doublebanana", "banana2"}:
                params = {
                    "amp": float(rng.uniform(0.10, 0.22)),
                    "offset": float(rng.uniform(0.08, 0.25)),
                    "sigma": float(rng.uniform(0.02, 0.06)),
                }
                family = "complex:double_banana"
            else:
                raise ValueError(f"Unknown complex family kind '{kind}' from '{fam_str}'")

            # Rotation is not used for these grid-constructed copulas.
            rotation = 0
            return family, params, rotation
        
        if family == 'gaussian':
            lo, hi = self.param_ranges.get('gaussian_rho', [-0.95, 0.95])
            params['rho'] = rng.uniform(lo, hi)
        elif family == 'student':
            lo, hi = self.param_ranges.get('student_rho', [-0.95, 0.95])
            params['rho'] = rng.uniform(lo, hi)
            # Support either student_nu or student_df
            if 'student_nu' in self.param_ranges:
                lo, hi = self.param_ranges.get('student_nu', [2.5, 30.0])
            else:
                lo, hi = self.param_ranges.get('student_df', [2.5, 30.0])
            params['nu'] = float(rng.uniform(lo, hi))
        elif family in ['clayton', 'gumbel', 'frank', 'joe']:
            key = f"{family}_theta"
            lo, hi = self.param_ranges.get(key, [1.0, 5.0])
            params['theta'] = rng.uniform(lo, hi)
        elif family in ['bb1', 'bb6', 'bb7', 'bb8']:
            # Two-parameter families
            theta_key = f"{family}_theta"
            delta_key = f"{family}_delta"
            lo, hi = self.param_ranges.get(theta_key, [1.0, 5.0])
            params['theta'] = rng.uniform(lo, hi)
            lo, hi = self.param_ranges.get(delta_key, [1.0, 5.0])
            params['delta'] = rng.uniform(lo, hi)
        elif family == 'independence':
            pass  # No parameters needed
        
        # Apply rotation for asymmetric copulas
        rotation = 0
        rotatable_families = ['clayton', 'gumbel', 'joe', 'bb1', 'bb6', 'bb7', 'bb8']
        if family in rotatable_families and rng.random() < self.rotation_prob:
            rotation = rng.choice([0, 90, 180, 270])
        
        return family, params, rotation

    def _generate_mixture(self, n_total: int):
        rng = self._get_rng()
        n_min, n_max = self.n_mixture_components
        k = rng.randint(n_min, n_max+1)
        weights = rng.dirichlet(np.ones(k))
        all_samples = []
        all_density = []
        for i, w in enumerate(weights):
            # Avoid complex families inside mixtures by default (they're expensive to sample
            # and mixtures are primarily meant for parametric augmentation).
            for _attempt in range(50):
                family, params, rotation = self._sample_family()
                if not str(family).startswith("complex"):
                    break
            n_k = int(w * n_total)
            if i == k - 1:
                n_k = n_total - sum(len(s) for s in all_samples)
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
        perm = rng.permutation(len(samples))
        samples = samples[perm]
        density_grid = sum(w*d for w,d in zip(weights, all_density))
        # Normalize mixture density to integrate to 1
        du = dv = 1.0 / self.m
        density_grid = density_grid / (density_grid.sum() * du * dv)
        # Clip mixture density safely (wider range to preserve peaks)
        density_grid = np.clip(density_grid, 1e-12, 1e6)
        return samples, density_grid

    def __getitem__(self, idx):
        rng = self._get_rng()
        n = self._sample_n()
        if rng.random() < self.mixture_prob:
            samples, density_grid = self._generate_mixture(n_total=n)
        else:
            family, params, rotation = self._sample_family()
            if str(family).startswith("complex:"):
                # Build projected copula density and sample from it.
                kind = str(family).split("complex:", 1)[1]
                density_grid = complex_copula_density_grid(kind, params, m=self.m, device=torch.device("cpu"), projection_iters=80)
                seed = int(rng.randint(0, 2**31 - 1))
                samples = DiffusionCopulaModel.sample_from_density(
                    density=density_grid,
                    n_samples=int(n),
                    rng=np.random.default_rng(seed),
                )
            else:
                samples = sample_bicop(family, params, n, rotation=rotation)
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
        
        # Conditioning histogram (density integrating to 1)
        hist_grid = scatter_to_hist(samples, m=self.m, reflect=True)
        hist_tensor = torch.from_numpy(hist_grid).float().unsqueeze(0)  # (1, m, m)
        if torch.isnan(hist_tensor).any() or torch.isinf(hist_tensor).any():
            hist_tensor = torch.ones(1, self.m, self.m)
        du = dv = 1.0 / self.m
        hist_tensor = hist_tensor / ((hist_tensor * du * dv).sum().clamp_min(1e-12))
        
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
        
        out = {
            'density': density_tensor,
            'hist': hist_tensor,
            'log_n': torch.tensor(float(np.log(n)), dtype=torch.float32),
            'n': int(n),
            'is_log_density': bool(self.transform_to_probit_space)  # Python bool, not tensor
        }
        if self._return_samples:
            out['samples'] = torch.from_numpy(samples).float()
        return out


