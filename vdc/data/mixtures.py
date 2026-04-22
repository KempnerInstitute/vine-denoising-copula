"""
Mixed copula generation for training data augmentation.

Generates mixtures of parametric copulas to expand the diversity
of training samples beyond single-family copulas.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import h5py
from pathlib import Path
from dataclasses import dataclass
import pyvinecopulib as pv

from vdc.data.generators import sample_bicop, compute_copula_density


@dataclass
class MixtureComponent:
    """Single component in a copula mixture."""
    family: str
    params: Dict[str, float]
    rotation: int
    weight: float  # Mixture weight (sum to 1)


class MixtureCopula:
    """
    Mixture of parametric copulas: C(u,v) = Σ w_i C_i(u,v)
    
    This creates more complex, multi-modal densities for training.
    """
    
    def __init__(self, components: List[MixtureComponent]):
        """
        Args:
            components: List of mixture components with weights summing to 1
        """
        self.components = components
        
        # Normalize weights
        total_weight = sum(c.weight for c in components)
        for c in self.components:
            c.weight /= total_weight
        
        self.n_components = len(components)
    
    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample from mixture copula.
        
        Algorithm:
        1. For each sample, choose component i with probability w_i
        2. Sample from C_i
        
        Args:
            n: Number of samples
            seed: Random seed
            
        Returns:
            Array (n, 2) of pseudo-observations
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Choose components for each sample
        weights = np.array([c.weight for c in self.components])
        component_indices = np.random.choice(
            self.n_components,
            size=n,
            p=weights
        )
        
        # Sample from each component
        samples = np.zeros((n, 2))
        
        for i, component in enumerate(self.components):
            # Indices for this component
            mask = (component_indices == i)
            n_i = mask.sum()
            
            if n_i > 0:
                # Sample from component i
                samples_i = sample_bicop(
                    family=component.family,
                    params=component.params,
                    n=n_i,
                    rotation=component.rotation,
                )
                samples[mask] = samples_i
        
        return samples
    
    def density_grid(self, m: int = 64) -> np.ndarray:
        """
        Compute mixture density on grid: c(u,v) = Σ w_i c_i(u,v)
        
        Args:
            m: Grid resolution
            
        Returns:
            Array (m, m) of density values
        """
        grid = np.linspace(0, 1, m)
        u_grid, v_grid = np.meshgrid(grid, grid, indexing='ij')
        
        # Stack for vectorized evaluation
        uv_pairs = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)
        
        # Accumulate weighted densities
        density = np.zeros(m * m)
        
        for component in self.components:
            # Compute density for this component
            c_i = compute_copula_density(
                family=component.family,
                params=component.params,
                uv=uv_pairs,
                rotation=component.rotation,
            )
            
            # Add weighted contribution
            density += component.weight * c_i
        
        return density.reshape(m, m)
    
    def log_density_grid(self, m: int = 64) -> np.ndarray:
        """Compute log-density on grid."""
        density = self.density_grid(m)
        return np.log(density + 1e-12)


def generate_random_mixture(
    n_components: int,
    families: Optional[List[str]] = None,
    tau_range: Tuple[float, float] = (-0.7, 0.7),
    seed: Optional[int] = None,
) -> MixtureCopula:
    """
    Generate a random mixture copula.
    
    Args:
        n_components: Number of components (1 to ~5 recommended)
        families: List of allowed families (None = use all)
        tau_range: Range of Kendall's tau
        seed: Random seed
        
    Returns:
        MixtureCopula instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    if families is None:
        families = [
            'gaussian', 'clayton', 'gumbel', 'frank', 't',
            'joe', 'bb1', 'bb7'
        ]
    
    # Generate random components
    components = []
    
    for i in range(n_components):
        # Random family
        family = np.random.choice(families)
        
        # Random tau and rotation
        tau = np.random.uniform(*tau_range)
        rotation = int(np.random.choice([0, 90, 180, 270]))
        
        # Convert tau to parameters
        if family == 'gaussian':
            rho = np.sin(tau * np.pi / 2)
            params = {'rho': rho}
        elif family == 't':
            rho = np.sin(tau * np.pi / 2)
            nu = np.random.uniform(3, 30)
            params = {'rho': rho, 'nu': nu}
        elif family == 'clayton':
            # Clayton: tau = theta / (theta + 2)
            if tau <= 0:
                rotation = (rotation + 180) % 360
                tau = abs(tau)
            theta = 2 * tau / (1 - tau) if tau < 0.999 else 20
            params = {'theta': max(theta, 0.1)}
        elif family == 'gumbel':
            # Gumbel: tau = 1 - 1/theta
            if tau <= 0:
                rotation = (rotation + 180) % 360
                tau = abs(tau)
            theta = 1 / (1 - tau) if tau > 0.01 else 1.1
            params = {'theta': max(theta, 1.01)}
        elif family == 'frank':
            # Frank: approximate tau = 1 - 4/theta * (1 - D_1(theta))
            # Use simple approximation
            theta = 10 * tau if abs(tau) > 0.1 else np.sign(tau) * 1
            params = {'theta': theta}
        elif family == 'joe':
            if tau <= 0:
                rotation = (rotation + 180) % 360
                tau = abs(tau)
            theta = 1 / (1 - tau) if tau > 0.01 else 1.1
            params = {'theta': max(theta, 1.01)}
        elif family == 'bb1':
            if tau <= 0:
                rotation = (rotation + 180) % 360
                tau = abs(tau)
            theta = np.random.uniform(0.1, 3)
            delta = np.random.uniform(1.01, 3)
            params = {'theta': theta, 'delta': delta}
        elif family == 'bb7':
            if tau <= 0:
                rotation = (rotation + 180) % 360
                tau = abs(tau)
            theta = np.random.uniform(1.01, 3)
            delta = np.random.uniform(0.1, 3)
            params = {'theta': theta, 'delta': delta}
        else:
            params = {}
        
        # Random weight
        weight = np.random.uniform(0.1, 1.0)
        
        components.append(MixtureComponent(
            family=family,
            params=params,
            rotation=rotation,
            weight=weight
        ))
    
    return MixtureCopula(components)


def generate_mixture_dataset(
    output_dir: str,
    n_samples: int,
    m: int = 64,
    n_components_range: Tuple[int, int] = (2, 4),
    families: Optional[List[str]] = None,
    tau_range: Tuple[float, float] = (-0.7, 0.7),
    n_points_per_sample: int = 1000,
    seed: int = 42,
    n_jobs: int = 1,
):
    """
    Generate dataset of mixture copulas.
    
    Args:
        output_dir: Output directory for HDF5 files
        n_samples: Number of mixture copulas to generate
        m: Grid resolution
        n_components_range: Range of components per mixture (min, max)
        families: Allowed copula families
        tau_range: Range of Kendall's tau
        n_points_per_sample: Number of scatter points per sample
        seed: Random seed
        n_jobs: Number of parallel jobs
    """
    from tqdm import tqdm
    import os
    from joblib import Parallel, delayed
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {n_samples} mixture copula samples...")
    print(f"  Components per mixture: {n_components_range[0]}-{n_components_range[1]}")
    print(f"  Grid resolution: {m}×{m}")
    print(f"  Points per sample: {n_points_per_sample}")
    print(f"  Output: {output_path}")
    
    def generate_single(idx: int):
        """Generate single mixture sample."""
        sample_seed = seed + idx
        
        # Random number of components
        n_comp = np.random.randint(
            n_components_range[0],
            n_components_range[1] + 1
        )
        
        # Generate mixture
        mixture = generate_random_mixture(
            n_components=n_comp,
            families=families,
            tau_range=tau_range,
            seed=sample_seed
        )
        
        # Sample points
        points = mixture.sample(n=n_points_per_sample, seed=sample_seed)
        
        # Compute density grid
        log_pdf_grid = mixture.log_density_grid(m=m)
        
        # Save to HDF5
        filename = output_path / f"mixture_{idx:08d}.h5"
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('points', data=points)
            f.create_dataset('log_pdf_grid', data=log_pdf_grid)
            
            # Metadata
            f.attrs['n_components'] = n_comp
            f.attrs['n'] = n_points_per_sample
            f.attrs['m'] = m
            f.attrs['type'] = 'mixture'
            
            # Store component info
            for i, comp in enumerate(mixture.components):
                f.attrs[f'comp_{i}_family'] = comp.family
                f.attrs[f'comp_{i}_rotation'] = comp.rotation
                f.attrs[f'comp_{i}_weight'] = comp.weight
                for key, val in comp.params.items():
                    f.attrs[f'comp_{i}_param_{key}'] = val
    
    # Generate in parallel
    if n_jobs == 1:
        for idx in tqdm(range(n_samples)):
            generate_single(idx)
    else:
        Parallel(n_jobs=n_jobs)(
            delayed(generate_single)(idx)
            for idx in tqdm(range(n_samples))
        )
    
    print(f"Generated {n_samples} mixture samples in {output_path}")


if __name__ == "__main__":
    # Test mixture generation
    np.random.seed(42)
    
    # Create a simple mixture
    components = [
        MixtureComponent(
            family='gaussian',
            params={'rho': 0.7},
            rotation=0,
            weight=0.6
        ),
        MixtureComponent(
            family='clayton',
            params={'theta': 2.0},
            rotation=0,
            weight=0.4
        )
    ]
    
    mixture = MixtureCopula(components)
    
    # Sample
    samples = mixture.sample(n=1000, seed=42)
    print(f"Samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Density
    density = mixture.density_grid(m=32)
    print(f"Density shape: {density.shape}")
    print(f"Density range: [{density.min():.3f}, {density.max():.3f}]")
    
    # Generate random mixture
    print("\nRandom mixture with 3 components:")
    random_mixture = generate_random_mixture(n_components=3, seed=123)
    for i, comp in enumerate(random_mixture.components):
        print(f"  Component {i}: {comp.family}, rotation={comp.rotation}°, weight={comp.weight:.3f}")
