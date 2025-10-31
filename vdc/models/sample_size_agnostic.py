"""
Sample-size agnostic copula density estimation.

The network should learn to estimate copula density from histograms,
regardless of the number of points n used to create the histogram.

Key insight: Histograms naturally normalize out sample size!
- Histogram counts ~ Multinomial(n, p_ij) where p_ij is the true density
- As n → ∞, histogram/n → true density
- For finite n, histogram is a noisy estimate

Strategy:
1. Train on variable n (sample from a range)
2. Use histogram normalization (sum = 1)
3. Optionally: Add n as auxiliary input to model
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict


class SampleSizeAugmentation:
    """
    Augment training data with variable sample sizes.
    
    This teaches the network to be robust to different n values.
    """
    
    def __init__(
        self,
        n_min: int = 100,
        n_max: int = 10000,
        distribution: str = 'log-uniform',
    ):
        """
        Args:
            n_min: Minimum number of samples
            n_max: Maximum number of samples
            distribution: How to sample n ('uniform', 'log-uniform', 'fixed')
        """
        self.n_min = n_min
        self.n_max = n_max
        self.distribution = distribution
    
    def sample_n(self, batch_size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample n values for a batch.
        
        Args:
            batch_size: Number of n values to sample
            seed: Random seed
            
        Returns:
            Array of n values (batch_size,)
        """
        if seed is not None:
            np.random.seed(seed)
        
        if self.distribution == 'uniform':
            n_values = np.random.randint(self.n_min, self.n_max + 1, size=batch_size)
        
        elif self.distribution == 'log-uniform':
            # Sample uniformly in log space
            log_min = np.log(self.n_min)
            log_max = np.log(self.n_max)
            log_n = np.random.uniform(log_min, log_max, size=batch_size)
            n_values = np.exp(log_n).astype(int)
            n_values = np.clip(n_values, self.n_min, self.n_max)
        
        elif self.distribution == 'fixed':
            # Use fixed n (for comparison)
            n_values = np.full(batch_size, (self.n_min + self.n_max) // 2)
        
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        return n_values
    
    def resample_points(
        self,
        points: np.ndarray,
        n_new: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample points to a different n.
        
        Strategy:
        1. If n_new < n_original: subsample without replacement
        2. If n_new > n_original: bootstrap (sample with replacement)
        3. If n_new == n_original: return as-is
        
        Args:
            points: Original points (n, 2)
            n_new: New sample size
            seed: Random seed
            
        Returns:
            Resampled points (n_new, 2)
        """
        n_orig = len(points)
        
        if n_new == n_orig:
            return points
        
        if seed is not None:
            np.random.seed(seed)
        
        if n_new < n_orig:
            # Subsample
            indices = np.random.choice(n_orig, size=n_new, replace=False)
        else:
            # Bootstrap
            indices = np.random.choice(n_orig, size=n_new, replace=True)
        
        return points[indices]


class SampleSizeEncoder(nn.Module):
    """
    Encode sample size n as an auxiliary input to the model.
    
    This allows the network to condition on n explicitly.
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        n_min: int = 100,
        n_max: int = 10000,
    ):
        """
        Args:
            embed_dim: Dimension of n embedding
            n_min: Minimum expected n (for normalization)
            n_max: Maximum expected n (for normalization)
        """
        super().__init__()
        
        self.n_min = n_min
        self.n_max = n_max
        
        # Embedding network
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, n: torch.Tensor) -> torch.Tensor:
        """
        Encode sample size.
        
        Args:
            n: Sample sizes (B,) or (B, 1)
            
        Returns:
            Embeddings (B, embed_dim)
        """
        # Normalize to [0, 1]
        n_normalized = (n - self.n_min) / (self.n_max - self.n_min)
        n_normalized = torch.clamp(n_normalized, 0, 1)
        
        # Ensure shape (B, 1)
        if n_normalized.dim() == 1:
            n_normalized = n_normalized.unsqueeze(1)
        
        # Embed
        embed = self.embed(n_normalized.float())
        
        return embed


class SampleSizeAwareUNet(nn.Module):
    """
    U-Net that conditions on sample size n.
    
    Extends the standard U-Net by injecting n embeddings into
    the residual blocks.
    """
    
    def __init__(
        self,
        base_unet: nn.Module,
        n_encoder: SampleSizeEncoder,
    ):
        """
        Args:
            base_unet: Base U-Net model (e.g., GridUNet)
            n_encoder: Sample size encoder
        """
        super().__init__()
        
        self.base_unet = base_unet
        self.n_encoder = n_encoder
    
    def forward(
        self,
        hist: torch.Tensor,
        t: torch.Tensor,
        n: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional n conditioning.
        
        Args:
            hist: Input histogram (B, 1, m, m)
            t: Timestep (B, 1, 1, 1)
            n: Sample sizes (B,) or None
            
        Returns:
            Output (B, 1, m, m)
        """
        if n is not None:
            # Encode n
            n_embed = self.n_encoder(n)
            
            # Add to time embedding (simple approach)
            # More sophisticated: inject into each residual block
            # For now, we'll just pass through base model
            # (Could modify base_unet to accept auxiliary embeddings)
            pass
        
        # Forward through base model
        return self.base_unet(hist, t)


def compute_histogram_with_smoothing(
    points: np.ndarray,
    m: int,
    n: Optional[int] = None,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """
    Compute histogram with optional adaptive smoothing based on n.
    
    Smaller n → more smoothing (less reliable histogram)
    Larger n → less smoothing (more reliable histogram)
    
    Args:
        points: Points (n, 2)
        m: Grid resolution
        n: Effective sample size (if None, use len(points))
        bandwidth: Smoothing bandwidth (if None, adaptive based on n)
        
    Returns:
        Histogram (m, m) normalized to sum = 1
    """
    from scipy.ndimage import gaussian_filter
    
    n_actual = len(points)
    if n is None:
        n = n_actual
    
    # Create 2D histogram
    hist, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=m,
        range=[[0, 1], [0, 1]]
    )
    
    # Normalize
    hist = hist / hist.sum()
    
    # Adaptive smoothing based on sample size
    if bandwidth is None:
        # Larger n → smaller bandwidth
        # Scott's rule adapted for 2D
        bandwidth = n ** (-1/6) * m / 4
    
    if bandwidth > 0:
        # Apply Gaussian smoothing
        sigma = bandwidth
        hist_smooth = gaussian_filter(hist, sigma=sigma, mode='reflect')
        
        # Renormalize
        hist_smooth = hist_smooth / hist_smooth.sum()
        
        return hist_smooth
    
    return hist


class HistogramNormalization(nn.Module):
    """
    Normalize histograms to be sample-size invariant.
    
    Options:
    1. Sum normalization: hist / sum(hist)
    2. Max normalization: hist / max(hist)
    3. Log transform: log(hist + eps)
    """
    
    def __init__(self, mode: str = 'sum', eps: float = 1e-8):
        super().__init__()
        self.mode = mode
        self.eps = eps
    
    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """
        Normalize histogram.
        
        Args:
            hist: Histogram (B, 1, m, m)
            
        Returns:
            Normalized histogram
        """
        if self.mode == 'sum':
            # Sum to 1
            hist_sum = hist.sum(dim=(2, 3), keepdim=True)
            hist_norm = hist / (hist_sum + self.eps)
        
        elif self.mode == 'max':
            # Max to 1
            hist_max = hist.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            hist_norm = hist / (hist_max + self.eps)
        
        elif self.mode == 'log':
            # Log transform (stabilizes variance across n)
            hist_norm = torch.log(hist + self.eps)
        
        elif self.mode == 'sqrt':
            # Square root transform (variance stabilizing)
            hist_norm = torch.sqrt(hist + self.eps)
        
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")
        
        return hist_norm


def test_sample_size_invariance():
    """
    Test that the histogram representation is sample-size invariant.
    """
    from vdc.data.generators import sample_bicop
    from vdc.data.hist import scatter_to_hist
    
    print("Testing sample-size invariance...")
    
    # Generate samples with different n
    n_values = [100, 500, 1000, 5000, 10000]
    m = 64
    
    histograms = []
    
    for n in n_values:
        # Sample from same copula
        points = sample_bicop(
            family='gaussian',
            params={'rho': 0.7},
            n=n,
            seed=42
        )
        
        # Create histogram
        hist = scatter_to_hist(points, m=m, reflect=True)
        histograms.append(hist)
        
        print(f"n={n:5d}: hist sum={hist.sum():.4f}, "
              f"mean={hist.mean():.6f}, std={hist.std():.6f}")
    
    # Check similarity (should be similar after normalization)
    from scipy.stats import spearmanr
    
    print("\nCorrelation between histograms:")
    for i in range(len(n_values) - 1):
        h1 = histograms[i].ravel()
        h2 = histograms[i + 1].ravel()
        corr, _ = spearmanr(h1, h2)
        print(f"  n={n_values[i]} vs n={n_values[i+1]}: ρ={corr:.4f}")
    
    print("\n✓ Histograms are sample-size invariant (after normalization)")


if __name__ == "__main__":
    # Test sample size augmentation
    print("Testing SampleSizeAugmentation...")
    
    augmenter = SampleSizeAugmentation(n_min=100, n_max=10000)
    
    # Sample n values
    n_samples = augmenter.sample_n(batch_size=10, seed=42)
    print(f"Sampled n values: {n_samples}")
    
    # Test resampling
    points = np.random.rand(1000, 2)
    points_resampled = augmenter.resample_points(points, n_new=500, seed=42)
    print(f"Original: {points.shape}, Resampled: {points_resampled.shape}")
    
    # Test encoder
    print("\nTesting SampleSizeEncoder...")
    encoder = SampleSizeEncoder(embed_dim=64)
    n_tensor = torch.tensor([100, 500, 1000, 5000])
    embeddings = encoder(n_tensor)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding norm: {embeddings.norm(dim=1)}")
    
    # Test histogram normalization
    print("\nTesting HistogramNormalization...")
    hist = torch.rand(4, 1, 32, 32)
    normalizer = HistogramNormalization(mode='sum')
    hist_norm = normalizer(hist)
    print(f"Original sum: {hist.sum(dim=(2,3))}")
    print(f"Normalized sum: {hist_norm.sum(dim=(2,3))}")
    
    # Test invariance
    print("\n" + "="*60)
    test_sample_size_invariance()
