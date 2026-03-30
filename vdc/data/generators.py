"""
Copula Data Generators for Training and Evaluation.

Provides generators for:
- Single-family copulas (Gaussian, Student-t, Clayton, Gumbel, Frank, Joe)
- Mixture copulas
- Vine copulas (D-vine, C-vine)
"""

import numpy as np
import torch
from scipy.stats import norm, t as student_t
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass


def scatter_to_hist(pts: np.ndarray, m: int, reflect: bool = True) -> np.ndarray:
    """
    Create 2D histogram from scatter points with optional boundary reflection.
    
    Args:
        pts: (n, 2) array of points in [0, 1]^2
        m: Grid size
        reflect: Whether to reflect points at boundaries
        
    Returns:
        (m, m) histogram array
    """
    if reflect:
        pts_reflected = []
        for dx, dy in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            shifted = pts.copy()
            shifted[:, 0] += dx
            shifted[:, 1] += dy
            mask = (shifted[:, 0] >= 0) & (shifted[:, 0] <= 1) & \
                   (shifted[:, 1] >= 0) & (shifted[:, 1] <= 1)
            pts_reflected.append(shifted[mask])
        pts_all = np.vstack(pts_reflected)
    else:
        pts_all = pts
    
    hist, _, _ = np.histogram2d(
        pts_all[:, 0], pts_all[:, 1],
        bins=m, range=[[0, 1], [0, 1]]
    )
    return hist.astype(np.float64)


# =============================================================================
# Copula Sampling Functions
# =============================================================================

def _normalize_rotation(rotation: int) -> int:
    r = int(rotation) % 360
    if r not in (0, 90, 180, 270):
        raise ValueError(f"rotation must be one of {{0,90,180,270}}, got {rotation}")
    return r


def _rotate_uv(u: np.ndarray, v: np.ndarray, rotation: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate copula coordinates (u,v) by 0/90/180/270 degrees (copula rotation conventions)."""
    r = _normalize_rotation(rotation)
    if r == 0:
        return u, v
    if r == 90:
        return 1.0 - u, v
    if r == 180:
        return 1.0 - u, 1.0 - v
    # r == 270
    return u, 1.0 - v


def sample_bicop(
    family: str,
    params: Dict[str, float],
    n: int,
    rotation: int = 0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Unified sampler for a bivariate copula family (with optional 0/90/180/270 rotation).

    This function is used throughout the codebase (training data, benchmarks, tests).
    """
    if seed is not None:
        np.random.seed(seed)

    fam = family.lower()
    if fam in {"student_t", "t", "studentt"}:
        fam = "student"

    if fam == "gaussian":
        samples = sample_gaussian_copula(n=n, rho=float(params["rho"]))
    elif fam == "student":
        rho = float(params["rho"])
        df = int(params.get("df", params.get("nu", 5)))
        samples = sample_student_copula(n=n, rho=rho, df=df)
    elif fam in {"independence", "indep"}:
        samples = np.random.uniform(0.0, 1.0, size=(n, 2))
    elif fam == "clayton":
        samples = sample_clayton_copula(n=n, theta=float(params["theta"]))
    elif fam == "gumbel":
        samples = sample_gumbel_copula(n=n, theta=float(params["theta"]))
    elif fam == "frank":
        samples = sample_frank_copula(n=n, theta=float(params["theta"]))
    elif fam == "joe":
        samples = sample_joe_copula(n=n, theta=float(params["theta"]))
    else:
        raise ValueError(f"Unknown family '{family}'")

    u, v = samples[:, 0], samples[:, 1]
    u, v = _rotate_uv(u, v, rotation)
    return np.clip(np.column_stack([u, v]), 1e-10, 1.0 - 1e-10)


def sample_gaussian_copula(n: int, rho: float) -> np.ndarray:
    """Sample from bivariate Gaussian copula."""
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    Z = np.random.multivariate_normal([0, 0], Sigma, n)
    return norm.cdf(Z)


def sample_student_copula(n: int, rho: float, df: int = 5) -> np.ndarray:
    """Sample from bivariate Student-t copula."""
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Z = np.random.standard_normal((n, 2))
    chi2 = np.random.chisquare(df, n)
    T = Z @ L.T * np.sqrt(df / chi2[:, np.newaxis])
    return student_t.cdf(T, df)


def sample_clayton_copula(n: int, theta: float) -> np.ndarray:
    """Sample from bivariate Clayton copula."""
    if theta < 1e-10:
        return np.random.uniform(0, 1, (n, 2))
    
    u = np.random.uniform(0, 1, n)
    w = np.random.uniform(0, 1, n)
    
    # Inverse conditional: v = ((u^(-theta) * (w^(-theta/(1+theta)) - 1) + 1)^(-1/theta)
    t = u ** (-theta) * (w ** (-theta / (1 + theta)) - 1) + 1
    v = np.clip(t ** (-1 / theta), 1e-10, 1 - 1e-10)
    
    return np.column_stack([u, v])


def sample_gumbel_copula(n: int, theta: float) -> np.ndarray:
    """Sample from bivariate Gumbel copula using stable distribution."""
    if theta <= 1:
        return np.random.uniform(0, 1, (n, 2))
    
    # Generate stable random variable
    alpha = 1.0 / theta
    w = np.random.exponential(1, n)
    
    # Stable distribution via Chambers-Mallows-Stuck method
    V = np.random.uniform(-np.pi/2, np.pi/2, n)
    W = np.random.exponential(1, n)
    
    S = np.sin(alpha * (V + np.pi/2)) / (np.cos(V) ** (1/alpha)) * \
        (np.cos(V - alpha * (V + np.pi/2)) / W) ** ((1-alpha)/alpha)
    
    # Generate copula samples
    u1 = np.random.uniform(0, 1, n)
    u2 = np.random.uniform(0, 1, n)
    
    e1 = -np.log(u1)
    e2 = -np.log(u2)
    
    U1 = np.exp(-(e1 / S) ** alpha)
    U2 = np.exp(-(e2 / S) ** alpha)
    
    return np.clip(np.column_stack([U1, U2]), 1e-10, 1 - 1e-10)


def sample_frank_copula(n: int, theta: float) -> np.ndarray:
    """Sample from bivariate Frank copula."""
    if abs(theta) < 1e-10:
        return np.random.uniform(0, 1, (n, 2))
    
    u = np.random.uniform(0, 1, n)
    w = np.random.uniform(0, 1, n)
    
    # Inverse conditional
    t = -np.log(1 + w * (np.exp(-theta) - 1) / (np.exp(-theta * u) + w * (1 - np.exp(-theta * u)))) / theta
    v = np.clip(t, 1e-10, 1 - 1e-10)
    
    return np.column_stack([u, v])


def sample_joe_copula(n: int, theta: float) -> np.ndarray:
    """Sample from bivariate Joe copula."""
    if theta <= 1:
        return np.random.uniform(0, 1, (n, 2))
    
    u = np.random.uniform(0, 1, n)
    w = np.random.uniform(0, 1, n)
    
    # Approximate inverse conditional via bisection
    v = np.zeros(n)
    for i in range(n):
        def h_func(v_val):
            t1 = (1 - u[i]) ** theta
            t2 = (1 - v_val) ** theta
            t3 = t1 + t2 - t1 * t2
            return (1 - v_val) ** (theta - 1) * (t2 - t1 * t2) ** (1/theta - 1) * t1
        
        # Simple bisection
        lo, hi = 1e-10, 1 - 1e-10
        target = w[i]
        for _ in range(50):
            mid = (lo + hi) / 2
            if h_func(mid) < target:
                hi = mid
            else:
                lo = mid
        v[i] = (lo + hi) / 2
    
    return np.column_stack([u, v])


# =============================================================================
# Copula Density Functions
# =============================================================================

def gaussian_copula_density(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    """Compute Gaussian copula density."""
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    
    x = norm.ppf(u)
    y = norm.ppf(v)
    
    rho2 = rho ** 2
    exponent = (rho2 * (x**2 + y**2) - 2*rho*x*y) / (2 * (1 - rho2))
    
    return np.exp(-exponent) / np.sqrt(1 - rho2)


def student_copula_density(u: np.ndarray, v: np.ndarray, rho: float, df: int) -> np.ndarray:
    """Compute Student-t copula density."""
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    
    x = student_t.ppf(u, df)
    y = student_t.ppf(v, df)
    
    rho2 = rho ** 2
    
    # Copula density formula
    num = 1 + (x**2 + y**2 - 2*rho*x*y) / (df * (1 - rho2))
    den = (1 + x**2/df) * (1 + y**2/df)
    
    from scipy.special import gamma
    const = gamma((df + 2)/2) * gamma(df/2) / (gamma((df + 1)/2)**2 * np.sqrt(1 - rho2))
    
    return const * (num ** (-(df + 2)/2)) / (den ** (-(df + 1)/2))


def clayton_copula_density(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Compute Clayton copula density."""
    if theta < 1e-10:
        return np.ones_like(u)
    
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    
    return (1 + theta) * (u * v) ** (-(1 + theta)) * \
           (u ** (-theta) + v ** (-theta) - 1) ** (-(2 + 1/theta))


def frank_copula_density(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Compute Frank copula density."""
    if abs(theta) < 1e-10:
        return np.ones_like(u)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    a = np.exp(-theta)
    au = np.exp(-theta * u)
    av = np.exp(-theta * v)
    num = theta * (1 - a) * au * av
    den = (1 - a - (1 - au) * (1 - av)) ** 2
    return num / np.clip(den, 1e-300, None)


def gumbel_copula_density(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Compute Gumbel copula density."""
    if theta <= 1 + 1e-10:
        return np.ones_like(u)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)

    lu = -np.log(u)
    lv = -np.log(v)
    a = lu ** theta
    b = lv ** theta
    s = a + b
    # Guard against tiny s causing numerical overflow in negative powers.
    s = np.clip(s, 1e-300, None)
    t = s ** (1.0 / theta)
    C = np.exp(-t)

    # c(u,v) = C * s^(1/theta - 2) * (t + (theta-1)) * lu^(theta-1) * lv^(theta-1) / (u v)
    density = (
        C
        * np.power(s, (1.0 / theta) - 2.0)
        * (t + (theta - 1.0))
        * (lu ** (theta - 1.0))
        * (lv ** (theta - 1.0))
        / (u * v)
    )
    return np.clip(density, 0.0, None)


def joe_copula_density(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Compute Joe copula density via Archimedean generator derivatives."""
    if theta <= 1 + 1e-10:
        return np.ones_like(u)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)

    # psi^{-1}(u) = -log(1 - (1-u)^theta)
    one_u = 1.0 - u
    one_v = 1.0 - v
    a_u = np.power(one_u, theta)
    a_v = np.power(one_v, theta)
    # Use log1p for numerical stability when a_u/a_v are very small (u,v near 1).
    inv_u = -np.log1p(-np.clip(a_u, 0.0, 1.0 - 1e-16))
    inv_v = -np.log1p(-np.clip(a_v, 0.0, 1.0 - 1e-16))

    inv_u_p = -theta * np.power(one_u, theta - 1.0) / np.clip(1.0 - a_u, 1e-300, None)
    inv_v_p = -theta * np.power(one_v, theta - 1.0) / np.clip(1.0 - a_v, 1e-300, None)

    s = inv_u + inv_v
    e = np.exp(-s)
    # b = 1 - exp(-s) computed stably for small s (prevents b rounding to 0).
    b = -np.expm1(-s)
    b = np.clip(b, 1e-300, None)

    # psi''(s) = (1/theta) * e^{-s} * b^{1/theta - 1}
    #            + (1/theta)(1 - 1/theta) * e^{-2s} * b^{1/theta - 2}
    inv_theta = 1.0 / theta
    term1 = inv_theta * e * np.power(b, inv_theta - 1.0)
    term2 = inv_theta * (1.0 - inv_theta) * (e ** 2) * np.power(b, inv_theta - 2.0)
    psi_pp = term1 + term2

    density = psi_pp * inv_u_p * inv_v_p
    return np.clip(density, 0.0, None)


def analytic_logpdf_grid(
    family: str,
    params: Dict[str, float],
    m: int = 64,
    rotation: int = 0,
) -> np.ndarray:
    """
    Compute an analytic copula log-density grid on a uniform m×m grid of cell-centers.

    Returns:
        log_density: (m, m) array, normalized so that ∫∫ exp(log_density) dudv ≈ 1.
    """
    fam = family.lower()
    if fam in {"student_t", "t", "studentt"}:
        fam = "student"

    u_grid = np.linspace(0.5 / m, 1.0 - 0.5 / m, m)
    v_grid = np.linspace(0.5 / m, 1.0 - 0.5 / m, m)
    U, V = np.meshgrid(u_grid, v_grid, indexing="ij")

    U0, V0 = _rotate_uv(U, V, rotation)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        if fam == "gaussian":
            density = gaussian_copula_density(U0, V0, float(params["rho"]))
        elif fam == "student":
            rho = float(params["rho"])
            df = int(params.get("df", params.get("nu", 5)))
            density = student_copula_density(U0, V0, rho=rho, df=df)
        elif fam in {"independence", "indep"}:
            density = np.ones_like(U0, dtype=np.float64)
        elif fam == "clayton":
            density = clayton_copula_density(U0, V0, float(params["theta"]))
        elif fam == "frank":
            density = frank_copula_density(U0, V0, float(params["theta"]))
        elif fam == "gumbel":
            density = gumbel_copula_density(U0, V0, float(params["theta"]))
        elif fam == "joe":
            density = joe_copula_density(U0, V0, float(params["theta"]))
        else:
            raise ValueError(f"Unknown family '{family}'")

    # Robustify: replace NaN/Inf before normalization.
    density = np.nan_to_num(density, nan=0.0, posinf=1e300, neginf=0.0)
    density = np.clip(density, 0.0, 1e300)

    # Normalize in log-space for stability.
    from scipy.special import logsumexp

    du = 1.0 / m
    log_d = np.log(np.clip(density, 1e-300, 1e300))
    log_mass = logsumexp(log_d) + np.log(du * du)
    if not np.isfinite(log_mass):
        # Fallback to uniform copula density if something went wrong.
        return np.zeros_like(log_d)
    return log_d - log_mass


# =============================================================================
# Main Generator Class
# =============================================================================

class CopulaGenerator:
    """
    Generator for copula training data.
    
    Generates batches of (density, histogram) pairs for training.
    """
    
    FAMILY_SAMPLERS = {
        'gaussian': sample_gaussian_copula,
        'student': sample_student_copula,
        'clayton': sample_clayton_copula,
        'gumbel': sample_gumbel_copula,
        'frank': sample_frank_copula,
        'joe': sample_joe_copula,
    }
    
    def __init__(
        self,
        families: List[str] = ['gaussian', 'student', 'clayton', 'gumbel'],
        sample_size_range: Tuple[int, int] = (200, 2000),
        mixture_prob: float = 0.3,
        m: int = 64,
        seed: Optional[int] = None,
    ):
        """
        Initialize generator.
        
        Args:
            families: List of copula families to sample from
            sample_size_range: Range of sample sizes for histogram
            mixture_prob: Probability of generating mixture copula
            m: Density grid size
            seed: Random seed
        """
        self.families = families
        self.sample_size_range = sample_size_range
        self.mixture_prob = mixture_prob
        self.m = m
        
        if seed is not None:
            np.random.seed(seed)
    
    def _sample_params(self, family: str) -> dict:
        """Sample random parameters for a copula family."""
        if family == 'gaussian':
            return {'rho': np.random.uniform(-0.95, 0.95)}
        elif family == 'student':
            return {'rho': np.random.uniform(-0.9, 0.9), 'df': np.random.randint(2, 30)}
        elif family == 'clayton':
            return {'theta': np.random.uniform(0.1, 10.0)}
        elif family == 'gumbel':
            return {'theta': np.random.uniform(1.1, 8.0)}
        elif family == 'frank':
            return {'theta': np.random.uniform(-15.0, 15.0)}
        elif family == 'joe':
            return {'theta': np.random.uniform(1.1, 8.0)}
        else:
            raise ValueError(f"Unknown family: {family}")
    
    def _compute_density_grid(self, family: str, params: dict) -> np.ndarray:
        """Compute true copula density on grid."""
        m = self.m
        u_grid = np.linspace(0.5/m, 1 - 0.5/m, m)
        v_grid = np.linspace(0.5/m, 1 - 0.5/m, m)
        U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
        
        if family == 'gaussian':
            density = gaussian_copula_density(U, V, params['rho'])
        elif family == 'student':
            density = student_copula_density(U, V, params['rho'], params['df'])
        elif family == 'clayton':
            density = clayton_copula_density(U, V, params['theta'])
        else:
            # For other families, estimate density from samples
            sampler = self.FAMILY_SAMPLERS[family]
            samples = sampler(50000, **params)
            density = scatter_to_hist(samples, m, reflect=True)
            du = 1.0 / m
            density = density / (density.sum() * du * du + 1e-12)
        
        # Normalize
        du = 1.0 / m
        density = np.clip(density, 1e-10, None)
        density = density / (density.sum() * du * du)
        
        return density
    
    def generate_single(self) -> Dict[str, np.ndarray]:
        """Generate a single (density, histogram, samples) tuple."""
        # Sample family and parameters
        family = np.random.choice(self.families)
        params = self._sample_params(family)
        
        # Sample size for histogram
        n = np.random.randint(*self.sample_size_range)
        
        # Generate samples
        sampler = self.FAMILY_SAMPLERS[family]
        samples = sampler(n, **params)
        
        # Compute histogram
        histogram = scatter_to_hist(samples, self.m, reflect=True)
        du = 1.0 / self.m
        histogram = histogram / (histogram.sum() * du * du + 1e-12)
        
        # Compute true density
        density = self._compute_density_grid(family, params)
        
        return {
            'density': density,
            'histogram': histogram,
            'samples': samples,
            'family': family,
            'params': params,
            'n_samples': n,
        }
    
    def generate_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Generate a batch of training data."""
        densities = []
        histograms = []
        
        for _ in range(batch_size):
            data = self.generate_single()
            densities.append(data['density'])
            histograms.append(data['histogram'])
        
        densities = np.stack(densities)[:, np.newaxis, :, :]
        histograms = np.stack(histograms)[:, np.newaxis, :, :]
        
        # Convert to log space
        log_densities = np.log(np.clip(densities, 1e-12, None))
        log_histograms = np.log(np.clip(histograms, 1e-12, None))
        
        return {
            'density': torch.from_numpy(densities).float(),
            'log_density': torch.from_numpy(log_densities).float(),
            'histogram': torch.from_numpy(histograms).float(),
            'log_histogram': torch.from_numpy(log_histograms).float(),
        }


# =============================================================================
# Vine Copula Generators
# =============================================================================

def generate_gaussian_vine(n: int, d: int, rho: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
    """Generate samples from Gaussian copula with AR(1) correlation."""
    if seed is not None:
        np.random.seed(seed)
    Sigma = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)])
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    return norm.cdf(Z)


def generate_student_vine(n: int, d: int, rho: float = 0.5, df: int = 5, seed: Optional[int] = None) -> np.ndarray:
    """Generate samples from Student-t copula with AR(1) correlation."""
    if seed is not None:
        np.random.seed(seed)
    Sigma = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)])
    L = np.linalg.cholesky(Sigma)
    Z = np.random.standard_normal((n, d))
    chi2 = np.random.chisquare(df, n)
    T = Z @ L.T * np.sqrt(df / chi2[:, np.newaxis])
    return student_t.cdf(T, df)


def generate_clayton_vine(n: int, d: int, theta: float = 2.0, seed: Optional[int] = None) -> np.ndarray:
    """Generate samples from Clayton D-vine."""
    if seed is not None:
        np.random.seed(seed)
    
    def clayton_hinv(u, v, theta):
        if theta < 1e-10:
            return u
        t = v ** (-theta) * (u ** (-theta / (1 + theta)) - 1) + 1
        return np.clip(t ** (-1 / theta), 1e-10, 1 - 1e-10)
    
    U = np.zeros((n, d))
    W = np.random.uniform(0, 1, (n, d))
    U[:, 0] = W[:, 0]
    
    for j in range(1, d):
        U[:, j] = clayton_hinv(W[:, j], U[:, j-1], theta)
    
    return U


def generate_mixed_vine(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate samples from mixed D-vine with alternating families."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n, d))
    W = np.random.uniform(0, 1, (n, d))
    U[:, 0] = W[:, 0]
    
    for j in range(1, d):
        v = U[:, j-1]
        
        # Alternate between families
        if j % 3 == 1:
            # Clayton
            theta = 2.0
            t = v ** (-theta) * (W[:, j] ** (-theta / (1 + theta)) - 1) + 1
            U[:, j] = np.clip(t ** (-1 / theta), 1e-10, 1 - 1e-10)
        elif j % 3 == 2:
            # Gaussian
            rho = 0.6
            z_v = norm.ppf(np.clip(v, 1e-6, 1-1e-6))
            z_w = norm.ppf(np.clip(W[:, j], 1e-6, 1-1e-6))
            z_u = rho * z_v + np.sqrt(1 - rho**2) * z_w
            U[:, j] = norm.cdf(z_u)
        else:
            # Frank
            theta = 5.0
            t = -np.log(1 + W[:, j] * (np.exp(-theta) - 1) / 
                       (np.exp(-theta * v) + W[:, j] * (1 - np.exp(-theta * v)))) / theta
            U[:, j] = np.clip(t, 1e-10, 1 - 1e-10)
    
    return U
