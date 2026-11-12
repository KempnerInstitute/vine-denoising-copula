"""Conditional CDFs and BB-family sampling utilities.

An h-function is a conditional CDF, not a bivariate copula. Fixed-condition
samples below are diagnostics and must not be used as bivariate training pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
from scipy.stats import norm, t as student_t


# =============================================================================
# h-Functions (Conditional CDFs): h(u|v) = ∂C(u,v)/∂v
# =============================================================================

def h_gaussian(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    """Gaussian copula h-function: h(u|v) = Φ((Φ⁻¹(u) - ρΦ⁻¹(v)) / √(1-ρ²))."""
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    x = norm.ppf(u)
    y = norm.ppf(v)
    return norm.cdf((x - rho * y) / np.sqrt(1 - rho**2))


def h_student(u: np.ndarray, v: np.ndarray, rho: float, df: int) -> np.ndarray:
    """Student-t copula h-function."""
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    x = student_t.ppf(u, df)
    y = student_t.ppf(v, df)
    # h(u|v) = t_{df+1}((x - ρy) / √((df + y²)(1 - ρ²)/(df + 1)))
    num = x - rho * y
    den = np.sqrt((df + y**2) * (1 - rho**2) / (df + 1))
    return student_t.cdf(num / den, df + 1)


def h_clayton(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Clayton copula h-function."""
    if theta < 1e-10:
        return u  # Independence
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    # h(u|v) = v^(-(θ+1)) * (u^(-θ) + v^(-θ) - 1)^(-(1+1/θ))
    return v**(-(theta + 1)) * (u**(-theta) + v**(-theta) - 1)**(-(1 + 1/theta))


def h_gumbel(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Gumbel copula h-function."""
    if theta <= 1 + 1e-10:
        return u  # Independence
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    
    lu = -np.log(u)
    lv = -np.log(v)
    s = lu**theta + lv**theta
    C = np.exp(-s**(1/theta))
    
    # h(u|v) = C(u,v) / v * ((-log v)^(θ-1)) / (s^(1-1/θ))
    return C / v * (lv**(theta - 1)) * (s**(1/theta - 1))


def h_frank(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Frank copula h-function."""
    if abs(theta) < 1e-10:
        return u  # Independence
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    
    # Both numerator and denominator are negative for positive theta.
    au = np.expm1(-theta * u)
    av = np.expm1(-theta * v)
    return np.exp(-theta * v) * au / (np.expm1(-theta) + au * av)


def h_joe(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Joe copula h-function."""
    if theta <= 1 + 1e-10:
        return u  # Independence
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    
    one_u = 1 - u
    one_v = 1 - v
    a_u = one_u**theta
    a_v = one_v**theta
    
    # C(u,v) = 1 - (ū^θ + v̄^θ - ū^θ v̄^θ)^(1/θ)
    s = a_u + a_v - a_u * a_v
    return one_v**(theta - 1) * (1 - a_u) * s**(1/theta - 1)


# =============================================================================
# Inverse h-Functions (for sampling)
# =============================================================================

def hinv_gaussian(w: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    """Inverse h-function for Gaussian copula."""
    w = np.clip(w, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    y = norm.ppf(v)
    z = norm.ppf(w)
    return norm.cdf(rho * y + np.sqrt(1 - rho**2) * z)


def hinv_student(w: np.ndarray, v: np.ndarray, rho: float, df: int) -> np.ndarray:
    """Inverse h-function for Student-t copula."""
    w = np.clip(w, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    y = student_t.ppf(v, df)
    z = student_t.ppf(w, df + 1)
    x = rho * y + z * np.sqrt((df + y**2) * (1 - rho**2) / (df + 1))
    return student_t.cdf(x, df)


def hinv_clayton(w: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Inverse h-function for Clayton copula."""
    if theta < 1e-10:
        return w
    w = np.clip(w, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    t = v**(-theta) * (w**(-theta / (1 + theta)) - 1) + 1
    return np.clip(t**(-1/theta), 1e-10, 1 - 1e-10)


def hinv_frank(w: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Inverse h-function for Frank copula."""
    if abs(theta) < 1e-10:
        return w
    w = np.clip(w, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    
    a = np.exp(-theta)
    av = np.exp(-theta * v)
    
    t = -np.log(1 + w * (a - 1) / (av + w * (1 - av))) / theta
    return np.clip(t, 1e-10, 1 - 1e-10)


# =============================================================================
# Two-Parameter Families (BB1, BB7)
# =============================================================================

def sample_bb1(n: int, theta: float, delta: float, seed: Optional[int] = None) -> np.ndarray:
    """Sample BB1 (theta > 0, delta >= 1) with validated conditional inversion."""
    from vdc.data._bicop import sample
    return sample("bb1", [theta, delta], n, seed)


def sample_bb7(n: int, theta: float, delta: float, seed: Optional[int] = None) -> np.ndarray:
    """Sample BB7 (theta >= 1, delta > 0) with validated conditional inversion."""
    from vdc.data._bicop import sample
    return sample("bb7", [theta, delta], n, seed)


def bb1_density(u: np.ndarray, v: np.ndarray, theta: float, delta: float) -> np.ndarray:
    """Evaluate the analytic BB1 density, preserving broadcast input shape."""
    from vdc.data._bicop import density
    return density("bb1", [theta, delta], u, v)


def bb7_density(u: np.ndarray, v: np.ndarray, theta: float, delta: float) -> np.ndarray:
    """Evaluate the analytic BB7 density, preserving broadcast input shape."""
    from vdc.data._bicop import density
    return density("bb7", [theta, delta], u, v)


# =============================================================================
# Conditional Copula Data Generator
# =============================================================================

@dataclass
class ConditionalCopulaSpec:
    """Specification for a conditional copula training example."""
    family: str
    params: Dict[str, Any]
    conditioning_value: float  # v in h(u|v)
    name: str


def generate_conditional_copula_samples(
    family: str,
    params: Dict[str, Any],
    v_condition: float,
    n: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate samples from conditional copula h(u|v=v_condition).
    
    The returned (u, w) pairs are deterministically related and do not have
    uniform margins. Use this function for conditional diagnostics only, not
    as a source of bivariate copula training examples.
    
    Args:
        family: Copula family
        params: Family parameters
        v_condition: Conditioning value
        n: Number of samples
        seed: Random seed
        
    Returns:
        (n, 2) array where column 0 is the conditioned u, column 1 is w (uniform)
    """
    if seed is not None:
        np.random.seed(seed)
    
    w = np.random.uniform(0, 1, n)
    v = np.full(n, v_condition)
    
    fam = family.lower()
    
    if fam == "gaussian":
        u = hinv_gaussian(w, v, params["rho"])
    elif fam == "student":
        u = hinv_student(w, v, params["rho"], int(params.get("df", params.get("nu", 5))))
    elif fam == "clayton":
        u = hinv_clayton(w, v, params["theta"])
    elif fam == "frank":
        u = hinv_frank(w, v, params["theta"])
    else:
        raise ValueError(f"Unsupported conditional family: {family}")
    
    return np.column_stack([u, w])


def generate_conditional_copula_density_slice(
    family: str,
    params: Dict[str, Any],
    v_condition: float,
    m: int = 256,
) -> np.ndarray:
    """
    Generate the conditional density c(u|v=v_condition) = c(u,v) / c_V(v).
    
    Returns:
        (m,) array of conditional density values at grid points.
    """
    u_grid = np.linspace(0.5/m, 1 - 0.5/m, m)
    v = np.full(m, v_condition)
    
    fam = family.lower()
    
    from vdc.data.generators import (
        gaussian_copula_density, student_copula_density, clayton_copula_density,
        frank_copula_density, gumbel_copula_density, joe_copula_density
    )
    
    if fam == "gaussian":
        joint_density = gaussian_copula_density(u_grid, v, params["rho"])
    elif fam == "student":
        joint_density = student_copula_density(
            u_grid, v, params["rho"], int(params.get("df", params.get("nu", 5)))
        )
    elif fam == "clayton":
        joint_density = clayton_copula_density(u_grid, v, params["theta"])
    elif fam == "frank":
        joint_density = frank_copula_density(u_grid, v, params["theta"])
    elif fam == "gumbel":
        joint_density = gumbel_copula_density(u_grid, v, params["theta"])
    elif fam == "joe":
        joint_density = joe_copula_density(u_grid, v, params["theta"])
    else:
        raise ValueError(f"Unsupported conditional family: {family}")
    
    # Normalize (conditional density integrates to 1)
    du = 1.0 / m
    mass = joint_density.sum() * du
    if mass > 1e-10:
        joint_density = joint_density / mass
    
    return joint_density


# =============================================================================
# H-Function Registry
# =============================================================================

H_FUNCTIONS = {
    "gaussian": h_gaussian,
    "student": h_student,
    "clayton": h_clayton,
    "gumbel": h_gumbel,
    "frank": h_frank,
    "joe": h_joe,
}

HINV_FUNCTIONS = {
    "gaussian": hinv_gaussian,
    "student": hinv_student,
    "clayton": hinv_clayton,
    "frank": hinv_frank,
}

TWO_PARAM_SAMPLERS = {
    "bb1": sample_bb1,
    "bb7": sample_bb7,
}

TWO_PARAM_DENSITIES = {
    "bb1": bb1_density,
    "bb7": bb7_density,
}


__all__ = [
    # h-functions
    "h_gaussian", "h_student", "h_clayton", "h_gumbel", "h_frank", "h_joe",
    "H_FUNCTIONS",
    # inverse h-functions
    "hinv_gaussian", "hinv_student", "hinv_clayton", "hinv_frank",
    "HINV_FUNCTIONS",
    # two-parameter families
    "sample_bb1", "sample_bb7",
    "bb1_density", "bb7_density",
    "TWO_PARAM_SAMPLERS", "TWO_PARAM_DENSITIES",
    # conditional generators
    "ConditionalCopulaSpec",
    "generate_conditional_copula_samples",
    "generate_conditional_copula_density_slice",
]
