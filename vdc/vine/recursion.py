"""
Vine recursion: h-function transforms, Rosenblatt, and inverse sampling.

Implements the recursive algorithm for:
- Computing conditional distributions via h-functions
- Forward Rosenblatt transform (data → uniforms)
- Inverse Rosenblatt transform (uniforms → data) for sampling
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from .structure import VineStructure, VineTree
from ..models.hfunc import HFuncLookup


@dataclass
class VinePairCopula:
    """
    A fitted pair-copula with h-function lookups.
    
    Attributes:
        edge: (i, j, conditioning_set) tuple
        density_grid: (m, m) tensor of copula density
        hfunc: HFuncLookup object for this pair
        level: Tree level (0 = Tree 1)
    """
    edge: Tuple[int, int, set]
    density_grid: torch.Tensor
    hfunc: HFuncLookup
    level: int
    
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate density at points (u, v)."""
        u_t = torch.from_numpy(u).float()
        v_t = torch.from_numpy(v).float()
        return self.hfunc.pdf(u_t, v_t).numpy()
    
    def h_u_given_v(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """h(u|v) = ∂C(u,v)/∂v."""
        u_t = torch.from_numpy(u).float()
        v_t = torch.from_numpy(v).float()
        return self.hfunc.h_u_given_v(u_t, v_t).numpy()
    
    def h_v_given_u(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """h(v|u) = ∂C(u,v)/∂u."""
        v_t = torch.from_numpy(v).float()
        u_t = torch.from_numpy(u).float()
        return self.hfunc.h_v_given_u(v_t, u_t).numpy()
    
    def hinv_u_given_v(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inverse h-function: solve h(u|v) = q for u."""
        q_t = torch.from_numpy(q).float()
        v_t = torch.from_numpy(v).float()
        return self.hfunc.hinv_u_given_v(q_t, v_t).numpy()
    
    def hinv_v_given_u(self, q: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Inverse h-function: solve h(v|u) = q for v."""
        q_t = torch.from_numpy(q).float()
        u_t = torch.from_numpy(u).float()
        return self.hfunc.hinv_v_given_u(q_t, u_t).numpy()


class VineRecursion:
    """
    Handles h-function recursion for vine copulas.
    
    This class manages:
    - Transforming data through vine trees via h-functions
    - Computing joint densities via the vine decomposition
    - Forward/inverse Rosenblatt transforms for sampling
    """
    
    def __init__(self, structure: VineStructure):
        """
        Args:
            structure: VineStructure defining the vine
        """
        self.structure = structure
        self.pair_copulas: List[List[VinePairCopula]] = []  # [tree_level][edge_idx]
        self.d = structure.d
        
        # Cache for h-function outputs
        self._h_cache: Dict[str, np.ndarray] = {}
    
    def set_pair_copulas(self, pair_copulas: List[List[VinePairCopula]]):
        """Set fitted pair copulas for all edges."""
        self.pair_copulas = pair_copulas
    
    def add_pair_copula(self, copula: VinePairCopula):
        """Add a fitted pair copula."""
        level = copula.level
        while len(self.pair_copulas) <= level:
            self.pair_copulas.append([])
        self.pair_copulas[level].append(copula)
    
    def _cache_key(self, i: int, j: int, cond: set) -> str:
        """Generate cache key for h-function output."""
        cond_str = ",".join(map(str, sorted(cond)))
        return f"{i}|{j}|{cond_str}"
    
    def compute_h_transforms(
        self,
        U: np.ndarray,
        tree_level: int,
        prev_transforms: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute h-function transforms for a given tree level.
        
        Args:
            U: (n, d) original pseudo-observations
            tree_level: Which tree level (0 = Tree 1)
            prev_transforms: Dictionary of h-transforms from previous levels
            
        Returns:
            Dictionary mapping cache keys to transformed values
        """
        if tree_level >= len(self.pair_copulas):
            return {}
        
        if prev_transforms is None:
            prev_transforms = {}
        
        transforms = {}
        tree_copulas = self.pair_copulas[tree_level]
        
        for copula in tree_copulas:
            i, j, cond = copula.edge
            
            if tree_level == 0:
                # Tree 1: use original data
                u_data = U[:, i]
                v_data = U[:, j]
            else:
                # Tree k: use h-transforms from previous level
                # Need to identify which previous h-functions to use
                # This is where the vine recursion gets complex
                
                # For now, simplified: look up in previous transforms
                # In reality, you need to trace back through the vine structure
                u_key = self._find_transform_key(i, cond, prev_transforms)
                v_key = self._find_transform_key(j, cond, prev_transforms)
                
                if u_key is None or v_key is None:
                    continue
                
                u_data = prev_transforms[u_key]
                v_data = prev_transforms[v_key]
            
            # Compute both h-functions
            h_u_v = copula.h_u_given_v(u_data, v_data)  # u|v
            h_v_u = copula.h_v_given_u(v_data, u_data)  # v|u
            
            # Store with appropriate keys
            # The conditioning set for the next level includes the conditioned variable
            new_cond_u = cond | {j}
            new_cond_v = cond | {i}
            
            transforms[self._cache_key(i, j, new_cond_u)] = h_u_v
            transforms[self._cache_key(j, i, new_cond_v)] = h_v_u
        
        return transforms
    
    def _find_transform_key(
        self,
        var: int,
        cond: set,
        transforms: Dict[str, np.ndarray]
    ) -> Optional[str]:
        """Find the right h-transform for a variable given a conditioning set."""
        # Search for a matching key
        for key in transforms.keys():
            parts = key.split("|")
            if len(parts) == 3:
                var_str, other_str, cond_str = parts
                if int(var_str) == var:
                    # Check if conditioning set matches
                    key_cond = set(map(int, cond_str.split(","))) if cond_str else set()
                    if key_cond == cond:
                        return key
        return None
    
    def logpdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute log-density of vine copula at points U.
        
        Uses the vine decomposition:
        c(u) = ∏_{trees} ∏_{edges in tree} c_{ij|D}(u_i|D, u_j|D)
        
        Args:
            U: (n, d) pseudo-observations
            
        Returns:
            (n,) log-density values
        """
        n = U.shape[0]
        logpdf = np.zeros(n)
        
        # Store all h-transforms
        all_transforms = [{}]  # transforms[tree_level] = dict
        
        # Iterate through trees
        for tree_level in range(len(self.pair_copulas)):
            tree_copulas = self.pair_copulas[tree_level]
            
            if tree_level == 0:
                prev_transforms = None
            else:
                prev_transforms = all_transforms[-1]
            
            # Compute h-transforms for this level
            transforms = self.compute_h_transforms(U, tree_level, prev_transforms)
            all_transforms.append(transforms)
            
            # Add pair-copula densities
            for copula in tree_copulas:
                i, j, cond = copula.edge
                
                if tree_level == 0:
                    u_data = U[:, i]
                    v_data = U[:, j]
                else:
                    # Get transformed data
                    u_key = self._find_transform_key(i, cond, prev_transforms)
                    v_key = self._find_transform_key(j, cond, prev_transforms)
                    
                    if u_key is None or v_key is None:
                        continue
                    
                    u_data = prev_transforms[u_key]
                    v_data = prev_transforms[v_key]
                
                # Add log-density
                pair_pdf = copula.pdf(u_data, v_data)
                logpdf += np.log(np.clip(pair_pdf, 1e-10, None))
        
        return logpdf
    
    def pdf(self, U: np.ndarray) -> np.ndarray:
        """Compute density (not log-density)."""
        return np.exp(self.logpdf(U))
    
    def rosenblatt(self, U: np.ndarray) -> np.ndarray:
        """
        Forward Rosenblatt transform: U → W ~ uniform.
        
        Transforms multivariate copula data to independent uniforms.
        
        Args:
            U: (n, d) pseudo-observations
            
        Returns:
            W: (n, d) independent uniforms
        """
        n, d = U.shape
        W = np.zeros_like(U)
        
        # First variable unchanged
        W[:, 0] = U[:, 0]
        
        # Follow vine order
        order = self.structure.order if self.structure.order else list(range(d))
        
        # Store transforms at each level
        transforms = {}
        
        for k in range(1, d):
            var = order[k]
            
            # Find the appropriate h-function from the vine
            # This is simplified - in practice, you follow the vine structure
            
            # For now, use the first available h-function involving this variable
            found = False
            for tree_level in range(len(self.pair_copulas)):
                if found:
                    break
                for copula in self.pair_copulas[tree_level]:
                    i, j, cond = copula.edge
                    if i == var or j == var:
                        # Use this copula to transform
                        if tree_level == 0:
                            if i == var:
                                W[:, k] = copula.h_u_given_v(U[:, i], U[:, j])
                            else:
                                W[:, k] = copula.h_v_given_u(U[:, j], U[:, i])
                        found = True
                        break
        
        return W
    
    def inverse_rosenblatt(self, W: np.ndarray) -> np.ndarray:
        """
        Inverse Rosenblatt transform: W ~ uniform → U.
        
        Samples from the vine copula by transforming independent uniforms.
        
        Args:
            W: (n, d) independent uniform random variables
            
        Returns:
            U: (n, d) samples from the vine copula
        """
        n, d = W.shape
        U = np.zeros_like(W)
        
        # First variable unchanged
        U[:, 0] = W[:, 0]
        
        # Follow vine order
        order = self.structure.order if self.structure.order else list(range(d))
        
        # Sequentially sample using inverse h-functions
        for k in range(1, d):
            var = order[k]
            
            # Find the appropriate inverse h-function
            # This is highly simplified
            found = False
            for tree_level in range(len(self.pair_copulas)):
                if found:
                    break
                for copula in self.pair_copulas[tree_level]:
                    i, j, cond = copula.edge
                    if i == var or j == var:
                        if tree_level == 0:
                            if i == var:
                                # Solve h(u_i|u_j) = w_k for u_i
                                U[:, k] = copula.hinv_u_given_v(W[:, k], U[:, order[0]])
                            else:
                                U[:, k] = copula.hinv_v_given_u(W[:, k], U[:, order[0]])
                        found = True
                        break
        
        return U
    
    def simulate(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample from the vine copula.
        
        Args:
            n: Number of samples
            seed: Random seed
            
        Returns:
            U: (n, d) samples from vine copula
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate independent uniforms
        W = np.random.uniform(0, 1, size=(n, self.d))
        
        # Transform via inverse Rosenblatt
        U = self.inverse_rosenblatt(W)
        
        return U


def build_vine_from_copulas(
    structure: VineStructure,
    pair_copulas: List[List[VinePairCopula]]
) -> VineRecursion:
    """
    Build a VineRecursion object from structure and fitted copulas.
    
    Args:
        structure: VineStructure
        pair_copulas: List of lists of fitted pair copulas
        
    Returns:
        VineRecursion ready for inference
    """
    vine = VineRecursion(structure)
    vine.set_pair_copulas(pair_copulas)
    return vine


if __name__ == "__main__":
    print("Testing vine recursion...")
    
    # This is a placeholder - proper testing requires fitted copulas
    from .structure import build_rvine_structure
    
    np.random.seed(42)
    n, d = 100, 4
    U = np.random.uniform(0, 1, size=(n, d))
    
    # Build structure
    structure = build_rvine_structure(U)
    
    # Create recursion object
    vine = VineRecursion(structure)
    
    print(f"Created vine recursion for d={d}")
    print(f"Structure has {len(structure.trees)} trees")
    
    print("\nVine recursion test completed!")
