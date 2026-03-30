"""
Vine recursion: h-function transforms, Rosenblatt, and inverse sampling.

Implements the recursive algorithm for:
- Computing conditional distributions via h-functions
- Forward Rosenblatt transform (data → uniforms)
- Inverse Rosenblatt transform (uniforms → data) for sampling
"""

import numpy as np
from typing import Dict, FrozenSet, List, Optional, Tuple
from dataclasses import dataclass

from .structure import VineStructure
from ..models.hfunc import HFuncLookup


@dataclass
class VinePairCopula:
    """
    A fitted pair-copula with h-function lookups.
    
    Attributes:
        edge: (i, j, conditioning_set) tuple
        density_grid: (m, m) numpy array of copula density
        hfunc: HFuncLookup object for this pair
        level: Tree level (0 = Tree 1)
    """
    edge: Tuple[int, int, set[int]]
    density_grid: np.ndarray
    hfunc: HFuncLookup
    level: int
    
    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate density at points (u, v)."""
        return self.hfunc.pdf(u, v)
    
    def h_u_given_v(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """h(u|v) = ∂C(u,v)/∂v."""
        return self.hfunc.h_u_given_v(u, v)
    
    def h_v_given_u(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """h(v|u) = ∂C(u,v)/∂u."""
        return self.hfunc.h_v_given_u(v, u)
    
    def hinv_u_given_v(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inverse h-function: solve h(u|v) = q for u."""
        return self.hfunc.hinv_u_given_v(q, v)
    
    def hinv_v_given_u(self, q: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Inverse h-function: solve h(v|u) = q for v."""
        return self.hfunc.hinv_v_given_u(q, u)


class VineRecursion:
    """
    Handles h-function recursion for vine copulas.
    
    This class manages:
    - Transforming data through vine trees via h-functions
    - Computing joint densities via the vine decomposition
    - Forward/inverse Rosenblatt transforms for sampling
    """
    
    def __init__(self, structure: VineStructure, vine_type: Optional[str] = None):
        """
        Args:
            structure: VineStructure defining the vine
        """
        self.structure = structure
        self.vine_type = (vine_type or "unknown").lower()
        self.pair_copulas: List[List[VinePairCopula]] = []  # [tree_level][edge_idx]
        self.d = structure.d
        
        # Fast lookup: (unordered_pair, conditioning_set) -> copula
        self._copula_lookup: Dict[Tuple[FrozenSet[int], FrozenSet[int]], VinePairCopula] = {}
    
    def set_pair_copulas(self, pair_copulas: List[List[VinePairCopula]]):
        """Set fitted pair copulas for all edges."""
        self.pair_copulas = pair_copulas
        self._rebuild_lookup()
    
    def add_pair_copula(self, copula: VinePairCopula):
        """Add a fitted pair copula."""
        level = copula.level
        while len(self.pair_copulas) <= level:
            self.pair_copulas.append([])
        self.pair_copulas[level].append(copula)
        self._add_to_lookup(copula)
    
    # ------------------------------------------------------------------
    # Internal lookup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _key(i: int, j: int, cond: FrozenSet[int]) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        return (frozenset((i, j)), cond)

    def _rebuild_lookup(self) -> None:
        self._copula_lookup = {}
        for level_copulas in self.pair_copulas:
            for copula in level_copulas:
                self._add_to_lookup(copula)

    def _add_to_lookup(self, copula: VinePairCopula) -> None:
        i, j, cond = copula.edge
        k = self._key(i, j, frozenset(cond))
        if k in self._copula_lookup:
            raise ValueError(f"Duplicate pair-copula for edge ({i},{j}|{sorted(cond)})")
        self._copula_lookup[k] = copula

    def get_pair_copula(self, i: int, j: int, cond: FrozenSet[int]) -> VinePairCopula:
        cop = self._copula_lookup.get(self._key(i, j, cond))
        if cop is None:
            raise KeyError(f"Missing pair-copula for ({i},{j}|{sorted(cond)})")
        return cop

    @staticmethod
    def _clip01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.clip(x, eps, 1.0 - eps)

    def _hinv_target_given_other(
        self,
        copula: VinePairCopula,
        target: int,
        other: int,
        q: np.ndarray,
        other_val: np.ndarray,
    ) -> np.ndarray:
        """Invert conditional CDF of target given other under this copula's orientation."""
        i, j, _ = copula.edge
        if target == i and other == j:
            return copula.hinv_u_given_v(q, other_val)
        if target == j and other == i:
            return copula.hinv_v_given_u(q, other_val)
        raise ValueError(f"Copula edge {copula.edge} does not connect ({target},{other})")

    def _h_target_given_other(
        self,
        copula: VinePairCopula,
        target: int,
        other: int,
        target_val: np.ndarray,
        other_val: np.ndarray,
    ) -> np.ndarray:
        """Compute conditional CDF h(target | other) under this copula's orientation."""
        i, j, _ = copula.edge
        if target == i and other == j:
            return copula.h_u_given_v(target_val, other_val)
        if target == j and other == i:
            return copula.h_v_given_u(target_val, other_val)
        raise ValueError(f"Copula edge {copula.edge} does not connect ({target},{other})")
    
    def compute_h_transforms(
        self,
        U: np.ndarray,
        tree_level: int,
        prev_transforms: Optional[Dict[Tuple[int, FrozenSet[int]], np.ndarray]] = None,
    ) -> Dict[Tuple[int, FrozenSet[int]], np.ndarray]:
        """
        Compute h-function transforms for a given tree level.
        
        Args:
            U: (n, d) original pseudo-observations
            tree_level: Which tree level (0 = Tree 1)
            prev_transforms: Dictionary of h-transforms from previous levels
            
        Returns:
            Dictionary mapping (var, conditioning_set) -> transformed values
        """
        if tree_level >= len(self.pair_copulas):
            return {}
        
        transforms: Dict[Tuple[int, FrozenSet[int]], np.ndarray] = {}
        tree_copulas = self.pair_copulas[tree_level]

        if tree_level > 0 and prev_transforms is None:
            raise ValueError("prev_transforms must be provided for tree_level > 0")

        for copula in tree_copulas:
            i, j, cond_set = copula.edge
            D = frozenset(cond_set)

            if tree_level == 0:
                u_i_D = U[:, i]
                u_j_D = U[:, j]
            else:
                assert prev_transforms is not None
                try:
                    u_i_D = prev_transforms[(i, D)]
                    u_j_D = prev_transforms[(j, D)]
                except KeyError as e:
                    raise RuntimeError(
                        f"Missing conditional pseudo-observations for edge ({i},{j}|{sorted(D)}) "
                        f"at tree_level={tree_level}. This usually indicates an invalid structure "
                        f"or a bug in previous transform propagation."
                    ) from e

            # Compute both h-functions (produce conditioning sets of size |D|+1)
            h_i_given_j = self._clip01(copula.h_u_given_v(u_i_D, u_j_D))  # i | (D ∪ {j})
            h_j_given_i = self._clip01(copula.h_v_given_u(u_j_D, u_i_D))  # j | (D ∪ {i})

            transforms[(i, D | frozenset({j}))] = h_i_given_j
            transforms[(j, D | frozenset({i}))] = h_j_given_i

        return transforms
    
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
        if not self.pair_copulas:
            raise RuntimeError("No pair copulas set on VineRecursion; call fit() or set_pair_copulas().")

        U = np.asarray(U, dtype=np.float64)
        n, d = U.shape
        if d != self.d:
            raise ValueError(f"Expected U with d={self.d}, got shape {U.shape}")

        # Conditional pseudo-observations cache: (var, cond_set) -> array
        cond_cache: Dict[Tuple[int, FrozenSet[int]], np.ndarray] = {
            (i, frozenset()): self._clip01(U[:, i]) for i in range(d)
        }

        logpdf = np.zeros(n, dtype=np.float64)

        for tree_level, tree_copulas in enumerate(self.pair_copulas):
            for copula in tree_copulas:
                i, j, cond_set = copula.edge
                D = frozenset(cond_set)

                try:
                    u_i_D = cond_cache[(i, D)]
                    u_j_D = cond_cache[(j, D)]
                except KeyError as e:
                    raise RuntimeError(
                        f"Missing conditional pseudo-observations for edge ({i},{j}|{sorted(D)}) "
                        f"at tree_level={tree_level}. This indicates an invalid structure or a "
                        f"bug in transform propagation."
                    ) from e

                pair_pdf = copula.pdf(u_i_D, u_j_D)
                logpdf += np.log(np.clip(pair_pdf, 1e-12, None))

                # Propagate conditionals to the next level
                cond_cache[(i, D | frozenset({j}))] = self._clip01(copula.h_u_given_v(u_i_D, u_j_D))
                cond_cache[(j, D | frozenset({i}))] = self._clip01(copula.h_v_given_u(u_j_D, u_i_D))

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
        if self.vine_type not in {"dvine", "cvine"}:
            raise NotImplementedError(
                f"rosenblatt() is implemented for D-vines / C-vines only (got vine_type='{self.vine_type}')."
            )
        if self.structure.order is None:
            raise RuntimeError("structure.order is required for Rosenblatt transform")

        U = np.asarray(U, dtype=np.float64)
        n, d = U.shape
        if d != self.d:
            raise ValueError(f"Expected U with d={self.d}, got shape {U.shape}")

        order = list(self.structure.order)

        # Compute conditional pseudo-observations for all edges
        cond_cache: Dict[Tuple[int, FrozenSet[int]], np.ndarray] = {
            (i, frozenset()): self._clip01(U[:, i]) for i in range(d)
        }
        for tree_level, tree_copulas in enumerate(self.pair_copulas):
            for copula in tree_copulas:
                i, j, cond_set = copula.edge
                D = frozenset(cond_set)
                u_i_D = cond_cache[(i, D)]
                u_j_D = cond_cache[(j, D)]
                cond_cache[(i, D | frozenset({j}))] = self._clip01(copula.h_u_given_v(u_i_D, u_j_D))
                cond_cache[(j, D | frozenset({i}))] = self._clip01(copula.h_v_given_u(u_j_D, u_i_D))

        W = np.zeros_like(U)
        for k, var in enumerate(order):
            if k == 0:
                W[:, var] = cond_cache[(var, frozenset())]
                continue
            D = frozenset(order[:k])
            key = (var, D)
            if key not in cond_cache:
                raise RuntimeError(
                    f"Missing Rosenblatt conditional for var={var} given {sorted(D)}. "
                    f"Likely the vine is truncated or inconsistent with the requested order."
                )
            W[:, var] = cond_cache[key]

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
        if self.vine_type == "cvine":
            return self._inverse_rosenblatt_cvine(W)
        if self.vine_type == "dvine":
            return self._inverse_rosenblatt_dvine(W)
        raise NotImplementedError(
            f"inverse_rosenblatt() is implemented for D-vines / C-vines only (got vine_type='{self.vine_type}')."
        )

    def _inverse_rosenblatt_cvine(self, W: np.ndarray) -> np.ndarray:
        """Inverse Rosenblatt for a C-vine with order = root sequence."""
        if self.structure.order is None:
            raise RuntimeError("structure.order is required for C-vine inverse Rosenblatt")
        W = np.asarray(W, dtype=np.float64)
        n, d = W.shape
        if d != self.d:
            raise ValueError(f"Expected W with d={self.d}, got shape {W.shape}")

        order = list(self.structure.order)
        U = np.zeros_like(W)

        # First variable: w = u
        v0 = order[0]
        U[:, v0] = self._clip01(W[:, v0])

        # Subsequent variables: peel off conditioning roots from last to first.
        for j_pos in range(1, d):
            var = order[j_pos]
            q = self._clip01(W[:, var])  # u_{var | order[:j_pos]}

            for k_pos in range(j_pos - 1, -1, -1):
                root = order[k_pos]
                D_prev = frozenset(order[:k_pos])  # roots before `root`
                root_val = self._clip01(W[:, root]) if k_pos > 0 else self._clip01(U[:, root])

                cop = self.get_pair_copula(root, var, D_prev)
                q = self._clip01(self._hinv_target_given_other(cop, target=var, other=root, q=q, other_val=root_val))

            U[:, var] = q

        return U

    def _inverse_rosenblatt_dvine(self, W: np.ndarray) -> np.ndarray:
        """
        Inverse Rosenblatt for a D-vine with order = path order.

        Uses a triangular cache:
          C[i][j] = u_{order[i] | order[i+1..j]} for i <= j.
        """
        if self.structure.order is None:
            raise RuntimeError("structure.order is required for D-vine inverse Rosenblatt")
        W = np.asarray(W, dtype=np.float64)
        n, d = W.shape
        if d != self.d:
            raise ValueError(f"Expected W with d={self.d}, got shape {W.shape}")

        order = list(self.structure.order)
        U = np.zeros_like(W)

        # Triangular cache of conditionals C[i][j] (object dtype: arrays of shape (n,))
        C: List[List[Optional[np.ndarray]]] = [[None for _ in range(d)] for _ in range(d)]

        # Sample first variable: w = u
        v0 = order[0]
        u0 = self._clip01(W[:, v0])
        U[:, v0] = u0
        C[0][0] = u0

        # Sample sequentially along the path
        for k in range(1, d):
            var = order[k]
            q = self._clip01(W[:, var])  # u_{var | order[:k]}

            # Store intermediate conditionals of var given suffix blocks:
            # after removing order[i], q becomes u_{var | order[i+1..k-1]}.
            q_after: List[Optional[np.ndarray]] = [None for _ in range(k)]

            # Remove conditioning variables one by one from left to right
            for i in range(0, k):
                root = order[i]
                D_between = frozenset(order[i + 1 : k])  # variables between root and var

                # root conditional given variables between root and var
                if i == k - 1:
                    root_val = C[i][i]
                else:
                    root_val = C[i][k - 1]
                if root_val is None:
                    raise RuntimeError(
                        f"Missing D-vine cache C[{i}][{k-1}] when sampling position k={k}."
                    )

                cop = self.get_pair_copula(root, var, D_between)
                q = self._clip01(
                    self._hinv_target_given_other(cop, target=var, other=root, q=q, other_val=root_val)
                )
                q_after[i] = q

            # Unconditional sample for var
            U[:, var] = q
            C[k][k] = q

            # Update caches C[i][k] = u_{order[i] | order[i+1..k]}
            for i in range(k - 1, -1, -1):
                root = order[i]
                D_between = frozenset(order[i + 1 : k])  # variables between root and var (excludes var)

                if i == k - 1:
                    u_root_D = C[i][i]
                else:
                    u_root_D = C[i][k - 1]
                if u_root_D is None:
                    raise RuntimeError(f"Missing cache C[{i}][{k-1}] while updating C[*][{k}].")

                u_var_D = q_after[i]
                if u_var_D is None:
                    raise RuntimeError(f"Missing q_after[{i}] while updating C[*][{k}].")

                cop = self.get_pair_copula(root, var, D_between)
                C[i][k] = self._clip01(
                    self._h_target_given_other(cop, target=root, other=var, target_val=u_root_D, other_val=u_var_D)
                )

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
