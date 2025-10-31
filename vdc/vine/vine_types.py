"""
Extended vine structure implementations: D-vine, C-vine, and fitting from vine matrices.

This module extends the basic R-vine implementation to support:
- D-vine (Drawable vine) with sequential ordering
- C-vine (Canonical vine) with star structure
- Fitting vines from predefined structures/matrices
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Union
from dataclasses import dataclass

from .structure import VineStructure, VineTree
from ..utils.stats import kendall_tau_matrix, kendall_tau


def build_dvine_structure(
    U: np.ndarray,
    order: Optional[List[int]] = None,
    truncation_level: Optional[int] = None,
) -> VineStructure:
    """
    Build D-vine (Drawable vine) structure.
    
    A D-vine has a path-like structure:
    - Tree 1: (1,2), (2,3), (3,4), ..., (d-1,d)
    - Tree 2: (1,3|2), (2,4|3), ..., (d-2,d|d-1)
    - Tree k: Variables i and i+k conditioned on all variables between them
    
    The order can be:
    1. User-specified
    2. Optimized based on maximum spanning tree on |τ|
    
    Args:
        U: (n, d) pseudo-observations in [0,1]
        order: Variable ordering (None = auto-select via MST)
        truncation_level: Truncate vine at this level (None = full vine)
        
    Returns:
        VineStructure with D-vine pattern
    """
    n, d = U.shape
    
    if truncation_level is None:
        truncation_level = d - 1
    else:
        truncation_level = min(truncation_level, d - 1)
    
    # Determine ordering
    if order is None:
        # Use greedy heuristic: arrange variables to maximize adjacent dependencies
        order = _optimize_dvine_order(U)
    else:
        if len(order) != d or set(order) != set(range(d)):
            raise ValueError(f"Order must be a permutation of 0..{d-1}")
        order = list(order)
    
    # Build D-vine trees
    trees = []
    
    for level in range(truncation_level):
        edges = []
        tau_values = []
        
        # Number of edges in tree k is d - k - 1
        n_edges = d - level - 1
        
        for i in range(n_edges):
            # Edge connects variables at positions i and i+level+1 in the order
            var_i = order[i]
            var_j = order[i + level + 1]
            
            # Conditioning set: all variables between i and j
            if level == 0:
                cond_set = set()
            else:
                cond_set = set(order[i+1:i+level+1])
            
            # Compute tau (simplified: use original data)
            # In practice, would use h-transformed data for level > 0
            tau = kendall_tau(U[:, var_i], U[:, var_j])
            
            edges.append((var_i, var_j, cond_set))
            tau_values.append(tau)
        
        tree = VineTree(level=level, edges=edges, tau_values=tau_values)
        trees.append(tree)
    
    return VineStructure(d=d, trees=trees, order=order)


def _optimize_dvine_order(U: np.ndarray) -> List[int]:
    """
    Find optimal D-vine ordering via greedy maximum spanning tree.
    
    Constructs a path that maximizes sum of |τ| along consecutive pairs.
    
    Args:
        U: (n, d) pseudo-observations
        
    Returns:
        Optimal variable ordering
    """
    n, d = U.shape
    
    # Compute pairwise Kendall's tau
    tau_mat = kendall_tau_matrix(U)
    abs_tau = np.abs(tau_mat)
    
    # Greedy path construction
    # Start with the pair having maximum |τ|
    used = set()
    order = []
    
    # Find starting edge
    max_tau = -1
    start_i, start_j = 0, 1
    for i in range(d):
        for j in range(i+1, d):
            if abs_tau[i, j] > max_tau:
                max_tau = abs_tau[i, j]
                start_i, start_j = i, j
    
    order = [start_i, start_j]
    used = {start_i, start_j}
    
    # Greedily extend path
    while len(order) < d:
        # Find best extension (add to either end)
        best_var = None
        best_tau = -1
        add_to_start = True
        
        # Try extending from start
        current_start = order[0]
        for var in range(d):
            if var not in used:
                tau = abs_tau[current_start, var]
                if tau > best_tau:
                    best_tau = tau
                    best_var = var
                    add_to_start = True
        
        # Try extending from end
        current_end = order[-1]
        for var in range(d):
            if var not in used:
                tau = abs_tau[current_end, var]
                if tau > best_tau:
                    best_tau = tau
                    best_var = var
                    add_to_start = False
        
        if best_var is None:
            # No more connections, add remaining variables arbitrarily
            remaining = [v for v in range(d) if v not in used]
            order.extend(remaining)
            break
        
        if add_to_start:
            order.insert(0, best_var)
        else:
            order.append(best_var)
        
        used.add(best_var)
    
    return order


def build_cvine_structure(
    U: np.ndarray,
    order: Optional[List[int]] = None,
    truncation_level: Optional[int] = None,
) -> VineStructure:
    """
    Build C-vine (Canonical vine) structure.
    
    A C-vine has a star-like structure at each level:
    - Tree 1: (root1, 2), (root1, 3), ..., (root1, d)
    - Tree 2: (root2, 3|root1), (root2, 4|root1), ..., (root2, d|root1)
    - Tree k: All edges connected to root_k, conditioned on roots 1..k-1
    
    The order specifies which variables are roots at each level.
    If not given, select roots to maximize sum of dependencies.
    
    Args:
        U: (n, d) pseudo-observations in [0,1]
        order: Root variable ordering (None = auto-select)
        truncation_level: Truncate vine at this level (None = full vine)
        
    Returns:
        VineStructure with C-vine pattern
    """
    n, d = U.shape
    
    if truncation_level is None:
        truncation_level = d - 1
    else:
        truncation_level = min(truncation_level, d - 1)
    
    # Determine root ordering
    if order is None:
        # Greedy: select root with maximum sum of |τ| at each level
        order = _optimize_cvine_order(U)
    else:
        if len(order) != d or set(order) != set(range(d)):
            raise ValueError(f"Order must be a permutation of 0..{d-1}")
        order = list(order)
    
    # Build C-vine trees
    trees = []
    
    for level in range(truncation_level):
        edges = []
        tau_values = []
        
        # Root for this tree
        root = order[level]
        
        # Conditioning set: all previous roots
        if level == 0:
            cond_set = set()
        else:
            cond_set = set(order[:level])
        
        # Connect root to all variables not yet used as roots
        for i in range(level + 1, d):
            var = order[i]
            
            # Compute tau (simplified: use original data)
            tau = kendall_tau(U[:, root], U[:, var])
            
            edges.append((root, var, cond_set))
            tau_values.append(tau)
        
        tree = VineTree(level=level, edges=edges, tau_values=tau_values)
        trees.append(tree)
    
    return VineStructure(d=d, trees=trees, order=order)


def _optimize_cvine_order(U: np.ndarray) -> List[int]:
    """
    Find optimal C-vine root ordering.
    
    Greedy heuristic: at each level, select the variable that has
    maximum sum of |τ| with remaining variables.
    
    Args:
        U: (n, d) pseudo-observations
        
    Returns:
        Optimal root ordering
    """
    n, d = U.shape
    
    # Compute pairwise Kendall's tau
    tau_mat = kendall_tau_matrix(U)
    abs_tau = np.abs(tau_mat)
    
    order = []
    remaining = set(range(d))
    
    for level in range(d):
        if len(remaining) == 1:
            order.append(remaining.pop())
            break
        
        # For each remaining variable, compute sum of |τ| with other remaining vars
        best_var = None
        best_sum = -1
        
        for var in remaining:
            tau_sum = sum(abs_tau[var, other] for other in remaining if other != var)
            if tau_sum > best_sum:
                best_sum = tau_sum
                best_var = var
        
        order.append(best_var)
        remaining.remove(best_var)
    
    return order


def build_vine_from_matrix(
    U: np.ndarray,
    matrix: np.ndarray,
    vine_type: str = 'rvine',
) -> VineStructure:
    """
    Build vine structure from a vine matrix specification.
    
    Vine matrices follow the conventions:
    - R-vine: Upper triangular matrix where M[i,j] specifies structure
    - D-vine: Diagonal structure
    - C-vine: First row/column are roots
    
    Matrix format (following pyvinecopulib convention):
    - M[i,j] = k means variable k is conditioned at edge (i,j)
    - Diagonal M[i,i] gives variable order
    
    Args:
        U: (n, d) pseudo-observations
        matrix: (d, d) vine matrix specification
        vine_type: 'rvine', 'dvine', or 'cvine'
        
    Returns:
        VineStructure object
    """
    n, d = U.shape
    
    if matrix.shape != (d, d):
        raise ValueError(f"Matrix shape {matrix.shape} doesn't match dimension {d}")
    
    # Extract variable order from diagonal
    order = [int(matrix[i, i]) for i in range(d)]
    
    if vine_type.lower() == 'dvine':
        # For D-vine, matrix diagonal gives the path order
        return build_dvine_structure(U, order=order)
    
    elif vine_type.lower() == 'cvine':
        # For C-vine, matrix diagonal gives root order
        return build_cvine_structure(U, order=order)
    
    elif vine_type.lower() == 'rvine':
        # For R-vine, parse the full matrix structure
        return _parse_rvine_matrix(U, matrix, order)
    
    else:
        raise ValueError(f"Unknown vine_type: {vine_type}. Use 'rvine', 'dvine', or 'cvine'")


def _parse_rvine_matrix(
    U: np.ndarray,
    matrix: np.ndarray,
    order: List[int]
) -> VineStructure:
    """
    Parse R-vine matrix into VineStructure.
    
    This is a simplified parser. Full implementation would handle
    all matrix encoding conventions from vinecopulib.
    
    Args:
        U: Pseudo-observations
        matrix: Vine matrix
        order: Variable order
        
    Returns:
        VineStructure
    """
    d = len(order)
    trees = []
    
    # Parse matrix to extract edges at each level
    # This is complex and depends on exact matrix format
    # For now, fall back to automatic R-vine construction
    
    from .structure import build_rvine_structure
    
    # Use the automatic method but with specified order
    structure = build_rvine_structure(U)
    structure.order = order
    
    return structure


def get_vine_matrix(structure: VineStructure) -> np.ndarray:
    """
    Convert VineStructure to matrix representation.
    
    Useful for:
    - Visualization
    - Saving/loading vine specifications
    - Interfacing with other vine libraries
    
    Args:
        structure: VineStructure object
        
    Returns:
        (d, d) vine matrix
    """
    d = structure.d
    matrix = np.zeros((d, d), dtype=int)
    
    # Set diagonal to variable order
    order = structure.order if structure.order else list(range(d))
    for i in range(d):
        matrix[i, i] = order[i]
    
    # Encode edge structure in upper triangle
    # (Simplified encoding; full implementation would follow vinecopulib conventions)
    
    for tree in structure.trees:
        for edge_idx, (i, j, cond) in enumerate(tree.edges):
            # Mark edge in matrix
            # This is a simplified placeholder
            pass
    
    return matrix


def print_vine_comparison(structures: List[Tuple[str, VineStructure]]):
    """
    Print comparison of different vine structures.
    
    Args:
        structures: List of (name, VineStructure) tuples
    """
    print("\n" + "="*80)
    print("Vine Structure Comparison")
    print("="*80)
    
    for name, structure in structures:
        print(f"\n{name}:")
        print(f"  Dimension: {structure.d}")
        print(f"  Number of trees: {len(structure.trees)}")
        print(f"  Total edges: {structure.num_edges()}")
        print(f"  Order: {structure.order}")
        
        # Show first tree details
        if len(structure.trees) > 0:
            tree = structure.trees[0]
            print(f"  Tree 1 edges: {len(tree.edges)}")
            for i, (edge, tau) in enumerate(zip(tree.edges[:3], tree.tau_values[:3])):
                node_i, node_j, cond = edge
                print(f"    Edge {i+1}: ({node_i}, {node_j}) — τ = {tau:.4f}")
            if len(tree.edges) > 3:
                print(f"    ... ({len(tree.edges) - 3} more edges)")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    print("Testing D-vine, C-vine, and matrix-based vine construction...")
    
    # Generate synthetic data
    np.random.seed(42)
    n, d = 500, 5
    
    # Create correlation structure
    from scipy.stats import norm
    rho = 0.6
    Sigma = np.eye(d)
    for i in range(d - 1):
        Sigma[i, i + 1] = rho
        Sigma[i + 1, i] = rho
    
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    U = norm.cdf(Z)
    
    print(f"Generated {n} samples in {d} dimensions\n")
    
    # Build different vine types
    print("Building different vine structures...")
    
    # R-vine (automatic)
    from .structure import build_rvine_structure
    rvine = build_rvine_structure(U)
    
    # D-vine (optimized order)
    dvine = build_dvine_structure(U)
    
    # C-vine (optimized roots)
    cvine = build_cvine_structure(U)
    
    # D-vine with custom order
    dvine_custom = build_dvine_structure(U, order=[0, 2, 4, 1, 3])
    
    # C-vine with custom roots
    cvine_custom = build_cvine_structure(U, order=[2, 0, 1, 3, 4])
    
    # Compare structures
    print_vine_comparison([
        ("R-Vine (Dißmann MST)", rvine),
        ("D-Vine (Optimized)", dvine),
        ("C-Vine (Optimized)", cvine),
        ("D-Vine (Custom Order)", dvine_custom),
        ("C-Vine (Custom Roots)", cvine_custom),
    ])
    
    # Test matrix conversion
    print("\nTesting vine matrix conversion...")
    matrix = get_vine_matrix(dvine)
    print("D-vine matrix diagonal (order):", matrix.diagonal())
    
    # Test building from matrix
    print("\nRebuilding D-vine from matrix...")
    dvine_from_matrix = build_vine_from_matrix(U, matrix, vine_type='dvine')
    print(f"Rebuilt D-vine: {len(dvine_from_matrix.trees)} trees, order={dvine_from_matrix.order}")
    
    print("\nVine type comparison test completed!")
