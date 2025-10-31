"""
Vine copula structure selection.

Implements the Dißmann et al. (2013) algorithm for building
Regular Vine (R-Vine) structures using maximum spanning trees
on Kendall's tau correlations.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import networkx as nx


@dataclass
class VineTree:
    """
    Represents one tree in the vine hierarchy.
    
    Attributes:
        level: Tree level (0 = Tree 1, 1 = Tree 2, etc.)
        edges: List of edges, each edge is (node_i, node_j, conditioning_set)
        tau_values: Kendall's tau for each edge
    """
    level: int
    edges: List[Tuple[int, int, Set[int]]]
    tau_values: List[float]


@dataclass
class VineStructure:
    """
    Complete vine structure specification.
    
    Attributes:
        d: Dimension
        trees: List of VineTree objects
        order: Variable ordering (for D-vines and C-vines)
    """
    d: int
    trees: List[VineTree]
    order: Optional[List[int]] = None
    
    def __len__(self) -> int:
        return len(self.trees)
    
    def num_edges(self) -> int:
        return sum(len(tree.edges) for tree in self.trees)


def build_rvine_structure(
    U: np.ndarray,
    method: str = 'dissmann',
    truncation_level: Optional[int] = None,
) -> VineStructure:
    """
    Build R-vine structure using maximum spanning tree on |τ|.
    
    Follows the Dißmann et al. (2013) algorithm:
    1. Tree 1: MST on |τ(U_i, U_j)| for all pairs
    2. Tree k: MST on |τ(h_i|D_i, h_j|D_j)| where D_i, D_j are conditioning sets
    
    Args:
        U: (n, d) pseudo-observations in [0,1]
        method: 'dissmann' (maximum spanning tree on |τ|)
        truncation_level: Truncate vine at this level (None = full vine)
        
    Returns:
        VineStructure object
    """
    from ..utils.stats import kendall_tau_matrix
    
    n, d = U.shape
    
    if truncation_level is None:
        truncation_level = d - 1
    else:
        truncation_level = min(truncation_level, d - 1)
    
    trees = []
    current_data = U.copy()  # Will be transformed via h-functions
    
    # Variables available at each level
    variables = list(range(d))
    
    for level in range(truncation_level):
        if level == 0:
            # Tree 1: Build MST on original data
            tree, current_data = _build_tree_1(current_data, variables)
        else:
            # Tree k: Build MST on transformed data
            tree, current_data = _build_tree_k(
                current_data, trees, variables, level
            )
        
        trees.append(tree)
        
        # Check if we can continue
        if len(tree.edges) == 0:
            break
    
    return VineStructure(d=d, trees=trees)


def _build_tree_1(U: np.ndarray, variables: List[int]) -> Tuple[VineTree, np.ndarray]:
    """
    Build first tree using MST on |Kendall's tau|.
    
    Args:
        U: (n, d) pseudo-observations
        variables: List of variable indices
        
    Returns:
        (tree, transformed_data) where transformed_data will be used for next tree
    """
    from ..utils.stats import kendall_tau_matrix
    
    n, d = U.shape
    
    # Compute Kendall's tau matrix
    tau_matrix = kendall_tau_matrix(U)
    
    # Use absolute tau for MST weights (we want strong dependence)
    abs_tau = np.abs(tau_matrix)
    
    # Convert to distance (maximize |tau| = minimize -|tau|)
    distance_matrix = 1.0 - abs_tau
    np.fill_diagonal(distance_matrix, 0)
    
    # Find MST using Kruskal/Prim
    mst = minimum_spanning_tree(csr_matrix(distance_matrix))
    mst = mst.toarray()
    
    # Extract edges from MST
    edges = []
    tau_values = []
    
    for i in range(d):
        for j in range(i + 1, d):
            if mst[i, j] > 0 or mst[j, i] > 0:
                # Edge exists in MST
                edges.append((i, j, set()))  # Empty conditioning set for Tree 1
                tau_values.append(tau_matrix[i, j])
    
    tree = VineTree(level=0, edges=edges, tau_values=tau_values)
    
    # For next tree, we don't transform yet (done in recursion)
    return tree, U


def _build_tree_k(
    U_transformed: np.ndarray,
    previous_trees: List[VineTree],
    variables: List[int],
    level: int,
) -> Tuple[VineTree, np.ndarray]:
    """
    Build tree k (k ≥ 2) using MST on conditional correlations.
    
    This is a simplified version. In practice, you would:
    1. Identify possible edges based on proximity condition
    2. Compute conditional tau for each possible edge
    3. Build MST on those edges
    
    Args:
        U_transformed: Transformed data from previous level
        previous_trees: List of trees built so far
        variables: Available variables
        level: Current tree level (0-indexed)
        
    Returns:
        (tree, transformed_data)
    """
    from ..utils.stats import kendall_tau
    
    # Get edges from previous tree
    prev_tree = previous_trees[-1]
    
    # Identify possible edges based on proximity condition
    # For an R-vine: can connect two edges from Tree k-1 if they share a common node
    possible_edges = []
    
    for i, edge_i in enumerate(prev_tree.edges):
        for j, edge_j in enumerate(prev_tree.edges):
            if i >= j:
                continue
            
            # Check if edges share exactly one node (proximity condition)
            nodes_i = set([edge_i[0], edge_i[1]])
            nodes_j = set([edge_j[0], edge_j[1]])
            common = nodes_i & nodes_j
            
            if len(common) == 1:
                # Can form an edge
                all_nodes = nodes_i | nodes_j
                conditioning_set = common  # The shared node becomes part of conditioning
                
                # The edge connects the two non-common nodes
                non_common = list(all_nodes - common)
                if len(non_common) == 2:
                    possible_edges.append((
                        non_common[0],
                        non_common[1],
                        conditioning_set | edge_i[2] | edge_j[2],  # Add previous conditioning
                        i,  # index in prev tree
                        j,  # index in prev tree
                    ))
    
    if len(possible_edges) == 0:
        # No more edges can be formed
        return VineTree(level=level, edges=[], tau_values=[]), U_transformed
    
    # Compute tau for each possible edge (using transformed data)
    tau_values_dict = {}
    
    for idx, (node_i, node_j, cond_set, prev_i, prev_j) in enumerate(possible_edges):
        # Use the transformed variables from previous tree
        # In practice, these would be h-function outputs
        if prev_i < U_transformed.shape[1] and prev_j < U_transformed.shape[1]:
            tau = kendall_tau(U_transformed[:, prev_i], U_transformed[:, prev_j])
            tau_values_dict[idx] = tau
        else:
            tau_values_dict[idx] = 0.0
    
    # Build MST on |tau|
    n_possible = len(possible_edges)
    if n_possible == 0:
        return VineTree(level=level, edges=[], tau_values=[]), U_transformed
    
    # Create distance matrix for possible edges
    # Map edge indices to matrix indices
    distance = np.ones((n_possible, n_possible)) * 1e6
    np.fill_diagonal(distance, 0)
    
    for i in range(n_possible):
        for j in range(i + 1, n_possible):
            # Check if these edges can coexist (simplified)
            distance[i, j] = 1.0 - abs(tau_values_dict.get(i, 0))
            distance[j, i] = distance[i, j]
    
    # MST
    if n_possible > 1:
        mst = minimum_spanning_tree(csr_matrix(distance))
        mst = mst.toarray()
        
        # Extract selected edges
        selected = []
        for i in range(n_possible):
            for j in range(i + 1, n_possible):
                if mst[i, j] > 0 or mst[j, i] > 0:
                    selected.append(i)
    else:
        selected = [0] if n_possible == 1 else []
    
    # Build tree
    edges = []
    tau_values = []
    
    for idx in selected:
        node_i, node_j, cond_set, _, _ = possible_edges[idx]
        edges.append((node_i, node_j, cond_set))
        tau_values.append(tau_values_dict[idx])
    
    tree = VineTree(level=level, edges=edges, tau_values=tau_values)
    
    return tree, U_transformed


def get_vine_order(structure: VineStructure) -> List[int]:
    """
    Extract variable ordering from vine structure.
    
    For D-vines and C-vines, there's a natural ordering.
    For R-vines, this returns one possible ordering.
    
    Args:
        structure: VineStructure object
        
    Returns:
        List of variable indices in vine order
    """
    if structure.order is not None:
        return structure.order
    
    # Extract from first tree
    if len(structure.trees) == 0:
        return list(range(structure.d))
    
    # Simple heuristic: depth-first traversal of first tree
    first_tree = structure.trees[0]
    
    if len(first_tree.edges) == 0:
        return list(range(structure.d))
    
    # Build graph
    G = nx.Graph()
    for edge in first_tree.edges:
        G.add_edge(edge[0], edge[1])
    
    # DFS from highest degree node
    degrees = dict(G.degree())
    start_node = max(degrees, key=degrees.get)
    
    order = list(nx.dfs_preorder(G, start_node))
    
    # Add any missing variables
    all_vars = set(range(structure.d))
    order.extend(list(all_vars - set(order)))
    
    return order


def print_vine_structure(structure: VineStructure):
    """Pretty-print vine structure."""
    print(f"Vine Structure (d={structure.d})")
    print(f"Number of trees: {len(structure.trees)}")
    print(f"Total edges: {structure.num_edges()}")
    print()
    
    for tree in structure.trees:
        print(f"Tree {tree.level + 1} ({len(tree.edges)} edges):")
        for i, (edge, tau) in enumerate(zip(tree.edges, tree.tau_values)):
            node_i, node_j, cond_set = edge
            if len(cond_set) == 0:
                print(f"  Edge {i+1}: ({node_i}, {node_j}) | {} — τ = {tau:.4f}")
            else:
                cond_str = ", ".join(map(str, sorted(cond_set)))
                print(f"  Edge {i+1}: ({node_i}, {node_j}) | {{{cond_str}}} — τ = {tau:.4f}")
        print()


if __name__ == "__main__":
    print("Testing vine structure selection...")
    
    # Generate synthetic data
    np.random.seed(42)
    n, d = 500, 5
    
    # Create correlation structure
    rho = 0.6
    Sigma = np.eye(d)
    for i in range(d - 1):
        Sigma[i, i + 1] = rho
        Sigma[i + 1, i] = rho
    
    # Generate Gaussian data
    from scipy.stats import norm
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    
    # Convert to uniforms
    U = norm.cdf(Z)
    
    print(f"Generated {n} samples in {d} dimensions")
    print(f"U shape: {U.shape}")
    print(f"U range: [{U.min():.4f}, {U.max():.4f}]")
    
    # Build vine structure
    print("\nBuilding vine structure...")
    structure = build_rvine_structure(U, method='dissmann')
    
    # Print structure
    print_vine_structure(structure)
    
    # Get ordering
    order = get_vine_order(structure)
    print(f"Variable ordering: {order}")
    
    print("\nVine structure test completed!")
