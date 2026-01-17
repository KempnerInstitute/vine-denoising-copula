"""
High-level API for fitting vine copulas with different structures.

Provides convenient functions for:
- Fitting D-vines, C-vines, R-vines
- Fitting from vine matrices/specifications
- Automatic family selection
- Model persistence (save/load)
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union
from tqdm import tqdm

from .structure import VineStructure, VineTree, build_rvine_structure, get_vine_order
from .vine_types import (
    build_dvine_structure,
    build_cvine_structure,
    build_vine_from_matrix,
    get_vine_matrix
)
from .recursion import VineRecursion, VinePairCopula
from ..models.unet_grid import GridUNet
from ..models.hfunc import HFuncLookup
from ..models.projection import copula_project
from ..inference.density import sample_density_grid, scatter_to_hist


class VineCopulaModel:
    """
    High-level vine copula model supporting D-vine, C-vine, and R-vine.
    
    Usage:
        # Fit automatically
        model = VineCopulaModel(vine_type='dvine')
        model.fit(U, diffusion_model)
        
        # Fit with custom structure
        model = VineCopulaModel(vine_type='dvine', order=[0, 2, 1, 3, 4])
        model.fit(U, diffusion_model)
        
        # Fit from matrix
        model = VineCopulaModel(vine_type='dvine')
        model.fit_from_matrix(U, vine_matrix, diffusion_model)
        
        # Evaluate and sample
        loglik = model.logpdf(U_test)
        samples = model.simulate(n=1000)
        
        # Save/load
        model.save('my_vine.pkl')
        model = VineCopulaModel.load('my_vine.pkl')
    """
    
    def __init__(
        self,
        vine_type: str = 'rvine',
        order: Optional[List[int]] = None,
        truncation_level: Optional[int] = None,
        m: int = 64,
        device: str = 'cuda',
        diffusion_steps: int = 50,
        cfg_scale: float = 1.0,
        projection_iters: int = 50,
        hfunc_use_spline: bool = True,
        batch_edges: bool = False,
        edge_batch_size: int = 256,
    ):
        """
        Initialize vine copula model.
        
        Args:
            vine_type: 'rvine', 'dvine', or 'cvine'
            order: Variable ordering (for D/C-vines) or None for automatic
            truncation_level: Truncate vine at this tree level
            m: Grid resolution for copula densities
            device: Device for neural network inference
            diffusion_steps: DDIM steps when `diffusion` is provided to `fit`
            cfg_scale: Classifier-free guidance scale for histogram-conditioned diffusion models
            projection_iters: IPFP/Sinkhorn iterations when projecting densities to valid copulas
            hfunc_use_spline: If True (default), build SciPy spline interpolators inside each pair-copula
                h-function lookup. If False, use lightweight bilinear interpolation (faster/leaner for
                high-dimensional scaling benchmarks).
            batch_edges: If True, estimate pair-copula densities in mini-batches per tree level (single-pass
                estimators only). This substantially reduces Python overhead and improves GPU utilization.
            edge_batch_size: Max number of pair-copulas to process in one batch (single-pass path).
        """
        self.vine_type = vine_type.lower()
        self.order = order
        self.truncation_level = truncation_level
        self.m = m
        self.device = device
        self.diffusion_steps = int(diffusion_steps)
        self.cfg_scale = float(cfg_scale)
        self.projection_iters = int(projection_iters)
        self.hfunc_use_spline = bool(hfunc_use_spline)
        self.batch_edges = bool(batch_edges)
        self.edge_batch_size = int(edge_batch_size)
        
        self.structure: Optional[VineStructure] = None
        self.vine: Optional[VineRecursion] = None
        self.fitted = False
        
        if self.vine_type not in ['rvine', 'dvine', 'cvine']:
            raise ValueError(f"vine_type must be 'rvine', 'dvine', or 'cvine', got {vine_type}")
    
    def fit(
        self,
        U: np.ndarray,
        diffusion_model: torch.nn.Module,
        diffusion=None,
        verbose: bool = True,
    ):
        """
        Fit vine copula to data.
        
        Args:
            U: (n, d) pseudo-observations in [0,1]
            diffusion_model: Trained pair-copula density estimator (diffusion UNet or single-pass model)
            diffusion: CopulaAwareDiffusion process (optional). If provided, uses iterative DDIM sampling.
            verbose: Show progress
        """
        n, d = U.shape

        # R-vines require structure selection that depends on conditional pseudo-observations.
        # The standalone `build_rvine_structure()` helper is intentionally simplified and does
        # not implement full Dißmann selection. For correctness, we build the R-vine while
        # fitting pair-copulas (Dißmann-style sequential MST).
        if self.vine_type == "rvine":
            self._fit_rvine_dissmann(U, diffusion_model, diffusion, verbose=verbose)
            return
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fitting {self.vine_type.upper()} to {n} samples in {d} dimensions")
            print(f"{'='*60}\n")
        
        # Step 1: Build structure
        if verbose:
            print("Step 1: Building vine structure...")
        
        if self.vine_type == 'dvine':
            self.structure = build_dvine_structure(
                U,
                order=self.order,
                truncation_level=self.truncation_level
            )
        
        elif self.vine_type == 'cvine':
            self.structure = build_cvine_structure(
                U,
                order=self.order,
                truncation_level=self.truncation_level
            )
        
        if verbose:
            print(f"✓ Built {self.vine_type.upper()} structure:")
            print(f"  Trees: {len(self.structure.trees)}")
            print(f"  Total edges: {self.structure.num_edges()}")
            print(f"  Order: {self.structure.order}")
        
        # Step 2: Fit pair copulas
        if verbose:
            print(f"\nStep 2: Fitting pair copulas with diffusion model...")
        
        self.vine = VineRecursion(self.structure, vine_type=self.vine_type)
        
        diffusion_model.eval()
        diffusion_model.to(self.device)
        # Only meaningful for diffusion_unet (2-channel input = [noisy_log_density, log_histogram]).
        use_histogram_conditioning = False
        if diffusion is not None:
            conv_in = getattr(diffusion_model, "conv_in", None)
            if conv_in is not None and hasattr(conv_in, "in_channels"):
                use_histogram_conditioning = int(conv_in.in_channels) > 1
        
        total_edges = self.structure.num_edges()
        pbar = tqdm(total=total_edges, disable=not verbose, desc="Fitting pairs")
        prev_transforms: Optional[Dict[Tuple[int, FrozenSet[int]], np.ndarray]] = None  # (var, cond_set) -> u_{var|cond}
        
        for tree_level in range(len(self.structure.trees)):
            tree = self.structure.trees[tree_level]
            tree_copulas: List[VinePairCopula] = []

            # Optional batching (single-pass only). For diffusion-based fitting, we fall back to per-edge.
            do_batch = bool(self.batch_edges) and (diffusion is None)

            if do_batch:
                # Build all pair_data arrays for this tree level first.
                edges: List[Tuple[int, int, set[int]]] = []
                pair_data_list: List[np.ndarray] = []
                for edge in tree.edges:
                    i, j, cond = edge
                    if tree_level == 0:
                        u_data = U[:, i]
                        v_data = U[:, j]
                    else:
                        if prev_transforms is None:
                            raise RuntimeError(f"Internal error: prev_transforms is None at tree_level={tree_level}.")
                        D = frozenset(cond)
                        try:
                            u_data = prev_transforms[(i, D)]
                            v_data = prev_transforms[(j, D)]
                        except KeyError as e:
                            raise RuntimeError(
                                f"Missing conditional pseudo-observations for edge ({i},{j}|{sorted(D)}) "
                                f"at tree_level={tree_level}. This indicates the structure is inconsistent "
                                f"or transform propagation failed."
                            ) from e

                    pair_data = np.column_stack([u_data, v_data])
                    pair_data = np.clip(pair_data, 1e-6, 1.0 - 1e-6)
                    edges.append(edge)
                    pair_data_list.append(pair_data)

                densities = self._estimate_pair_densities_from_samples_batched(
                    model=diffusion_model,
                    pair_data_list=pair_data_list,
                )

                for edge, density_grid in zip(edges, densities):
                    hfunc = HFuncLookup(density_grid, use_spline=self.hfunc_use_spline)
                    copula = VinePairCopula(edge=edge, density_grid=density_grid, hfunc=hfunc, level=tree_level)
                    tree_copulas.append(copula)
                    self.vine.add_pair_copula(copula)
                    pbar.update(1)
            else:
                for edge in tree.edges:
                    i, j, cond = edge
                    # Extract correctly conditioned pair data
                    if tree_level == 0:
                        u_data = U[:, i]
                        v_data = U[:, j]
                    else:
                        if prev_transforms is None:
                            raise RuntimeError(
                                f"Internal error: prev_transforms is None at tree_level={tree_level}."
                            )
                        D = frozenset(cond)
                        try:
                            u_data = prev_transforms[(i, D)]
                            v_data = prev_transforms[(j, D)]
                        except KeyError as e:
                            raise RuntimeError(
                                f"Missing conditional pseudo-observations for edge ({i},{j}|{sorted(D)}) "
                                f"at tree_level={tree_level}. This indicates the structure is inconsistent "
                                f"or transform propagation failed."
                            ) from e
                    
                    pair_data = np.column_stack([u_data, v_data])
                    pair_data = np.clip(pair_data, 1e-6, 1.0 - 1e-6)
                    
                    density_grid = self._estimate_pair_density_from_samples(
                        model=diffusion_model,
                        diffusion=diffusion,
                        pair_data=pair_data,
                        use_histogram_conditioning=use_histogram_conditioning,
                    )
                    hfunc = HFuncLookup(density_grid, use_spline=self.hfunc_use_spline)
                    
                    copula = VinePairCopula(
                        edge=edge,
                        density_grid=density_grid,
                        hfunc=hfunc,
                        level=tree_level
                    )
                    
                    tree_copulas.append(copula)
                    self.vine.add_pair_copula(copula)
                    pbar.update(1)
            
            # After fitting this tree, compute transforms for next level
            prev_transforms = self.vine.compute_h_transforms(U, tree_level, prev_transforms)
        
        pbar.close()
        self.fitted = True
        
        if verbose:
            print(f"\n✓ Vine fitting complete!")
            print(f"{'='*60}\n")

    @torch.no_grad()
    def _estimate_pair_densities_from_samples_batched(
        self,
        *,
        model: torch.nn.Module,
        pair_data_list: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Batched single-pass pair-copula inference:
          (histogram + optional channels) -> model -> density -> copula projection

        This is used for scaling benchmarks and fast vine fitting.
        """
        if not pair_data_list:
            return []

        device = torch.device(self.device)
        model.eval()
        model.to(device)

        # Determine whether the model expects extra conditioning channels.
        in_ch: Optional[int] = None
        if hasattr(model, "in_conv"):
            in_ch = int(getattr(model, "in_conv").in_channels)
        elif hasattr(model, "input_conv"):
            try:
                in_ch = int(getattr(model, "input_conv")[0].in_channels)
            except Exception:
                in_ch = None
        elif hasattr(model, "conv_in"):
            in_ch = int(getattr(model, "conv_in").in_channels)

        # Prefer explicit metadata set by build_model(); fall back to channel-count heuristics.
        use_log_n = bool(getattr(model, "vdc_use_log_n", False))
        use_coords = bool(getattr(model, "vdc_use_coordinates", False))
        use_probit_coords = bool(getattr(model, "vdc_use_probit_coords", False))
        probit_coord_eps = float(getattr(model, "vdc_probit_coord_eps", 1e-4))

        if in_ch is not None and not hasattr(model, "vdc_use_log_n"):
            if in_ch in (2, 4):
                use_log_n = True
        if in_ch is not None and not hasattr(model, "vdc_use_coordinates"):
            if in_ch in (3, 4):
                use_coords = True

        out: List[np.ndarray] = []
        B_total = len(pair_data_list)
        bs = max(1, int(self.edge_batch_size))

        # Precompute coordinate channels once per batch (depends only on m).
        m = int(self.m)
        u = torch.linspace(0.5 / m, 1.0 - 0.5 / m, m, device=device)
        v = torch.linspace(0.5 / m, 1.0 - 0.5 / m, m, device=device)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        if use_probit_coords and use_coords:
            eps = max(float(probit_coord_eps), 1.0 / (m * m))
            uu = torch.erfinv(2 * uu.clamp(eps, 1 - eps) - 1) * (2.0 ** 0.5)
            vv = torch.erfinv(2 * vv.clamp(eps, 1 - eps) - 1) * (2.0 ** 0.5)
        coords_1 = torch.stack([uu, vv], dim=0).unsqueeze(0)  # (1,2,m,m)

        for start in range(0, B_total, bs):
            chunk = pair_data_list[start : start + bs]

            # Histograms (density integrating to 1).
            hists = [scatter_to_hist(p, m=m, reflect=True) for p in chunk]
            hist_t = torch.from_numpy(np.stack(hists, axis=0)).float().unsqueeze(1).to(device)  # (B,1,m,m)
            x = hist_t

            if use_log_n:
                ln = np.log([max(1, int(p.shape[0])) for p in chunk]).astype(np.float32)  # (B,)
                ln_t = torch.from_numpy(ln).to(device=device, dtype=x.dtype).view(-1, 1, 1, 1).expand(-1, 1, m, m)
                x = torch.cat([x, ln_t], dim=1)

            if use_coords:
                coords = coords_1.expand(x.shape[0], -1, -1, -1)
                x = torch.cat([x, coords], dim=1)

            if in_ch is not None and x.shape[1] != in_ch:
                raise ValueError(
                    f"Single-pass batched inference input channel mismatch: built x has C={x.shape[1]} but model expects C_in={in_ch}. "
                    f"(use_log_n={use_log_n}, use_coords={use_coords}, use_probit_coords={use_probit_coords})"
                )

            # Forward: denoiser is time-conditioned, CNNs are not
            try:
                out_t = model(x, torch.zeros(x.shape[0], device=device))
            except TypeError:
                out_t = model(x)

            if isinstance(out_t, dict):
                if "density" in out_t:
                    d = out_t["density"]
                elif "log_density" in out_t:
                    d = torch.exp(out_t["log_density"].clamp(min=-20, max=20))
                elif "residual" in out_t:
                    d = torch.exp((torch.log(hist_t.clamp_min(1e-12)) + out_t["residual"]).clamp(min=-20, max=20))
                else:
                    raise ValueError(f"Unknown model output keys: {list(out_t.keys())}")
            else:
                d = out_t

            d = torch.nan_to_num(d, nan=1e-12, posinf=1e6, neginf=1e-12).clamp(min=1e-12, max=1e6)
            du = 1.0 / m
            d = d / ((d * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
            d = copula_project(d, iters=int(self.projection_iters))
            d = d / ((d * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))

            out.extend([d[i, 0].detach().cpu().numpy() for i in range(d.shape[0])])

        return out

    def _fit_rvine_dissmann(
        self,
        U: np.ndarray,
        diffusion_model: torch.nn.Module,
        diffusion,
        verbose: bool = True,
    ) -> None:
        """
        Fit an R-vine via a Dißmann-style sequential maximum spanning tree construction.

        This implements the *structure selection loop* and the *pair-copula fitting loop*
        together, because selecting higher trees requires conditional pseudo-observations
        computed from previously fitted pair-copulas.

        Notes
        -----
        - Edges are weighted by |Kendall's tau| on the relevant (conditional) pseudo-observations.
        - Pair-copulas are estimated nonparametrically via the provided diffusion model.
        """
        import networkx as nx
        from ..utils.stats import kendall_tau, kendall_tau_matrix

        n, d = U.shape

        if self.truncation_level is None:
            truncation_level = d - 1
        else:
            truncation_level = int(min(self.truncation_level, d - 1))

        if verbose:
            print(f"\n{'='*60}")
            print(f"Fitting RVINE (Dißmann-style) to {n} samples in {d} dimensions")
            print(f"{'='*60}\n")

        # Initialize empty structure and recursion (we'll append trees as we go).
        self.structure = VineStructure(d=d, trees=[], order=None)
        self.vine = VineRecursion(self.structure, vine_type="rvine")

        diffusion_model.eval()
        diffusion_model.to(self.device)
        use_histogram_conditioning = False
        if diffusion is not None:
            conv_in = getattr(diffusion_model, "conv_in", None)
            if conv_in is not None and hasattr(conv_in, "in_channels"):
                use_histogram_conditioning = int(conv_in.in_channels) > 1

        total_edges = sum((d - k) for k in range(1, truncation_level + 1))
        pbar = tqdm(total=total_edges, disable=not verbose, desc="Fitting pairs (rvine)")

        # ------------------------------------------------------------------
        # Tree 1: MST on |tau(U_i, U_j)|
        # ------------------------------------------------------------------
        tau_mat = kendall_tau_matrix(U)
        G0 = nx.Graph()
        G0.add_nodes_from(range(d))
        for i in range(d):
            for j in range(i + 1, d):
                tau = float(tau_mat[i, j])
                w = abs(tau)
                if not np.isfinite(w):
                    w = 0.0
                    tau = 0.0
                G0.add_edge(i, j, weight=w, tau=tau)

        T0 = nx.maximum_spanning_tree(G0, weight="weight")
        if T0.number_of_edges() != d - 1:
            raise RuntimeError(
                f"Failed to build a spanning tree for Tree 1 (edges={T0.number_of_edges()}, expected={d-1})."
            )

        tree1_edges: List[Tuple[int, int, set[int]]] = []
        tree1_taus: List[float] = []
        for i, j, attrs in T0.edges(data=True):
            tree1_edges.append((int(i), int(j), set()))
            tree1_taus.append(float(attrs.get("tau", 0.0)))

        self.structure.trees.append(VineTree(level=0, edges=tree1_edges, tau_values=tree1_taus))

        # Fit Tree 1 pair copulas
        for edge in tree1_edges:
            i, j, cond = edge
            pair_data = np.column_stack([U[:, i], U[:, j]])
            pair_data = np.clip(pair_data, 1e-6, 1.0 - 1e-6)

            density_grid = self._estimate_pair_density_from_samples(
                model=diffusion_model,
                diffusion=diffusion,
                pair_data=pair_data,
                use_histogram_conditioning=use_histogram_conditioning,
            )
            hfunc = HFuncLookup(density_grid, use_spline=self.hfunc_use_spline)

            self.vine.add_pair_copula(
                VinePairCopula(edge=edge, density_grid=density_grid, hfunc=hfunc, level=0)
            )
            pbar.update(1)

        # Conditional pseudo-observations for Tree 2 selection (conditioning set size 1)
        prev_transforms: Dict[Tuple[int, FrozenSet[int]], np.ndarray] = self.vine.compute_h_transforms(
            U, tree_level=0, prev_transforms=None
        )

        # ------------------------------------------------------------------
        # Higher trees
        # ------------------------------------------------------------------
        for tree_level in range(1, truncation_level):
            prev_tree = self.structure.trees[tree_level - 1]
            prev_edges = prev_tree.edges
            n_nodes = len(prev_edges)

            # Each node is the union set of variables involved in the corresponding previous edge
            node_sets: List[FrozenSet[int]] = [
                frozenset({a, b}) | frozenset(cond) for (a, b, cond) in prev_edges
            ]

            # Candidate graph between previous-tree edges (nodes), using proximity condition
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))

            for p in range(n_nodes):
                for q in range(p + 1, n_nodes):
                    inter = node_sets[p] & node_sets[q]
                    if len(inter) != tree_level:
                        continue

                    left = list(node_sets[p] - inter)
                    right = list(node_sets[q] - inter)
                    if len(left) != 1 or len(right) != 1:
                        continue

                    i_var = int(left[0])
                    j_var = int(right[0])
                    D = frozenset(inter)

                    try:
                        u_i = prev_transforms[(i_var, D)]
                        u_j = prev_transforms[(j_var, D)]
                    except KeyError:
                        # If transforms are missing, this candidate can't be evaluated.
                        continue

                    tau = float(kendall_tau(u_i, u_j))
                    if not np.isfinite(tau):
                        tau = 0.0

                    G.add_edge(
                        p,
                        q,
                        weight=abs(tau),
                        tau=tau,
                        i=i_var,
                        j=j_var,
                        cond=D,
                    )

            if G.number_of_edges() == 0:
                if verbose:
                    print(f"Stopping at tree {tree_level+1}: no admissible edges under proximity condition.")
                break

            T = nx.maximum_spanning_tree(G, weight="weight")
            expected_edges = n_nodes - 1
            if T.number_of_edges() != expected_edges:
                raise RuntimeError(
                    f"Tree {tree_level+1} is disconnected (MST edges={T.number_of_edges()}, expected={expected_edges})."
                )

            edges: List[Tuple[int, int, set[int]]] = []
            tau_values: List[float] = []
            for p, q, attrs in T.edges(data=True):
                i_var = int(attrs["i"])
                j_var = int(attrs["j"])
                D = frozenset(attrs["cond"])
                edges.append((i_var, j_var, set(D)))
                tau_values.append(float(attrs.get("tau", 0.0)))

            self.structure.trees.append(VineTree(level=tree_level, edges=edges, tau_values=tau_values))

            # Fit pair copulas at this tree level, using the correct conditional pseudo-observations
            for edge in edges:
                i, j, cond = edge
                D = frozenset(cond)
                try:
                    u_data = prev_transforms[(i, D)]
                    v_data = prev_transforms[(j, D)]
                except KeyError as e:
                    raise RuntimeError(
                        f"Missing conditional pseudo-observations for selected edge ({i},{j}|{sorted(D)}) "
                        f"at tree_level={tree_level}."
                    ) from e

                pair_data = np.column_stack([u_data, v_data])
                pair_data = np.clip(pair_data, 1e-6, 1.0 - 1e-6)

                density_grid = self._estimate_pair_density_from_samples(
                    model=diffusion_model,
                    diffusion=diffusion,
                    pair_data=pair_data,
                    use_histogram_conditioning=use_histogram_conditioning,
                )
                hfunc = HFuncLookup(density_grid, use_spline=self.hfunc_use_spline)

                self.vine.add_pair_copula(
                    VinePairCopula(edge=edge, density_grid=density_grid, hfunc=hfunc, level=tree_level)
                )
                pbar.update(1)

            # Compute transforms for the next tree selection
            prev_transforms = self.vine.compute_h_transforms(U, tree_level=tree_level, prev_transforms=prev_transforms)

        pbar.close()

        # Store a heuristic variable order for convenience/debugging (not used for R-vine recursion here)
        try:
            self.structure.order = get_vine_order(self.structure)
        except Exception:
            self.structure.order = None

        self.fitted = True

        if verbose:
            print(f"\n✓ R-vine fitting complete!")
            print(f"  Trees: {len(self.structure.trees)}")
            print(f"  Total edges: {self.structure.num_edges()}")
            if self.structure.order is not None:
                print(f"  Heuristic order: {self.structure.order}")
            print(f"{'='*60}\n")

    def _estimate_pair_density_from_samples(
        self,
        model: torch.nn.Module,
        diffusion,
        pair_data: np.ndarray,
        use_histogram_conditioning: bool,
    ) -> np.ndarray:
        """
        Turn (n,2) pseudo-observations into an (m,m) copula density grid.

        - If `diffusion` is provided, uses iterative reverse diffusion (DDIM-style).
        - If `diffusion` is None, uses a single forward pass histogram->density (for denoiser/CNNs).
        """
        if diffusion is not None:
            transform_to_probit_space = bool(getattr(model, "vdc_transform_to_probit_space", False))
            return sample_density_grid(
                model=model,
                diffusion=diffusion,
                samples=pair_data,
                m=self.m,
                device=torch.device(self.device),
                num_steps=int(self.diffusion_steps),
                cfg_scale=float(self.cfg_scale),
                use_histogram_conditioning=use_histogram_conditioning,
                projection_iters=int(self.projection_iters),
                transform_to_probit_space=transform_to_probit_space,
            )

        # Single-pass path (expects a histogram-conditioned estimator)
        hist = scatter_to_hist(pair_data, m=self.m, reflect=True)
        hist_t = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Determine whether the model expects extra conditioning channels.
        in_ch: Optional[int] = None
        if hasattr(model, "in_conv"):
            in_ch = int(getattr(model, "in_conv").in_channels)
        elif hasattr(model, "input_conv"):
            try:
                in_ch = int(getattr(model, "input_conv")[0].in_channels)
            except Exception:
                in_ch = None
        elif hasattr(model, "conv_in"):
            in_ch = int(getattr(model, "conv_in").in_channels)

        x = hist_t

        # Prefer explicit metadata set by build_model(); fall back to channel-count heuristics.
        use_log_n = bool(getattr(model, "vdc_use_log_n", False))
        use_coords = bool(getattr(model, "vdc_use_coordinates", False))
        use_probit_coords = bool(getattr(model, "vdc_use_probit_coords", False))
        probit_coord_eps = float(getattr(model, "vdc_probit_coord_eps", 1e-4))

        if in_ch is not None and not hasattr(model, "vdc_use_log_n"):
            if in_ch in (2, 4):
                use_log_n = True
        if in_ch is not None and not hasattr(model, "vdc_use_coordinates"):
            if in_ch in (3, 4):
                use_coords = True

        if use_log_n:
            ln = float(np.log(max(1, pair_data.shape[0])))
            ln_chan = torch.full((1, 1, self.m, self.m), ln, device=self.device, dtype=x.dtype)
            x = torch.cat([x, ln_chan], dim=1)

        if use_coords:
            m = self.m
            u = torch.linspace(0.5 / m, 1.0 - 0.5 / m, m, device=self.device, dtype=x.dtype)
            v = torch.linspace(0.5 / m, 1.0 - 0.5 / m, m, device=self.device, dtype=x.dtype)
            uu, vv = torch.meshgrid(u, v, indexing="ij")
            if use_probit_coords:
                eps = max(probit_coord_eps, 1.0 / (m * m))
                uu = torch.erfinv(2 * uu.clamp(eps, 1 - eps) - 1) * (2.0 ** 0.5)
                vv = torch.erfinv(2 * vv.clamp(eps, 1 - eps) - 1) * (2.0 ** 0.5)
            coords = torch.stack([uu, vv], dim=0).unsqueeze(0)  # (1,2,m,m)
            x = torch.cat([x, coords], dim=1)

        if in_ch is not None and x.shape[1] != in_ch:
            raise ValueError(
                f"Single-pass inference input channel mismatch: built x has C={x.shape[1]} but model expects C_in={in_ch}. "
                f"(use_log_n={use_log_n}, use_coords={use_coords}, use_probit_coords={use_probit_coords})"
            )

        # Forward: denoiser is time-conditioned, CNNs are not
        try:
            out = model(x, torch.zeros(1, device=self.device))
        except TypeError:
            out = model(x)

        if isinstance(out, dict):
            if "density" in out:
                d = out["density"]
            elif "log_density" in out:
                d = torch.exp(out["log_density"].clamp(min=-20, max=20))
            elif "residual" in out:
                d = torch.exp((torch.log(hist_t.clamp_min(1e-12)) + out["residual"]).clamp(min=-20, max=20))
            else:
                raise ValueError(f"Unknown model output keys: {list(out.keys())}")
        else:
            d = out

        d = d.clamp(min=1e-12, max=1e6)
        du = 1.0 / self.m
        d = d / ((d * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
        d = copula_project(d, iters=int(self.projection_iters))
        d = d / ((d * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
        return d[0, 0].detach().cpu().numpy()
    
    def fit_from_matrix(
        self,
        U: np.ndarray,
        vine_matrix: np.ndarray,
        diffusion_model: torch.nn.Module,
        diffusion,
        verbose: bool = True,
    ):
        """
        Fit vine copula from a predefined vine matrix.
        
        Args:
            U: (n, d) pseudo-observations
            vine_matrix: (d, d) vine matrix specification
            diffusion_model: Trained diffusion model
            diffusion: CopulaAwareDiffusion process (required)
            verbose: Show progress
        """
        # Build structure from matrix
        self.structure = build_vine_from_matrix(U, vine_matrix, self.vine_type)
        
        # Extract order from structure
        self.order = self.structure.order
        
        # Fit using the structure
        self.fit(U, diffusion_model, diffusion, verbose=verbose)
    
    def _estimate_pair_copula(
        self,
        model: torch.nn.Module,
        pair_data: np.ndarray,
    ) -> tuple:
        """
        Estimate bivariate copula using diffusion model.
        
        Args:
            model: Trained diffusion model
            pair_data: (n, 2) pseudo-observations
            
        Returns:
            (density_grid, hfunc) tuple
        """
        # Create histogram
        hist = scatter_to_hist(pair_data, m=self.m, reflect=True)
        hist_t = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict density
        with torch.no_grad():
            # Time tensor should be (B,) shape for the model
            t = torch.ones(1, device=self.device) * 0.5
            # Log of sample size
            log_n = torch.tensor([np.log(len(pair_data))], device=self.device)
            logD_raw = model(hist_t, t, log_n)
            D_pos = torch.exp(logD_raw).clamp(min=1e-12, max=1e6)
            D_copula = copula_project(D_pos, iters=20)
        
        # Create h-function lookup
        density_np = D_copula[0, 0].detach().cpu().numpy()
        hfunc = HFuncLookup(density_np, use_spline=self.hfunc_use_spline)
        
        return density_np, hfunc
    
    def logpdf(self, U: np.ndarray) -> np.ndarray:
        """
        Compute log-density at points.
        
        Args:
            U: (n, d) pseudo-observations
            
        Returns:
            (n,) log-density values
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.vine.logpdf(U)
    
    def pdf(self, U: np.ndarray) -> np.ndarray:
        """Compute density (not log-density)."""
        return np.exp(self.logpdf(U))
    
    def simulate(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate samples from vine copula.
        
        Args:
            n: Number of samples
            seed: Random seed
            
        Returns:
            (n, d) samples
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.vine.simulate(n, seed=seed)
    
    def rosenblatt(self, U: np.ndarray) -> np.ndarray:
        """
        Forward Rosenblatt transform: copula data → independent uniforms.
        
        Args:
            U: (n, d) pseudo-observations
            
        Returns:
            (n, d) independent uniforms
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.vine.rosenblatt(U)
    
    def inverse_rosenblatt(self, W: np.ndarray) -> np.ndarray:
        """
        Inverse Rosenblatt transform: independent uniforms → copula data.
        
        Args:
            W: (n, d) independent uniforms
            
        Returns:
            (n, d) copula samples
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.vine.inverse_rosenblatt(W)
    
    def get_structure_matrix(self) -> np.ndarray:
        """
        Get vine structure as matrix.
        
        Returns:
            (d, d) vine matrix
        """
        if self.structure is None:
            raise RuntimeError("No structure available. Call fit() first.")
        
        return get_vine_matrix(self.structure)
    
    def save(self, filepath: Union[str, Path]):
        """
        Save fitted vine model to file.
        
        Args:
            filepath: Output file path (.pkl)
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted model.")
        
        filepath = Path(filepath)
        
        # Prepare state for saving
        state = {
            'vine_type': self.vine_type,
            'order': self.order,
            'truncation_level': self.truncation_level,
            'm': self.m,
            'device': self.device,
            'structure': self.structure,
            'vine': self.vine,
            'fitted': self.fitted,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"✓ Saved vine model to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'VineCopulaModel':
        """
        Load fitted vine model from file.
        
        Args:
            filepath: Input file path (.pkl)
            
        Returns:
            Loaded VineCopulaModel
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Reconstruct model
        model = cls(
            vine_type=state['vine_type'],
            order=state['order'],
            truncation_level=state['truncation_level'],
            m=state['m'],
            device=state['device'],
        )
        
        model.structure = state['structure']
        model.vine = state['vine']
        model.fitted = state['fitted']
        
        print(f"✓ Loaded vine model from {filepath}")
        return model
    
    def summary(self) -> Dict[str, Any]:
        """
        Get model summary.
        
        Returns:
            Dictionary with model info
        """
        if not self.fitted:
            return {
                'fitted': False,
                'vine_type': self.vine_type,
            }
        
        return {
            'fitted': True,
            'vine_type': self.vine_type,
            'd': self.structure.d,
            'num_trees': len(self.structure.trees),
            'num_edges': self.structure.num_edges(),
            'order': self.structure.order,
            'truncation_level': self.truncation_level,
            'm': self.m,
        }
    
    def __repr__(self) -> str:
        summary = self.summary()
        if not summary['fitted']:
            return f"VineCopulaModel(vine_type='{self.vine_type}', fitted=False)"
        
        return (
            f"VineCopulaModel(\n"
            f"  vine_type='{summary['vine_type']}',\n"
            f"  d={summary['d']},\n"
            f"  num_trees={summary['num_trees']},\n"
            f"  num_edges={summary['num_edges']},\n"
            f"  order={summary['order']},\n"
            f"  fitted=True\n"
            f")"
        )


# Convenience functions for fitting

def fit_rvine(
    U: np.ndarray,
    diffusion_model: torch.nn.Module,
    diffusion,
    truncation_level: Optional[int] = None,
    m: int = 64,
    device: str = 'cuda',
) -> VineCopulaModel:
    """Fit R-vine copula."""
    model = VineCopulaModel(vine_type='rvine', truncation_level=truncation_level, m=m, device=device)
    model.fit(U, diffusion_model, diffusion)
    return model


def fit_dvine(
    U: np.ndarray,
    diffusion_model: torch.nn.Module,
    diffusion,
    order: Optional[List[int]] = None,
    truncation_level: Optional[int] = None,
    m: int = 64,
    device: str = 'cuda',
) -> VineCopulaModel:
    """Fit D-vine copula."""
    model = VineCopulaModel(vine_type='dvine', order=order, truncation_level=truncation_level, m=m, device=device)
    model.fit(U, diffusion_model, diffusion)
    return model


def fit_cvine(
    U: np.ndarray,
    diffusion_model: torch.nn.Module,
    diffusion,
    order: Optional[List[int]] = None,
    truncation_level: Optional[int] = None,
    m: int = 64,
    device: str = 'cuda',
) -> VineCopulaModel:
    """Fit C-vine copula."""
    model = VineCopulaModel(vine_type='cvine', order=order, truncation_level=truncation_level, m=m, device=device)
    model.fit(U, diffusion_model, diffusion)
    return model


if __name__ == "__main__":
    print("Testing high-level vine copula API...")
    
    # Generate test data
    np.random.seed(42)
    n, d = 300, 4
    
    from scipy.stats import norm
    rho = 0.5
    Sigma = np.eye(d)
    for i in range(d-1):
        Sigma[i, i+1] = rho
        Sigma[i+1, i] = rho
    
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    U = norm.cdf(Z)
    
    print(f"Generated {n} samples in {d} dimensions")
    
    # Create dummy model (in practice, load trained model)
    print("\nCreating dummy diffusion model...")
    dummy_model = GridUNet(m=64)
    dummy_model.eval()
    
    # Test R-vine
    print("\n" + "="*60)
    print("Testing R-Vine")
    print("="*60)
    rvine_model = VineCopulaModel(vine_type='rvine', m=64, device='cpu')
    print("Note: Using untrained model, results will be meaningless")
    # rvine_model.fit(U, dummy_model, verbose=True)
    # print(rvine_model)
    
    # Test D-vine with custom order
    print("\n" + "="*60)
    print("Testing D-Vine with custom order")
    print("="*60)
    dvine_model = VineCopulaModel(vine_type='dvine', order=[0, 2, 1, 3], m=64, device='cpu')
    # dvine_model.fit(U, dummy_model, verbose=True)
    
    # Test C-vine
    print("\n" + "="*60)
    print("Testing C-Vine")
    print("="*60)
    cvine_model = VineCopulaModel(vine_type='cvine', m=64, device='cpu')
    # cvine_model.fit(U, dummy_model, verbose=True)
    
    print("\n✓ High-level API test completed!")
    print("\nUsage example:")
    print("""
    # Load trained diffusion model
    model = GridUNet(m=64)
    model.load_state_dict(torch.load('checkpoints/best.pt')['model_state_dict'])
    model.eval()
    
    # Fit D-vine with custom order
    vine = fit_dvine(U, model, order=[0, 2, 1, 3, 4])
    
    # Evaluate and sample
    loglik = vine.logpdf(U_test)
    samples = vine.simulate(n=1000)
    
    # Save for later use
    vine.save('my_dvine.pkl')
    """)
