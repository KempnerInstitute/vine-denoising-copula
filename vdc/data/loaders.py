"""
PyTorch datasets and dataloaders for copula training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, Any, List
from vdc.data.hist import scatter_to_hist


class CopulaPairs(Dataset):
    """
    Dataset of bivariate copula pairs for training.
    
    Each sample contains:
    - histogram: (1, m, m) normalized histogram
    - points: (n, 2) scatter of pseudo-observations
    - log_pdf_grid: (m, m) target log-density (teacher)
    - meta: dictionary with family, parameters, tau, etc.
    """
    
    def __init__(
        self,
        data_root: str,
        m: int = 64,
        split: str = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
        precompute_histograms: bool = True,
        reflect: bool = True,
        smooth_sigma: Optional[float] = None,
    ):
        """
        Args:
            data_root: Path to directory containing HDF5 files
            m: Grid resolution (must match the data)
            split: One of 'train', 'val', 'test'
            train_frac: Fraction of data for training
            val_frac: Fraction of data for validation
            seed: Random seed for splitting
            precompute_histograms: If True, precompute histograms; else compute on-the-fly
            reflect: Use reflection padding for histograms
            smooth_sigma: Optional smoothing for histograms
        """
        self.data_root = Path(data_root)
        self.m = m
        self.split = split
        self.reflect = reflect
        self.smooth_sigma = smooth_sigma
        self.precompute_histograms = precompute_histograms
        
        # Find all HDF5 files
        self.file_paths = sorted(list(self.data_root.glob("*.h5")))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No HDF5 files found in {data_root}")
        
        # Split into train/val/test
        n_total = len(self.file_paths)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        
        # Shuffle with seed
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_total)
        
        if split == "train":
            indices = indices[:n_train]
        elif split == "val":
            indices = indices[n_train:n_train + n_val]
        elif split == "test":
            indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.file_paths = [self.file_paths[i] for i in indices]
        
        print(f"Loaded {len(self.file_paths)} samples for {split} split")
        
        # Optionally precompute histograms
        self.histograms = {}
        if precompute_histograms:
            print("Precomputing histograms...")
            for idx in range(len(self.file_paths)):
                with h5py.File(self.file_paths[idx], 'r') as f:
                    points = f['points'][:]
                hist = scatter_to_hist(points, self.m, self.reflect, self.smooth_sigma)
                self.histograms[idx] = hist
                
                if (idx + 1) % 1000 == 0:
                    print(f"  {idx + 1}/{len(self.file_paths)}")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary with:
            - hist: (1, m, m) tensor
            - points: (n, 2) tensor
            - log_pdf_grid: (m, m) tensor
            - meta: dict with family, params, tau, etc.
        """
        file_path = self.file_paths[idx]
        
        with h5py.File(file_path, 'r') as f:
            points = f['points'][:]
            log_pdf_grid = f['log_pdf_grid'][:]
            
            # Load metadata
            meta = {
                'family': f.attrs.get('family', 'unknown'),
                'rotation': f.attrs.get('rotation', 0),
                'tau': f.attrs.get('tau', 0.0),
                'n': f.attrs.get('n', points.shape[0]),
                'm': f.attrs.get('m', self.m),
            }
            
            # Load parameters
            for key in f.attrs.keys():
                if key.startswith('param_'):
                    param_name = key.replace('param_', '')
                    meta[param_name] = f.attrs[key]
        
        # Get or compute histogram
        if self.precompute_histograms:
            hist = self.histograms[idx]
        else:
            hist = scatter_to_hist(points, self.m, self.reflect, self.smooth_sigma)
        
        # Convert to tensors
        hist_tensor = torch.from_numpy(hist).unsqueeze(0).float()  # (1, m, m)
        points_tensor = torch.from_numpy(points).float()  # (n, 2)
        log_pdf_tensor = torch.from_numpy(log_pdf_grid).float()  # (m, m)
        
        return {
            'hist': hist_tensor,
            'points': points_tensor,
            'log_pdf_grid': log_pdf_tensor,
            'teacher_logpdf': log_pdf_tensor,
            'meta': meta,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable-length points.
    
    For points, we pad to the maximum length in the batch.
    """
    # Stack histograms and log_pdf_grids normally
    hists = torch.stack([item['hist'] for item in batch])  # (B, 1, m, m)
    log_pdfs = torch.stack([item['log_pdf_grid'] for item in batch])  # (B, m, m)
    
    # For points, we need to pad
    max_n = max(item['points'].shape[0] for item in batch)
    points_list = []
    masks = []
    
    for item in batch:
        pts = item['points']  # (n, 2)
        n = pts.shape[0]
        
        if n < max_n:
            # Pad with zeros
            padding = torch.zeros(max_n - n, 2)
            pts_padded = torch.cat([pts, padding], dim=0)
            mask = torch.cat([torch.ones(n), torch.zeros(max_n - n)])
        else:
            pts_padded = pts
            mask = torch.ones(n)
        
        points_list.append(pts_padded)
        masks.append(mask)
    
    points = torch.stack(points_list)  # (B, max_n, 2)
    masks = torch.stack(masks)  # (B, max_n)
    
    # Collect metadata
    meta = [item['meta'] for item in batch]
    
    return {
        'hist': hists,
        'points': points,
        'points_mask': masks,
        'log_pdf_grid': log_pdfs,
        'teacher_logpdf': log_pdfs,
        'meta': meta,
    }


def get_dataloader(
    data_root: str,
    m: int = 64,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a dataloader for copula training.
    
    Args:
        data_root: Path to data directory
        m: Grid resolution
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        distributed: Whether to use DistributedSampler
        **dataset_kwargs: Additional arguments for CopulaPairs
        
    Returns:
        DataLoader instance
    """
    dataset = CopulaPairs(
        data_root=data_root,
        m=m,
        split=split,
        **dataset_kwargs,
    )
    
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=(split == "train"))
        shuffle = False
    else:
        sampler = None
        shuffle = (split == "train")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )
    
    return loader


# Backward-compatible alias used by older training entry points.
CopulaPairsDataset = CopulaPairs


if __name__ == "__main__":
    # Test dataset
    dataset = CopulaPairs(
        data_root="data/zoo",
        m=64,
        split="train",
        precompute_histograms=False,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Histogram shape: {sample['hist'].shape}")
    print(f"Points shape: {sample['points'].shape}")
    print(f"Log-PDF grid shape: {sample['log_pdf_grid'].shape}")
    print(f"Meta: {sample['meta']}")
    
    # Test dataloader
    loader = get_dataloader(
        data_root="data/zoo",
        m=64,
        split="train",
        batch_size=4,
        num_workers=0,
    )
    
    batch = next(iter(loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch histogram shape: {batch['hist'].shape}")
    print(f"Batch points shape: {batch['points'].shape}")
    print(f"Batch mask shape: {batch['points_mask'].shape}")
