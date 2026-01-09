"""
Efficient streaming data loader for very large copula datasets.

Features:
- Memory-efficient streaming from disk
- On-the-fly histogram computation
- Data augmentation (rotation, reflection)
- Caching for frequently accessed samples
"""

import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
from functools import lru_cache
import random

from vdc.data.hist import scatter_to_hist


class StreamingCopulaDataset(IterableDataset):
    """
    Streaming dataset that efficiently loads copula data from disk.
    
    Optimized for:
    - Very large datasets that don't fit in memory
    - Multi-worker data loading
    - Minimal memory overhead
    """
    
    def __init__(
        self,
        data_root: str,
        m: int = 64,
        split: str = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
        reflect: bool = True,
        smooth_sigma: Optional[float] = None,
        cache_size: int = 100,
        augment: bool = True,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
    ):
        """
        Args:
            data_root: Path to directory containing HDF5 files
            m: Grid resolution
            split: One of 'train', 'val', 'test'
            train_frac: Fraction of data for training
            val_frac: Fraction of data for validation
            seed: Random seed for splitting
            reflect: Use reflection padding for histograms
            smooth_sigma: Optional smoothing for histograms
            cache_size: Number of samples to cache in memory (per worker)
            augment: Apply data augmentation (rotation, reflection)
            shuffle: Whether to shuffle data
            shuffle_buffer_size: Size of shuffle buffer
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.m = m
        self.split = split
        self.reflect = reflect
        self.smooth_sigma = smooth_sigma
        self.cache_size = cache_size
        self.augment = augment and (split == "train")
        self.shuffle = shuffle and (split == "train")
        self.shuffle_buffer_size = shuffle_buffer_size
        
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
        self.n_samples = len(self.file_paths)
        
        print(f"StreamingCopulaDataset: {self.n_samples} samples for {split} split")
        
        # For worker assignment
        self.start_idx = 0
        self.end_idx = self.n_samples
    
    def _get_worker_info(self):
        """Get worker-specific slice of data."""
        worker_info = get_worker_info()
        
        if worker_info is None:
            # Single-process loading
            return 0, self.n_samples, 0
        
        # Multi-process loading: split data among workers
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        
        per_worker = int(np.ceil(self.n_samples / num_workers))
        start = worker_id * per_worker
        end = min(start + per_worker, self.n_samples)
        
        return start, end, worker_id
    
    @lru_cache(maxsize=100)
    def _load_sample_cached(self, idx: int) -> Dict[str, Any]:
        """Load a sample with LRU caching."""
        return self._load_sample(idx)
    
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """Load a single sample from disk."""
        file_path = self.file_paths[idx]
        
        with h5py.File(file_path, 'r') as f:
            points = f['points'][:]
            log_pdf_grid = f['log_pdf_grid'][:]
            
            # Load metadata
            meta = {
                'family': f.attrs.get('family', 'unknown'),
                'rotation': f.attrs.get('rotation', 0),
                'tau': f.attrs.get('tau', 0.0),
            }
        
        return {
            'points': points,
            'log_pdf_grid': log_pdf_grid,
            'meta': meta,
        }
    
    def _augment_sample(self, points: np.ndarray, log_pdf_grid: np.ndarray) -> tuple:
        """Apply data augmentation."""
        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            log_pdf_grid = np.rot90(log_pdf_grid, k=k)
            
            # Rotate points
            for _ in range(k):
                u, v = points[:, 0], points[:, 1]
                points = np.stack([1 - v, u], axis=1)
        
        # Random flip
        if random.random() > 0.5:
            log_pdf_grid = np.fliplr(log_pdf_grid)
            points[:, 0] = 1 - points[:, 0]
        
        if random.random() > 0.5:
            log_pdf_grid = np.flipud(log_pdf_grid)
            points[:, 1] = 1 - points[:, 1]
        
        return points, log_pdf_grid
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over samples."""
        start, end, worker_id = self._get_worker_info()
        
        # Create index sequence
        indices = list(range(start, end))
        
        if self.shuffle:
            # Shuffle indices
            random.shuffle(indices)
        
        # Shuffle buffer for better randomization
        buffer = []
        
        for idx in indices:
            # Load sample
            if self.cache_size > 0:
                sample = self._load_sample_cached(idx)
            else:
                sample = self._load_sample(idx)
            
            points = sample['points'].copy()
            log_pdf_grid = sample['log_pdf_grid'].copy()
            meta = sample['meta']
            
            # Augmentation
            if self.augment:
                points, log_pdf_grid = self._augment_sample(points, log_pdf_grid)
            
            # Compute histogram
            hist = scatter_to_hist(points, self.m, self.reflect, self.smooth_sigma)
            
            # Convert to tensors
            hist_tensor = torch.from_numpy(hist).unsqueeze(0).float()  # (1, m, m)
            points_tensor = torch.from_numpy(points).float()  # (n, 2)
            log_pdf_tensor = torch.from_numpy(log_pdf_grid).float()  # (m, m)
            
            sample_dict = {
                'hist': hist_tensor,
                'points': points_tensor,
                'log_pdf_grid': log_pdf_tensor,
                'meta': meta,
            }
            
            if self.shuffle:
                # Add to shuffle buffer
                buffer.append(sample_dict)
                
                # Yield from buffer when full
                if len(buffer) >= self.shuffle_buffer_size:
                    random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []
            else:
                yield sample_dict
        
        # Yield remaining items in buffer
        if buffer:
            if self.shuffle:
                random.shuffle(buffer)
            for item in buffer:
                yield item


class InfiniteStreamingDataset(IterableDataset):
    """
    Infinite streaming dataset that cycles through data indefinitely.
    
    Useful for training where you want to control stopping by steps rather than epochs.
    """
    
    def __init__(self, base_dataset: StreamingCopulaDataset):
        super().__init__()
        self.base_dataset = base_dataset
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Infinitely iterate over samples."""
        while True:
            for sample in self.base_dataset:
                yield sample


def collate_streaming_batch(batch):
    """
    Collate function for streaming batches.
    
    Handles variable-length points by padding.
    """
    # Stack histograms and log_pdf_grids
    hists = torch.stack([item['hist'] for item in batch])  # (B, 1, m, m)
    log_pdfs = torch.stack([item['log_pdf_grid'] for item in batch])  # (B, m, m)
    
    # Pad points
    max_n = max(item['points'].shape[0] for item in batch)
    points_list = []
    masks = []
    
    for item in batch:
        pts = item['points']  # (n, 2)
        n = pts.shape[0]
        
        if n < max_n:
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
    
    # Metadata
    meta = [item['meta'] for item in batch]
    
    return {
        'hist': hists,
        'points': points,
        'points_mask': masks,
        'log_pdf_grid': log_pdfs,
        'meta': meta,
    }


if __name__ == "__main__":
    # Test streaming dataset
    dataset = StreamingCopulaDataset(
        data_root="data/zoo",
        m=64,
        split="train",
        cache_size=10,
        augment=True,
        shuffle=True,
    )
    
    print("Testing streaming dataset...")
    for i, sample in enumerate(dataset):
        print(f"Sample {i}:")
        print(f"  Histogram shape: {sample['hist'].shape}")
        print(f"  Points shape: {sample['points'].shape}")
        print(f"  Log-PDF shape: {sample['log_pdf_grid'].shape}")
        
        if i >= 5:
            break
    
    print("\nTesting with DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        collate_fn=collate_streaming_batch,
    )
    
    for batch in loader:
        print(f"Batch shapes:")
        print(f"  hist: {batch['hist'].shape}")
        print(f"  points: {batch['points'].shape}")
        print(f"  mask: {batch['points_mask'].shape}")
        break
