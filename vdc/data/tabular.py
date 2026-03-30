"""Utilities for real/tabular datasets used in paper experiments (E2–E5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_npz_arrays(path: Path, *, required: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"Missing keys {missing} in {path}. Found: {list(f.keys())}")
        out = {k: np.asarray(f[k]) for k in required}
    return out


@dataclass
class EmpiricalMarginals:
    """Empirical per-feature CDF and inverse CDF for copula transforms."""

    sorted_train: np.ndarray  # (n_train, d)

    @classmethod
    def fit(cls, X_train: np.ndarray) -> "EmpiricalMarginals":
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected X_train shape (n,d), got {X.shape}")
        s = np.sort(X, axis=0)
        return cls(sorted_train=s)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map X -> U in (0,1) using the train empirical CDF."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected X shape (n,d), got {X.shape}")
        s = self.sorted_train
        if X.shape[1] != s.shape[1]:
            raise ValueError(f"Dim mismatch: X has d={X.shape[1]}, marginals have d={s.shape[1]}")
        n_train = s.shape[0]
        # Vectorized searchsorted per column
        U = np.empty_like(X, dtype=np.float64)
        for j in range(X.shape[1]):
            # rank in [0..n_train]
            r = np.searchsorted(s[:, j], X[:, j], side="right")
            U[:, j] = (r + 0.5) / (n_train + 1.0)
        return np.clip(U, 1e-6, 1.0 - 1e-6)

    def inverse_transform(self, U: np.ndarray) -> np.ndarray:
        """Map U in (0,1) -> X using the train empirical quantile function."""
        U = np.asarray(U, dtype=np.float64)
        if U.ndim != 2:
            raise ValueError(f"Expected U shape (n,d), got {U.shape}")
        s = self.sorted_train
        if U.shape[1] != s.shape[1]:
            raise ValueError(f"Dim mismatch: U has d={U.shape[1]}, marginals have d={s.shape[1]}")
        n_train = s.shape[0]
        X = np.empty_like(U, dtype=np.float64)
        # Simple nearest-rank inverse (robust + fast).
        idx = np.clip((U * (n_train - 1)).astype(int), 0, n_train - 1)
        for j in range(U.shape[1]):
            X[:, j] = s[idx[:, j], j]
        return X


def train_test_split(
    X: np.ndarray,
    *,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    n = X.shape[0]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_test = max(1, int(round(float(test_frac) * n)))
    te = perm[:n_test]
    tr = perm[n_test:]
    return X[tr], X[te]


def maybe_load_uci(name: str, data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a UCI dataset from the OUTPUT_BASE datasets folder.

    Expected locations (first match wins):
      - {data_root}/uci/{name}.npz with keys: train, test
      - {data_root}/uci/{name}/train.npy and test.npy
    """
    name_l = str(name).lower()
    base = Path(data_root) / "uci"
    p_npz = base / f"{name_l}.npz"
    if p_npz.exists():
        arrs = load_npz_arrays(p_npz, required=("train", "test"))
        return arrs["train"], arrs["test"]

    p_dir = base / name_l
    p_tr = p_dir / "train.npy"
    p_te = p_dir / "test.npy"
    if p_tr.exists() and p_te.exists():
        return np.load(p_tr), np.load(p_te)

    raise FileNotFoundError(
        f"Could not find UCI dataset '{name}'. Looked for:\n"
        f"  - {p_npz} (npz with keys train/test)\n"
        f"  - {p_tr} and {p_te}\n"
        "User request: stage datasets under OUTPUT_BASE/datasets (or set DATA_ROOT)."
    )


def maybe_load_finance_sp100_returns(data_root: Path) -> np.ndarray:
    """Load S&P 100 returns array for VaR/ES experiments.

    Expected file:
      - {data_root}/finance/sp100_returns.npy   shape (T, 100)

    (We keep it simple + cluster-friendly: no internet download.)
    """
    p = Path(data_root) / "finance" / "sp100_returns.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing finance dataset: {p}\n"
            "Expected a NumPy array of shape (T,100) with daily returns."
        )
    x = np.load(p)
    if x.ndim != 2 or x.shape[1] != 100:
        raise ValueError(f"Expected sp100_returns.npy with shape (T,100), got {x.shape}")
    return np.asarray(x, dtype=np.float64)


def iter_pyod_npz_datasets(data_root: Path):
    """Yield (name, X_train, X_test, y_test) from {data_root}/pyod/*.npz.

    Expected keys per file:
      - X_train, X_test, y_test
    """
    base = Path(data_root) / "pyod"
    if not base.exists():
        raise FileNotFoundError(
            f"Missing PyOD dataset directory: {base}\n"
            "Expected .npz files with keys X_train/X_test/y_test."
        )
    for p in sorted(base.glob("*.npz")):
        arrs = load_npz_arrays(p, required=("X_train", "X_test", "y_test"))
        yield p.stem, arrs["X_train"], arrs["X_test"], arrs["y_test"]

