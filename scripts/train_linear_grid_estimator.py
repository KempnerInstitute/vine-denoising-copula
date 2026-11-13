#!/usr/bin/env python3
# ruff: noqa: E402
"""Pretrain, build fine-tuning data for, or fine-tune a linear-grid VDC model.

The YAML file selects the stage (``pretrain``, ``cache`` or ``finetune``);
``key=value`` arguments replace its entries. The recipes of the released
models are in ``configs/train/linear_grid_*.yaml``:

    python scripts/train_linear_grid_estimator.py configs/train/linear_grid_pilot.yaml
    python scripts/train_linear_grid_estimator.py configs/train/linear_grid_cache.yaml
    python scripts/train_linear_grid_estimator.py configs/train/linear_grid_vdc64_v2.yaml
    python scripts/train_linear_grid_estimator.py configs/train/linear_grid_vdc128_v2.yaml

See docs/IMPROVED_ESTIMATORS.md for hardware, run times and reproducibility.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdc.train.linear_grid_training import main

if __name__ == "__main__":
    main()
