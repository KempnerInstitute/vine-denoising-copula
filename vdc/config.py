"""
Configuration management for Vine Denoising Copula.

Provides utilities for:
- Loading YAML configurations
- Merging configs with command-line overrides
- Validating configuration values
- Managing experiment directories
"""

import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import json


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override config into base config.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def parse_overrides(overrides: list) -> Dict[str, Any]:
    """
    Parse command-line overrides in the form key=value or key.subkey=value.
    
    Args:
        overrides: List of strings like ["training.lr=0.001", "model.channels=128"]
        
    Returns:
        Nested dictionary of overrides
    """
    result = {}
    for override in overrides:
        if '=' not in override:
            continue
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Try to parse value as JSON (handles numbers, bools, lists)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # Keep as string
        
        # Build nested dict
        current = result
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    return result


class Config:
    """
    Configuration container with attribute access.
    
    Example:
        config = Config.load("configs/train/default.yaml")
        print(config.model.base_channels)
        print(config.training.learning_rate)
    """
    
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
        self._data = data
    
    @classmethod
    def load(
        cls,
        config_path: Union[str, Path],
        overrides: Optional[list] = None,
        base_config: Optional[Union[str, Path]] = None,
    ) -> 'Config':
        """
        Load configuration from YAML file with optional overrides.
        
        Args:
            config_path: Path to main config file
            overrides: List of command-line overrides
            base_config: Optional base config to merge with
            
        Returns:
            Config object
        """
        config = load_yaml(config_path)
        
        if base_config is not None:
            base = load_yaml(base_config)
            config = merge_configs(base, config)
        
        if overrides:
            override_dict = parse_overrides(overrides)
            config = merge_configs(config, override_dict)
        
        return cls(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        return self._data
    
    def save(self, path: Union[str, Path]):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self._data, f, default_flow_style=False)
    
    def __repr__(self):
        return f"Config({self._data})"


def get_run_dir(
    base_dir: str = "results",
    experiment_name: str = "experiment",
    include_timestamp: bool = True,
    include_job_id: bool = True,
) -> Path:
    """
    Create a unique run directory for experiment outputs.
    
    Format: {base_dir}/{experiment_name}_{timestamp}_{job_id}/
    
    Args:
        base_dir: Base directory for results
        experiment_name: Name of the experiment
        include_timestamp: Whether to include timestamp
        include_job_id: Whether to include SLURM job ID
        
    Returns:
        Path to the run directory (created if doesn't exist)
    """
    parts = [experiment_name]
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)
    
    if include_job_id:
        job_id = os.environ.get("SLURM_JOB_ID", None)
        if job_id:
            parts.append(f"job{job_id}")
    
    run_name = "_".join(parts)
    run_dir = Path(base_dir) / run_name
    
    # Create subdirectories
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    
    return run_dir


def save_run_config(config: Config, run_dir: Path):
    """Save configuration to run directory for reproducibility."""
    config.save(run_dir / "config.yaml")
    
    # Also save as JSON for easy parsing
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


# Default config paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "default.yaml"
DEFAULT_INFERENCE_CONFIG = REPO_ROOT / "configs" / "inference" / "default.yaml"
