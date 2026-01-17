"""
Utility Functions
General-purpose utilities for the project.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], filepath: str):
    """Save data to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if not."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary as string."""
    return " | ".join([
        f"{k}: {v:.{precision}f}" for k, v in metrics.items()
    ])


__all__ = [
    'load_yaml',
    'save_yaml',
    'load_json',
    'save_json',
    'ensure_dir',
    'format_metrics'
]
