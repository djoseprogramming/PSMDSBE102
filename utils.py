"""
Utility Functions Module

Shared helper functions used across different scripts.
Includes logging, file I/O, and common operations.
"""

import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import joblib


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def save_pickle(obj: Any, filepath: Path) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Saved to {filepath}")


def load_pickle(filepath: Path) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return joblib.load(filepath)


def ensure_dir(dirpath: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        dirpath: Directory path
        
    Returns:
        Directory path
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def print_section(title: str, width: int = 80) -> None:
    """
    Print formatted section header.
    
    Args:
        title: Section title
        width: Line width
    """
    print("\n" + "="*width)
    print(title)
    print("="*width + "\n")


def format_metrics(metrics: Dict[str, float], decimal_places: int = 4) -> str:
    """
    Format metrics dictionary as string.
    
    Args:
        metrics: Dictionary of metric names and values
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.{decimal_places}f}")
        else:
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


def get_top_n_items(items: List[Any], scores: List[float], n: int = 5) -> List[tuple]:
    """
    Get top N items by score.
    
    Args:
        items: List of items
        scores: List of scores
        n: Number of top items to return
        
    Returns:
        List of (item, score) tuples
    """
    sorted_pairs = sorted(zip(items, scores), key=lambda x: x, reverse=True)
    return sorted_pairs[:n]


# Logging utilities
def log_info(message: str) -> None:
    """Print info message."""
    print(f"{message}")


def log_warning(message: str) -> None:
    """Print warning message."""
    print(f"{message}")


def log_error(message: str) -> None:
    """Print error message."""
    print(f"{message}")


if __name__ == "__main__":
    print("Utility functions module")
    print("Import this module to use shared utilities")