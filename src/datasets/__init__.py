# src/datasets/__init__.py
"""
Dataset modules for neural operators on dynamical systems.
"""

from .base import BaseDataset, DynSys
from .dynamical import (
    Lorenz63, Lorenz96, KuramotoSivashinsky,
    Lorenz63Dataset, Lorenz96Dataset, KSDataset,
    load_dataset
)
from .transforms import (
    UnitGaussianNormalizer, create_sequences, 
    add_noise, subsample_trajectory, split_trajectories
)

# Main dataset imports
__all__ = [
    'BaseDataset', 'DynSys',
    'Lorenz63', 'Lorenz96', 'KuramotoSivashinsky', 
    'Lorenz63Dataset', 'Lorenz96Dataset', 'KSDataset',
    'load_dataset',
    'UnitGaussianNormalizer', 'create_sequences',
    'add_noise', 'subsample_trajectory', 'split_trajectories'
]