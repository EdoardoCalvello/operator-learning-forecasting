"""
TNO (Transformer Neural Operator) models.

This module contains the core TNO implementations for neural operator learning
on dynamical systems.
"""

from .TNO_lightning import SimpleEncoderModule
from .TNO_pytorch import SimpleEncoder

__all__ = [
    'SimpleEncoderModule',  # PyTorch Lightning module for training
    'SimpleEncoder',        # Core PyTorch model
]
