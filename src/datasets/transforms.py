"""
Data transformation utilities for neural operators.
"""

import torch
import numpy as np


class UnitGaussianNormalizer:
    """Normalizes data to have zero mean and unit variance using PyTorch tensors."""
    
    def __init__(self, x: torch.Tensor, eps: float = 0.00001):
        """
        Initialize normalizer.
        
        Args:
            x: Data to compute statistics from, shape (N, D)
            eps: Small value to avoid division by zero
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")
            
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True)
        # Avoid division by zero
        self.std = torch.where(self.std < eps, torch.ones_like(self.std), self.std)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input data."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")
            
        return (x - self.mean.to(x.device).to(x.dtype)) / self.std.to(x.device).to(x.dtype)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize the data back to original scale."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")
            
        return x * self.std.to(x.device).to(x.dtype) + self.mean.to(x.device).to(x.dtype)


def create_sequences(data: torch.Tensor, sequence_length: int, 
                    prediction_horizon: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create input-output sequences for time series prediction.
    
    Args:
        data: Time series data, shape (time_steps, features)
        sequence_length: Length of input sequences
        prediction_horizon: How many steps ahead to predict
        
    Returns:
        inputs: Shape (n_sequences, sequence_length, features)
        targets: Shape (n_sequences, prediction_horizon, features)
    """
    n_sequences = len(data) - sequence_length - prediction_horizon + 1
    
    inputs = torch.zeros(n_sequences, sequence_length, data.shape[-1])
    targets = torch.zeros(n_sequences, prediction_horizon, data.shape[-1])
    
    for i in range(n_sequences):
        inputs[i] = data[i:i + sequence_length]
        targets[i] = data[i + sequence_length:i + sequence_length + prediction_horizon]
    
    return inputs, targets


def add_noise(data: torch.Tensor, noise_level: float = 0.01, 
              noise_type: str = 'gaussian') -> torch.Tensor:
    """
    Add noise to data for data augmentation.
    
    Args:
        data: Input data
        noise_level: Standard deviation of noise
        noise_type: Type of noise ('gaussian' or 'uniform')
        
    Returns:
        Noisy data
    """
    if noise_type == 'gaussian':
        noise = torch.randn_like(data) * noise_level
    elif noise_type == 'uniform':
        noise = (torch.rand_like(data) - 0.5) * 2 * noise_level
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return data + noise


def subsample_trajectory(data: torch.Tensor, times: torch.Tensor, 
                        stride: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample trajectory data.
    
    Args:
        data: Trajectory data, shape (time_steps, ...)
        times: Time points, shape (time_steps,)
        stride: Subsampling stride
        
    Returns:
        Subsampled data and times
    """
    return data[::stride], times[::stride]


def split_trajectories(data: torch.Tensor, train_ratio: float = 0.8, 
                      val_ratio: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split trajectory data into train/val/test sets.
    
    Args:
        data: Trajectory data, shape (n_trajectories, time_steps, features)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        
    Returns:
        train_data, val_data, test_data
    """
    n_traj = data.shape[0]
    n_train = int(n_traj * train_ratio)
    n_val = int(n_traj * val_ratio)
    
    # Shuffle indices
    indices = torch.randperm(n_traj)
    
    train_data = data[indices[:n_train]]
    val_data = data[indices[n_train:n_train + n_val]]
    test_data = data[indices[n_train + n_val:]]
    
    return train_data, val_data, test_data