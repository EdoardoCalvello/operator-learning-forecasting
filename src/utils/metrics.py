"""
Standard metrics for neural operator evaluation on dynamical systems.

This module provides evaluation metrics for neural operator learning
on dynamical systems, including specialized metrics for operator tasks.
"""

import torch
import numpy as np
from typing import Union, Optional, Dict, Any
import torch.nn.functional as F


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Mean Squared Error loss.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        MSE loss
    """
    return F.mse_loss(y_pred, y_true, reduction=reduction)


def relative_l2_loss(y_pred: torch.Tensor, y_true: torch.Tensor, 
                     eps: float = 1e-8, reduction: str = 'mean') -> torch.Tensor:
    """
    Relative L2 loss: ||y_pred - y_true||_2 / ||y_true||_2
    
    Key metric for neural operator evaluation that normalizes
    the error by the magnitude of the true solution.
    
    Args:
        y_pred: Predicted values 
        y_true: True values
        eps: Small value to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Relative L2 loss
    """
    # Compute L2 norms along appropriate dimensions
    if y_pred.dim() == 3:  # (batch, seq_len, features)
        # Average over sequence and features, keep batch dimension
        numerator = torch.sqrt(torch.mean(torch.mean((y_pred - y_true)**2, dim=2), dim=1))
        denominator = torch.sqrt(torch.mean(torch.mean(y_true**2, dim=2), dim=1))
    elif y_pred.dim() == 2:  # (batch, features)
        numerator = torch.sqrt(torch.mean((y_pred - y_true)**2, dim=1))
        denominator = torch.sqrt(torch.mean(y_true**2, dim=1))
    else:
        numerator = torch.sqrt(torch.mean((y_pred - y_true)**2))
        denominator = torch.sqrt(torch.mean(y_true**2))
    
    rel_loss = numerator / (denominator + eps)
    
    if reduction == 'mean':
        return torch.mean(rel_loss)
    elif reduction == 'sum':
        return torch.sum(rel_loss)
    else:
        return rel_loss


def sup_norm_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Supremum (maximum) norm loss: max|y_pred - y_true|
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        Maximum absolute error
    """
    return torch.max(torch.abs(y_pred - y_true))


def mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Mean Absolute Error loss.
    
    Args:
        y_pred: Predicted values
        y_true: True values  
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        MAE loss
    """
    return F.l1_loss(y_pred, y_true, reduction=reduction)


def rmse_loss(y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Root Mean Squared Error loss.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        RMSE loss
    """
    mse = mse_loss(y_pred, y_true, reduction=reduction)
    return torch.sqrt(mse)


def relative_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor, 
                      eps: float = 1e-8, reduction: str = 'mean') -> torch.Tensor:
    """
    Relative Mean Absolute Error: |y_pred - y_true| / |y_true|
    
    Args:
        y_pred: Predicted values
        y_true: True values
        eps: Small value to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Relative MAE loss
    """
    abs_error = torch.abs(y_pred - y_true)
    abs_true = torch.abs(y_true)
    rel_error = abs_error / (abs_true + eps)
    
    if reduction == 'mean':
        return torch.mean(rel_error)
    elif reduction == 'sum':
        return torch.sum(rel_error)
    else:
        return rel_error


def pointwise_relative_error(y_pred: torch.Tensor, y_true: torch.Tensor, 
                            eps: float = 1e-8) -> torch.Tensor:
    """
    Pointwise relative error for detailed error analysis.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        eps: Small value to avoid division by zero
    
    Returns:
        Pointwise relative errors (same shape as input)
    """
    return torch.abs(y_pred - y_true) / (torch.abs(y_true) + eps)


def compute_all_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, 
                       prefix: str = '') -> Dict[str, float]:
    """
    Compute all standard metrics at once.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        prefix: Prefix for metric names (e.g., 'train/', 'val/')
    
    Returns:
        Dictionary of all computed metrics
    """
    metrics = {}
    
    # Convert to float for consistent output
    metrics[f'{prefix}mse'] = mse_loss(y_pred, y_true).item()
    metrics[f'{prefix}mae'] = mae_loss(y_pred, y_true).item()
    metrics[f'{prefix}rmse'] = rmse_loss(y_pred, y_true).item()
    metrics[f'{prefix}rel_l2'] = relative_l2_loss(y_pred, y_true).item()
    metrics[f'{prefix}rel_mae'] = relative_mae_loss(y_pred, y_true).item()
    metrics[f'{prefix}sup_norm'] = sup_norm_loss(y_pred, y_true).item()
    
    return metrics


class MetricsTracker:
    """
    Utility class to track metrics over multiple batches/epochs.
    """
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
               prefix: str = '', batch_size: Optional[int] = None):
        """
        Update metrics with new batch.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            prefix: Prefix for metric names
            batch_size: Batch size for weighted averaging (default: y_pred.shape[0])
        """
        if batch_size is None:
            batch_size = y_pred.shape[0]
        
        batch_metrics = compute_all_metrics(y_pred, y_true, prefix)
        
        for name, value in batch_metrics.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += value * batch_size
            self.counts[name] += batch_size
    
    def compute_averages(self) -> Dict[str, float]:
        """
        Compute average metrics across all updates.
        
        Returns:
            Dictionary of averaged metrics
        """
        return {name: self.metrics[name] / self.counts[name] 
                for name in self.metrics.keys()}
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.counts.clear()


def find_error_samples(errors: np.ndarray, 
                      sample_type: str = 'median') -> int:
    """
    Find samples with specific error characteristics.
    
    Args:
        errors: Array of error values
        sample_type: 'median', 'min', 'max', or percentile value (0-100)
    
    Returns:
        Index of the selected sample
    """
    if sample_type == 'median':
        return int(np.argsort(errors)[len(errors) // 2])
    elif sample_type == 'min':
        return int(np.argmin(errors))
    elif sample_type == 'max':
        return int(np.argmax(errors))
    elif isinstance(sample_type, (int, float)):
        # Percentile
        percentile_idx = int(len(errors) * sample_type / 100)
        return int(np.argsort(errors)[percentile_idx])
    else:
        raise ValueError(f"Unknown sample_type: {sample_type}")


# Available loss functions for neural operator evaluation
LOSS_FUNCTIONS = {
    'mse': mse_loss,
    'l2': mse_loss,
    'mae': mae_loss, 
    'l1': mae_loss,
    'rmse': rmse_loss,
    'rel_l2': relative_l2_loss,
    'rel_mae': relative_mae_loss,
    'sup': sup_norm_loss,
    'sup_norm': sup_norm_loss
}
