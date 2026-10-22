"""
Utility functions for neural operator experiments on dynamical systems.

This package provides metrics, plotting functions, and data
transformation utilities for neural operator learning tasks.

Main modules:
- metrics: Evaluation metrics (MSE, relative L2, etc.)
- plotting: Visualization functions for dynamical systems
- baselines: Classical forecasting baselines
"""

# Import key functions
from .metrics import (
    mse_loss, mae_loss, rmse_loss,
    relative_l2_loss, relative_mae_loss, sup_norm_loss,
    compute_all_metrics, MetricsTracker, find_error_samples,
    LOSS_FUNCTIONS
)

from .plotting import (
    plot_1d_trajectories, plot_2d_fields, plot_error_samples,
    plot_training_history, plot_comparison, PLOT_CONFIG
)

__all__ = [
    # Metrics
    'mse_loss', 'mae_loss', 'rmse_loss',
    'relative_l2_loss', 'relative_mae_loss', 'sup_norm_loss',
    'compute_all_metrics', 'MetricsTracker', 'find_error_samples',
    'LOSS_FUNCTIONS',
    
    # Plotting  
    'plot_1d_trajectories', 'plot_2d_fields', 'plot_error_samples',
    'plot_training_history', 'plot_comparison', 'PLOT_CONFIG'
]
