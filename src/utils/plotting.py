"""
Plotting utilities for neural operator experiments on dynamical systems.

This module provides plotting functions for neural operator experiments
with support for 1D time series and 2D spatial data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional, Tuple, Union, List, Dict, Any
import matplotlib.colors as mcolors
from scipy.interpolate import griddata


# Standard plotting configuration for neural operator experiments
PLOT_CONFIG = {
    'figure_size_1d': (14, 7),
    'figure_size_2d': (16, 12),
    'title_fontsize': 25,
    'label_fontsize': 20,
    'legend_fontsize': 20,
    'tick_fontsize': 16,
    'linewidth': 3,
    'colormap_data': 'viridis',
    'colormap_error': 'inferno',
    'color_truth': 'blue',
    'color_prediction': 'red',
    'color_input': 'blue',
    'hspace': 0.2,
    'wspace': 0.1
}


def plot_1d_trajectories(
    x_input: np.ndarray,
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    error_value: float,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot 1D trajectory comparison for neural operator evaluation.
    
    Args:
        x_input: Input trajectory data (seq_len, input_dim)
        y_true: True output trajectories (seq_len, output_dim)
        y_pred: Predicted output trajectories (seq_len, output_dim)
        coords_x: Time coordinates for input (seq_len,)
        coords_y: Time coordinates for output (seq_len,)
        error_value: Relative L2 error value for title
        title_suffix: Additional text for title
        save_path: Path to save figure (optional)
        show: Whether to display the figure
    
    Returns:
        Matplotlib figure object
    """
    output_dim = y_true.shape[-1]
    input_dim = x_input.shape[-1]
    
    # Create figure with appropriate size
    fig, axs = plt.subplots(
        nrows=output_dim + 1,  # +1 for input plot
        ncols=1,
        figsize=(PLOT_CONFIG['figure_size_1d'][0], 
                PLOT_CONFIG['figure_size_1d'][1] * (output_dim + 1)),
        sharex=True,
        squeeze=False
    )
    
    # Flatten axes for easier indexing
    axs = axs.flatten()
    
    # Plot input trajectories
    for i in range(input_dim):
        axs[0].plot(
            coords_x.flatten(),
            x_input[:, i],
            linewidth=PLOT_CONFIG['linewidth'],
            color=PLOT_CONFIG['color_input'],
            label=f"Input {i+1}" if input_dim > 1 else "Input"
        )
    
    axs[0].set_ylabel("Input", fontsize=PLOT_CONFIG['label_fontsize'])
    axs[0].legend(fontsize=PLOT_CONFIG['legend_fontsize'])
    axs[0].tick_params(labelsize=PLOT_CONFIG['tick_fontsize'])
    
    # Plot output trajectories
    for i in range(output_dim):
        # Ground truth
        axs[i+1].plot(
            coords_y.flatten(),
            y_true[:, i],
            linewidth=PLOT_CONFIG['linewidth'],
            color=PLOT_CONFIG['color_truth'],
            label="Ground Truth"
        )
        
        # Prediction  
        axs[i+1].plot(
            coords_y.flatten(),
            y_pred[:, i],
            linewidth=PLOT_CONFIG['linewidth'],
            color=PLOT_CONFIG['color_prediction'],
            label="Prediction"
        )
        
        axs[i+1].set_ylabel(f"Output {i+1}", fontsize=PLOT_CONFIG['label_fontsize'])
        axs[i+1].legend(fontsize=PLOT_CONFIG['legend_fontsize'])
        axs[i+1].tick_params(labelsize=PLOT_CONFIG['tick_fontsize'])
    
    # Set x-label for bottom plot
    axs[-1].set_xlabel("Time", fontsize=PLOT_CONFIG['label_fontsize'])
    
    # Set title
    error_text = f"Relative L2 Error: {error_value:.2e}"
    title = f"Trajectories: Prediction vs. Truth{title_suffix}\n{error_text}"
    fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'])
    
    # Adjust spacing
    plt.subplots_adjust(hspace=PLOT_CONFIG['hspace'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_2d_fields(
    x_input: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    error_value: float,
    title_suffix: str = "",
    n_grid: int = 100,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot 2D field comparison for neural operator evaluation.
    
    Args:
        x_input: Input field data
        y_true: True output field  
        y_pred: Predicted output field
        coords_x: Spatial coordinates for input
        coords_y: Spatial coordinates for output
        error_value: Relative L2 error value
        title_suffix: Additional text for title
        n_grid: Grid resolution for interpolation
        save_path: Path to save figure
        show: Whether to display the figure
    
    Returns:
        Matplotlib figure object
    """
    # Create 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=PLOT_CONFIG['figure_size_2d'])
    
    # Interpolate data to regular grid for visualization
    def interpolate_to_grid(data, coords):
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        xi = np.linspace(x_min, x_max, n_grid)
        yi = np.linspace(y_min, y_max, n_grid)
        xi, yi = np.meshgrid(xi, yi)
        
        zi = griddata(coords, data.flatten(), (xi, yi), method='cubic')
        return zi
    
    # Prepare data
    if coords_x.shape[-1] == 2:  # 2D coordinates
        x_grid = interpolate_to_grid(x_input, coords_x)
        y_true_grid = interpolate_to_grid(y_true, coords_y)
        y_pred_grid = interpolate_to_grid(y_pred, coords_y)
    else:
        # Assume data is already on a grid
        x_grid = x_input.reshape(n_grid, n_grid)
        y_true_grid = y_true.reshape(n_grid, n_grid)
        y_pred_grid = y_pred.reshape(n_grid, n_grid)
    
    # Compute pointwise error
    error_grid = np.abs(y_pred_grid - y_true_grid)
    
    # Plot panels
    panels = [
        (x_grid, "Input Field", PLOT_CONFIG['colormap_data']),
        (y_true_grid, "Ground Truth", PLOT_CONFIG['colormap_data']),
        (y_pred_grid, "Prediction", PLOT_CONFIG['colormap_data']),
        (error_grid, "Pointwise Absolute Error", PLOT_CONFIG['colormap_error'])
    ]
    
    for i, (data, title, cmap) in enumerate(panels):
        row, col = i // 2, i % 2
        im = axs[row, col].imshow(data, cmap=cmap, aspect='auto', origin='lower')
        axs[row, col].set_title(title, fontsize=PLOT_CONFIG['label_fontsize'])
        
        # Set coordinate labels
        axs[row, col].set_xticks(np.linspace(0, n_grid-1, 3))
        axs[row, col].set_xticklabels(['0', '0.5', '1'])
        axs[row, col].set_yticks(np.linspace(0, n_grid-1, 3))
        axs[row, col].set_yticklabels(['0', '0.5', '1'])
        
        # Add colorbar
        plt.colorbar(im, ax=axs[row, col], shrink=0.6)
    
    # Set overall title
    error_text = f"Relative L2 Error: {error_value:.2e}"
    title = f"2D Fields: Prediction vs. Truth{title_suffix}\n{error_text}"
    fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'])
    
    # Adjust spacing
    plt.subplots_adjust(hspace=PLOT_CONFIG['hspace'], wspace=PLOT_CONFIG['wspace'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_error_samples(
    samples: List[Dict[str, Any]],
    sample_types: List[str] = ['median', 'max'],
    domain_type: str = '1d',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot multiple error samples in a comparison grid.
    
    Args:
        samples: List of sample dictionaries with keys:
                {'x_input', 'y_true', 'y_pred', 'coords_x', 'coords_y', 'error'}
        sample_types: Types of samples ('median', 'max', etc.)
        domain_type: '1d' or '2d'
        save_path: Path to save figure
        show: Whether to display figure
    
    Returns:
        Matplotlib figure object
    """
    if domain_type == '1d':
        # Create figure with samples side by side
        n_samples = len(samples)
        output_dim = samples[0]['y_true'].shape[-1]
        
        fig, axs = plt.subplots(
            nrows=output_dim + 1,
            ncols=n_samples,
            figsize=(PLOT_CONFIG['figure_size_1d'][0] * n_samples, 
                    PLOT_CONFIG['figure_size_1d'][1] * (output_dim + 1)),
            sharex=True,
            squeeze=False
        )
        
        for col, (sample, sample_type) in enumerate(zip(samples, sample_types)):
            # Plot input
            axs[0, col].plot(
                sample['coords_x'].flatten(),
                sample['x_input'][:, 0],
                linewidth=PLOT_CONFIG['linewidth'],
                color=PLOT_CONFIG['color_input'],
                label="Input"
            )
            axs[0, col].set_title(
                f"{sample_type.title()} Error Sample\n"
                f"Rel. L2 Error: {sample['error']:.2e}",
                fontsize=PLOT_CONFIG['label_fontsize']
            )
            axs[0, col].legend(fontsize=PLOT_CONFIG['legend_fontsize'])
            
            # Plot outputs
            for i in range(output_dim):
                axs[i+1, col].plot(
                    sample['coords_y'].flatten(),
                    sample['y_true'][:, i],
                    linewidth=PLOT_CONFIG['linewidth'],
                    color=PLOT_CONFIG['color_truth'],
                    label="Ground Truth"
                )
                axs[i+1, col].plot(
                    sample['coords_y'].flatten(),
                    sample['y_pred'][:, i],
                    linewidth=PLOT_CONFIG['linewidth'],
                    color=PLOT_CONFIG['color_prediction'],
                    label="Prediction"
                )
                axs[i+1, col].legend(fontsize=PLOT_CONFIG['legend_fontsize'])
        
        # Labels for leftmost column
        axs[0, 0].set_ylabel("Input", fontsize=PLOT_CONFIG['label_fontsize'])
        for i in range(output_dim):
            axs[i+1, 0].set_ylabel(f"Output {i+1}", fontsize=PLOT_CONFIG['label_fontsize'])
        
        # X-labels for bottom row
        for col in range(n_samples):
            axs[-1, col].set_xlabel("Time", fontsize=PLOT_CONFIG['label_fontsize'])
    
    else:  # 2D
        # Create comparison grid for 2D data
        fig, axs = plt.subplots(
            len(samples), 4,
            figsize=(PLOT_CONFIG['figure_size_2d'][0], 
                    PLOT_CONFIG['figure_size_2d'][1] * len(samples))
        )
        
        if len(samples) == 1:
            axs = axs.reshape(1, -1)
        
        for row, (sample, sample_type) in enumerate(zip(samples, sample_types)):
            # Similar to plot_2d_fields but in grid format
            panels = [
                sample['x_input'], sample['y_true'], 
                sample['y_pred'], np.abs(sample['y_pred'] - sample['y_true'])
            ]
            titles = ["Input", "Truth", "Prediction", "Error"]
            
            for col, (data, title) in enumerate(zip(panels, titles)):
                cmap = PLOT_CONFIG['colormap_error'] if col == 3 else PLOT_CONFIG['colormap_data']
                im = axs[row, col].imshow(data, cmap=cmap, aspect='auto')
                
                if row == 0:
                    axs[row, col].set_title(title, fontsize=PLOT_CONFIG['label_fontsize'])
                
                plt.colorbar(im, ax=axs[row, col], shrink=0.6)
            
            # Add error info to leftmost plot
            axs[row, 0].text(
                0.02, 0.98, 
                f"{sample_type.title()}\nError: {sample['error']:.2e}",
                transform=axs[row, 0].transAxes,
                verticalalignment='top',
                fontsize=PLOT_CONFIG['legend_fontsize'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
    
    fig.suptitle("Error Sample Comparison", fontsize=PLOT_CONFIG['title_fontsize'])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_training_history(
    metrics_history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training metrics over epochs.
    
    Args:
        metrics_history: Dict with metric names as keys, lists of values as values
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    
    Returns:
        Matplotlib figure object
    """
    n_metrics = len(metrics_history)
    if n_metrics == 0:
        return None
    
    # Determine grid layout
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if n_metrics == 1:
        axs = [axs]
    elif rows == 1:
        axs = axs.flatten()
    else:
        axs = axs.flatten()
    
    for i, (metric_name, values) in enumerate(metrics_history.items()):
        if i < len(axs):
            axs[i].plot(values, linewidth=2)
            axs[i].set_title(metric_name, fontsize=PLOT_CONFIG['label_fontsize'])
            axs[i].set_xlabel("Epoch", fontsize=PLOT_CONFIG['label_fontsize'])
            axs[i].grid(True, alpha=0.3)
            axs[i].tick_params(labelsize=PLOT_CONFIG['tick_fontsize'])
    
    # Hide unused subplots
    for i in range(len(metrics_history), len(axs)):
        axs[i].set_visible(False)
    
    fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_comparison(
    x_input: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    coords_x: Optional[Union[np.ndarray, torch.Tensor]] = None,
    coords_y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    title: str = "Model Comparison",
    domain_type: str = 'auto'
) -> plt.Figure:
    """
    Convenience plotting function for model evaluation visualization.
    
    Args:
        x_input: Input data
        y_true: True output
        y_pred: Predicted output
        coords_x: Input coordinates (optional)
        coords_y: Output coordinates (optional)
        title: Plot title
        domain_type: '1d', '2d', or 'auto'
    
    Returns:
        Matplotlib figure object
    """
    # Convert tensors to numpy
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x
    
    x_input = to_numpy(x_input)
    y_true = to_numpy(y_true) 
    y_pred = to_numpy(y_pred)
    
    if coords_x is not None:
        coords_x = to_numpy(coords_x)
    if coords_y is not None:
        coords_y = to_numpy(coords_y)
    
    # Auto-detect domain type
    if domain_type == 'auto':
        if coords_x is not None and coords_x.shape[-1] == 2:
            domain_type = '2d'
        elif x_input.ndim >= 2 and x_input.shape[-1] > 10:  # Likely 2D flattened
            domain_type = '2d'
        else:
            domain_type = '1d'
    
    # Create default coordinates if not provided
    if coords_x is None:
        coords_x = np.arange(x_input.shape[0]).reshape(-1, 1)
    if coords_y is None:
        coords_y = np.arange(y_true.shape[0]).reshape(-1, 1)
    
    # Compute error
    error = np.sqrt(np.mean((y_pred - y_true)**2)) / np.sqrt(np.mean(y_true**2))
    
    # Plot based on domain type
    if domain_type == '1d':
        return plot_1d_trajectories(
            x_input, y_true, y_pred, coords_x, coords_y, 
            error, title_suffix=f" - {title}"
        )
    else:
        return plot_2d_fields(
            x_input, y_true, y_pred, coords_x, coords_y,
            error, title_suffix=f" - {title}"
        )
