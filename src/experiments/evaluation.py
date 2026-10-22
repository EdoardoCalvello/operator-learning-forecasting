"""
Evaluation experiment for loading trained models and computing test metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import wandb
from pathlib import Path

from src.experiments.base import BaseExperiment
from src.utils.metrics import relative_l2_loss
from src.utils.plotting import plot_1d_trajectories


class EvaluationExperiment(BaseExperiment):
    """Experiment class for evaluating trained models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        wandb_run_id: Optional[str] = None,
        wandb_project: Optional[str] = None,
        system_name: str = "L63",
        **kwargs
    ):
        """
        Initialize evaluation experiment.
        
        Args:
            checkpoint_path: Path to model checkpoint
            wandb_run_id: WandB run ID to load config from (optional)
            wandb_project: WandB project name (optional)
            system_name: Name of dynamical system
        """
        self.checkpoint_path = checkpoint_path
        self.wandb_run_id = wandb_run_id
        self.wandb_project = wandb_project
        self.system_name = system_name
        
        # If wandb run ID provided, load config from there
        if wandb_run_id and wandb_project:
            self._load_config_from_wandb()
        
        # Initialize base experiment
        super().__init__(**kwargs)
        
    def _load_config_from_wandb(self):
        """Load configuration from WandB run."""
        api = wandb.Api()
        run = api.run(f"{self.wandb_project}/{self.wandb_run_id}")
        
        # Extract configuration
        config = run.config
        
        # Parse into structured configs
        # NOTE: Exclude 'plot_inds' from WandB - it should come from local config only
        self.dataset_config = {
            k: v for k, v in config.items() 
            if k in ['dyn_sys_name', 'T', 'dt', 'sample_rate', 'burn_in_time',
                    'prediction_horizon', 'input_inds', 'output_inds',
                    'n_trajectories_train', 'n_trajectories_val', 'n_trajectories_test',
                    'params']
        }
        
        self.model_config = {
            k: v for k, v in config.items()
            if k in ['d_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout',
                    'activation', 'use_positional_encoding', 'append_position_to_x',
                    'pos_enc_coeff', 'domain_dim']
        }
        
        self.training_config = {
            k: v for k, v in config.items()
            if k in ['batch_size', 'learning_rate', 'max_epochs', 'monitor_metric',
                    'lr_scheduler_params', 'gradient_clip_val', 'log_every_n_steps',
                    'loss_name']
        }
        
    def _configure_experiment_specifics(self):
        """Not needed for evaluation."""
        pass
    
    def load_model_from_checkpoint(self):
        """Load trained model from checkpoint."""
        from src.models.tno import SimpleEncoderModule
        
        # Load checkpoint
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        
        # Create model with saved configuratio
        
        if self.system_name in ('KSE', 'KuramotoSivashinsky'):
            input_dim = self.dataset_config.get('state_dim', 128)
            output_dim = self.dataset_config.get('state_dim', 128)
        else:
            input_dim = len(self.dataset_config.get('input_inds', [0]))
            output_dim = len(self.dataset_config.get('output_inds', [1, 2]))
        
        model_config = self.model_config.copy()
        use_cross_attention = model_config.pop('use_cross_attention', True)
        prediction_horizon = self.dataset_config.get('prediction_horizon', 0)
        sample_rate = self.dataset_config.get(
            'sample_rate', self.dataset_config.get('dt', 1))
        if prediction_horizon > 0 and use_cross_attention:
            query_seq_len = int(prediction_horizon / sample_rate)
            query_coords = torch.linspace(
                sample_rate, prediction_horizon, query_seq_len
            ).unsqueeze(-1) / prediction_horizon
            output_seq_len = None
        else:
            query_seq_len = None
            query_coords = None
            output_seq_len = (
                int(prediction_horizon / sample_rate)
                if prediction_horizon > 0 else None
            )
        model_config.update({
            'input_dim': input_dim,
            'output_dim': output_dim,
            'learning_rate': self.training_config['learning_rate'],
            'lr_scheduler_params': self.training_config['lr_scheduler_params'],
            'loss_name': self.training_config.get('loss_name', 'l2_rel'),
            'monitor_metric': self.training_config.get('monitor_metric', 'loss/val/rel_L2'),
            'query_seq_len': query_seq_len,
            'query_coords': query_coords,
            'output_seq_len': output_seq_len,
        })
        
        self.model = SimpleEncoderModule(**model_config)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ Model loaded from {self.checkpoint_path}")
        return self.model
    
    def prepare_test_dataset_only(self):
        """Create only the test dataset (skip train/val for evaluation)."""
        from src.datasets.dynamical import load_dataset
        import torch
        
        if self.dataset_config.get('saved_data_dir'):
            from src.datasets.saved import load_saved_split
            self.test_dataset = load_saved_split(self.dataset_config, 'eval', self.seed, compositions=1)
            self.test_T = self.dataset_config['T']
            return

        # Create test dataset only
        test_config = self.dataset_config.copy()
        test_config['size'] = 2000#test_config.get('n_trajectories_test', 2000)
        test_config['test'] = True
        
        # CRITICAL: Ensure test data independence from training data
        # Strategy 1: Use different seed for test data
        test_seed = 12345  # Different from training seed = 0
        torch.manual_seed(test_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(test_seed)
        
        # Strategy 2: Use different burn-in time to sample different parts of attractor
        # Increase burn-in time by 50% for test set
        original_burn_in = test_config.get('burn_in_time', 200.0)
        test_config['burn_in_time'] = original_burn_in * 5
        
        print(f"Loading test dataset with:")
        print(f"  - Seed: {test_seed} (train uses seed 0)")
        print(f"  - Burn-in time: {test_config['burn_in_time']:.1f} (train uses {original_burn_in:.1f})")
        print(f"  - Number of trajectories: {test_config['size']}")
        
        self.test_dataset = load_dataset(**test_config)
        print(f"✓ Test dataset loaded - guaranteed independent from training data")
    
    def evaluate(self, save_plots: bool = True, n_examples: int = 5, compute_baseline: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            save_plots: Whether to save prediction plots
            n_examples: Number of examples to plot
            compute_baseline: Whether to compute baseline errors (for forecasting experiments)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating {self.system_name} model...")
        
        # Prepare ONLY test dataset (skip train/val to save memory)
        self.prepare_test_dataset_only()
        
        # Load model
        self.load_model_from_checkpoint()
        
        # Set normalizers for denormalization
        self.model.y_normalizer = self.test_dataset.y_normalizer
        self.model.x_normalizer = getattr(self.test_dataset, 'x_normalizer', None)
        
        # Create test data loader
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,  # Evaluate one at a time for detailed analysis
            shuffle=False
        )

        for name, param in self.model.named_parameters():
            print(f"{name}: min={param.min().item():.4f}, max={param.max().item():.4f}, mean={param.abs().mean().item():.4f}")

        device = next(self.model.parameters()).device
        
        # Collect predictions and errors
        errors = []
        predictions = []
        ground_truths = []
        inputs = []
        
        print("Computing predictions on test set...")
        with torch.no_grad():
            for idx, (x, y, coords_x, coords_y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                coords_x = coords_x.to(device)
                coords_y = coords_y.to(device)
                
                # Forward pass
                y_pred = self.model(x, y, coords_x, coords_y)
                
                # Compute relative L2 error
                error = relative_l2_loss(y_pred, y, reduction='none')
                
                # Store results as tensors (keep on CPU)
                errors.append(error.cpu())
                predictions.append(y_pred.cpu())
                ground_truths.append(y.cpu())
                inputs.append(x.cpu())
        
        # Stack into tensors: (n_test, 1, seq_len, features)
        errors = torch.stack(errors).squeeze()        # (n_test,)
        predictions = torch.stack(predictions)        # (n_test, 1, seq_len, n_modes)
        ground_truths = torch.stack(ground_truths)
        inputs = torch.stack(inputs)

        # For KS: denormalize and convert Fourier → physical space, then recompute errors
        if self.dataset_config.get('dyn_sys_name') == 'KuramotoSivashinsky':
            print("KS detected: denormalizing and converting Fourier → physical space for error computation...")
            y_normalizer = self.model.y_normalizer
            x_normalizer = self.model.x_normalizer

            # Squeeze batch dim: (n_test, seq_len, n_modes)
            y_p = predictions[:, 0] if predictions.ndim == 4 else predictions
            y_t = ground_truths[:, 0] if ground_truths.ndim == 4 else ground_truths
            x_in = inputs[:, 0] if inputs.ndim == 4 else inputs

            if y_normalizer is not None:
                y_p = y_normalizer.decode(y_p)
                y_t = y_normalizer.decode(y_t)
            if x_normalizer is not None:
                x_in = x_normalizer.decode(x_in)

            # Recompute relative L2 errors in physical space
            errors = (
                torch.sqrt(torch.mean((y_p - y_t) ** 2, dim=(-2, -1))) /
                (torch.sqrt(torch.mean(y_t ** 2, dim=(-2, -1))) + 1e-8)
            )  # (n_test,)

            # Re-add batch dim of 1 to keep the rest of the code consistent
            predictions = y_p.unsqueeze(1)    # (n_test, 1, seq_len, state_dim)
            ground_truths = y_t.unsqueeze(1)
            inputs = x_in.unsqueeze(1)

        # Convert to numpy for all downstream code
        errors = errors.numpy()
        predictions = predictions.numpy()
        ground_truths = ground_truths.numpy()
        inputs = inputs.numpy()

        # Compute statistics
        avg_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        
        # Initialize baseline variables
        baseline_errors = None
        avg_baseline_error = None
        median_baseline_error = None
        improvement = None
        
        # Compute baseline error: predict constant from last input point (only for forecasting)
        if compute_baseline:
            print("Computing baseline (constant prediction) error...")
            baseline_errors = []
            for i in range(len(inputs)):
                x_i = inputs[i]
                y_true_i = ground_truths[i]
                
                # Get last point of input - shape depends on batch dimension
                if x_i.ndim == 3:
                    last_input = x_i[0, -1, :]  # (features,)
                elif x_i.ndim == 2:
                    last_input = x_i[-1, :]  # (features,)
                else:
                    last_input = x_i[-1]  # scalar
                
                # Create constant prediction: repeat last input for all output timesteps
                if y_true_i.ndim == 3:
                    y_baseline = np.tile(last_input, (y_true_i.shape[1], 1))  # (seq_len, features)
                    y_baseline = y_baseline[np.newaxis, :, :]  # Add batch dim
                    y_true_temp = y_true_i
                elif y_true_i.ndim == 2:
                    y_baseline = np.tile(last_input, (y_true_i.shape[0], 1))  # (seq_len, features)
                    y_true_temp = y_true_i
                else:
                    y_baseline = np.full_like(y_true_i, last_input)
                    y_true_temp = y_true_i
                
                # Compute relative L2 error for baseline
                baseline_error = np.sqrt(np.mean((y_baseline - y_true_temp)**2)) / (np.sqrt(np.mean(y_true_temp**2)) + 1e-8)
                baseline_errors.append(baseline_error)
            
            baseline_errors = np.array(baseline_errors)
            avg_baseline_error = np.mean(baseline_errors)
            median_baseline_error = np.median(baseline_errors)
            
            # Compute improvement over baseline
            improvement = ((avg_baseline_error - avg_error) / avg_baseline_error) * 100 if avg_baseline_error > 0 else 0
        
        
        # Compute mean and variance statistics per trajectory
        print("Computing per-trajectory statistics (mean and variance over time)...")
        
        # For predictions - compute per trajectory (across time only, keep features separate)
        # predictions shape: (n_test, 1, seq_len, features) or (n_test, seq_len, features)
        if predictions.ndim == 4:
            preds_for_stats = predictions[:, 0, :, :]  # (n_test, seq_len, features)
            truths_for_stats = ground_truths[:, 0, :, :]
        else:
            preds_for_stats = predictions  # Already (n_test, seq_len, features)
            truths_for_stats = ground_truths
        
        # Compute mean and variance per trajectory over time (axis=1 is time dimension)
        # This gives us mean/var for each trajectory and each feature separately
        pred_means_per_traj = np.mean(preds_for_stats, axis=1)  # (n_test, features)
        pred_vars_per_traj = np.var(preds_for_stats, axis=1)    # (n_test, features)
        true_means_per_traj = np.mean(truths_for_stats, axis=1)  # (n_test, features)
        true_vars_per_traj = np.var(truths_for_stats, axis=1)    # (n_test, features)
        
        # Initialize baseline statistics variables
        mean_error_baseline = None
        var_error_baseline = None
        
        # Compute baseline statistics per trajectory (only for forecasting)
        if compute_baseline:
            baseline_means_per_traj = []
            baseline_vars_per_traj = []
            
            for i in range(len(inputs)):
                x_i = inputs[i]
                y_true_i = ground_truths[i]
                
                # Get last point of input
                if x_i.ndim == 3:
                    last_input = x_i[0, -1, :]
                elif x_i.ndim == 2:
                    last_input = x_i[-1, :]
                else:
                    last_input = x_i[-1]
                
                # Create constant baseline prediction
                if y_true_i.ndim == 3:
                    y_baseline = np.tile(last_input, (y_true_i.shape[1], 1))
                elif y_true_i.ndim == 2:
                    y_baseline = np.tile(last_input, (y_true_i.shape[0], 1))
                else:
                    y_baseline = np.full_like(y_true_i, last_input)
                
                # Compute mean and variance for this baseline trajectory over time (per feature)
                baseline_means_per_traj.append(np.mean(y_baseline, axis=0))  # (features,)
                baseline_vars_per_traj.append(np.var(y_baseline, axis=0))    # (features,)
            
            baseline_means_per_traj = np.array(baseline_means_per_traj)  # (n_test, features)
            baseline_vars_per_traj = np.array(baseline_vars_per_traj)    # (n_test, features)
            
            # Compute errors per feature, then average across features
            # First compute absolute errors for each trajectory and feature
            mean_errors_baseline_per_feature = np.abs(baseline_means_per_traj - true_means_per_traj)  # (n_test, features)
            var_errors_baseline_per_feature = np.abs(baseline_vars_per_traj - true_vars_per_traj)      # (n_test, features)
            
            # Average across features first, then across trajectories
            mean_error_baseline = np.mean(np.mean(mean_errors_baseline_per_feature, axis=1))
            var_error_baseline = np.mean(np.mean(var_errors_baseline_per_feature, axis=1))
        
        # Compute errors per feature, then average across features
        # First compute absolute errors for each trajectory and feature
        mean_errors_learned_per_feature = np.abs(pred_means_per_traj - true_means_per_traj)  # (n_test, features)
        var_errors_learned_per_feature = np.abs(pred_vars_per_traj - true_vars_per_traj)      # (n_test, features)
        
        # Average across features first, then across trajectories
        mean_error_learned = np.mean(np.mean(mean_errors_learned_per_feature, axis=1))  # Average over features then trajectories
        var_error_learned = np.mean(np.mean(var_errors_learned_per_feature, axis=1))
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Model Performance:")
        print(f"  Average Relative L2 Error:  {avg_error:.6f}")
        print(f"  Median Relative L2 Error:   {median_error:.6f}")
        print(f"  Std Dev Relative L2 Error:  {std_error:.6f}")
        print(f"  Min Relative L2 Error:      {min_error:.6f}")
        print(f"  Max Relative L2 Error:      {max_error:.6f}")
        
        if compute_baseline:
            print(f"\nBaseline Performance (constant from last input):")
            print(f"  Average Relative L2 Error:  {avg_baseline_error:.6f}")
            print(f"  Median Relative L2 Error:   {median_baseline_error:.6f}")
            print(f"\nImprovement over Baseline:   {improvement:.2f}%")
            print(f"\nTrajectory Statistics Comparison:")
            print(f"  Mean Error (Learned):       {mean_error_learned:.6f}")
            print(f"  Mean Error (Baseline):      {mean_error_baseline:.6f}")
            print(f"  Variance Error (Learned):   {var_error_learned:.6f}")
            print(f"  Variance Error (Baseline):  {var_error_baseline:.6f}")
        else:
            print(f"\nTrajectory Statistics:")
            print(f"  Mean Error (Learned):       {mean_error_learned:.6f}")
            print(f"  Variance Error (Learned):   {var_error_learned:.6f}")
        
        print("="*60 + "\n")
        
        # Create plots for selected examples
        if save_plots:
            self._plot_examples(
                inputs, ground_truths, predictions, errors,
                n_examples=n_examples
            )
        
        # Return results
        results = {
            'avg_error': avg_error,
            'median_error': median_error,
            'std_error': std_error,
            'min_error': min_error,
            'max_error': max_error,
            'avg_baseline_error': avg_baseline_error,
            'median_baseline_error': median_baseline_error,
            'improvement_over_baseline': improvement,
            'mean_error_learned': mean_error_learned,
            'var_error_learned': var_error_learned,
            'mean_error_baseline': mean_error_baseline,
            'var_error_baseline': var_error_baseline,
            'all_errors': errors,
            'all_baseline_errors': baseline_errors,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'inputs': inputs
        }
        
        return results
    


    def _plot_examples(
        self, 
        inputs: np.ndarray,
        ground_truths: np.ndarray,
        predictions: np.ndarray,
        errors: np.ndarray,
        n_examples: int = 5
    ):
        """Plot median and worst cases in a 2-panel layout."""
        print(f"Generating 2-panel plot (median and worst cases)...")
        
        # Get example indices - median and worst
        sorted_indices = np.argsort(errors)
        
        median_idx = sorted_indices[len(sorted_indices) // 2]
        worst_idx = sorted_indices[-1]
        
        # Create plots directory
        plots_dir = Path("evaluation_plots")
        plots_dir.mkdir(exist_ok=True)
        
        coords_x = self.test_dataset.times_x.numpy()
        coords_y = self.test_dataset.times_y.numpy()
        
        # Ensure coords are 1D
        if coords_x.ndim > 1:
            coords_x = coords_x.squeeze()
        if coords_y.ndim > 1:
            coords_y = coords_y.squeeze()
        
        # Prepare data for both cases
        cases = {
            'median': median_idx,
            'worst': worst_idx
        }
        
        case_data = {}
        for name, idx in cases.items():
            # Get data - inputs[idx] has shape (1, seq_len, features) from batch_size=1
            x = inputs[idx]  # (1, seq_len, features)
            y_true = ground_truths[idx]  # (1, seq_len, features)
            y_pred = predictions[idx]  # (1, seq_len, features)
            error = errors[idx]
            
            # Remove batch dimension but keep (seq_len, features)
            if x.ndim == 3:
                x = x[0]  # Now (seq_len, features)
            if y_true.ndim == 3:
                y_true = y_true[0]
            if y_pred.ndim == 3:
                y_pred = y_pred[0]
            
            # If features dimension is 1, ensure we still have 2D array
            if x.ndim == 1:
                x = x[:, np.newaxis]  # (seq_len, 1)
            if y_true.ndim == 1:
                y_true = y_true[:, np.newaxis]
            if y_pred.ndim == 1:
                y_pred = y_pred[:, np.newaxis]
            
            # Denormalize for plotting (skip for KS: already in physical space from evaluate())
            is_ks = self.dataset_config.get('dyn_sys_name') == 'KuramotoSivashinsky'
            if not is_ks:
                if self.model.y_normalizer is not None:
                    y_true = self.model.y_normalizer.decode(
                        torch.from_numpy(y_true).unsqueeze(0)
                    )[0].numpy()  # Keep (seq_len, features) shape
                    y_pred = self.model.y_normalizer.decode(
                        torch.from_numpy(y_pred).unsqueeze(0)
                    )[0].numpy()

                if self.model.x_normalizer is not None:
                    x = self.model.x_normalizer.decode(
                        torch.from_numpy(x).unsqueeze(0)
                    )[0].numpy()

            case_data[name] = {
                'x': x,
                'y_true': y_true,
                'y_pred': y_pred,
                'error': error,
                'idx': idx
            }
        
        # Create combined 2-panel plot
        fig = self._create_two_panel_plot(
            case_data=case_data,
            coords_x=coords_x,
            coords_y=coords_y
        )
        
        # Save
        save_path = plots_dir / f"{self.system_name}_median_worst_comparison.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {save_path}")
        print(f"✓ Plot saved to {plots_dir}/")
    
    def _variable_label(self, output_idx):
        """Panel title naming the physical variable shown in an output column."""
        if self.system_name == 'KSE':
            # Tilde marks a low-pass field: the output keeps fewer Fourier modes than the grid
            # supports (forecasting). Smoothing outputs keep every mode, i.e. the KSE solution u.
            n_modes = self.dataset_config.get('params', {}).get('state_dim', 128) // 2 + 1
            filtered = len(self.dataset_config.get('output_inds', range(n_modes))) < n_modes
            u = r'\widetilde{u}' if filtered else 'u'
            return rf'${u}(x_{{{output_idx + 1}}})$'
        output_inds = self.dataset_config.get('output_inds', [])
        dim = output_inds[output_idx] if output_idx < len(output_inds) else output_idx
        if self.system_name == 'L63':
            return f"{'xyz'[dim]} variable"
        return rf'$X_{{{dim + 1}}}$'

    def _create_two_panel_plot(
        self,
        case_data: dict,
        coords_x: np.ndarray,
        coords_y: np.ndarray
    ):
        """
        Create a 2-panel plot showing median (top) and worst (bottom) cases.
        
        Args:
            case_data: Dictionary with 'median' and 'worst' case data
            coords_x: Time coordinates for input
            coords_y: Time coordinates for output
        """
        # Get indices to plot from config
        plot_inds = self.dataset_config.get('plot_inds', [0])
        input_inds = self.dataset_config.get('input_inds', [0])
        output_inds = self.dataset_config.get('output_inds', [0])
        
        # Determine which output dimensions to plot
        is_ks = self.dataset_config.get('dyn_sys_name') == 'KuramotoSivashinsky'

        if is_ks:
            # For KS physical space: data is (seq_len, state_dim).
            # plot_inds are treated as spatial grid indices to plot vs time.
            state_dim = self.dataset_config.get('params', {}).get('state_dim', 128)
            output_plot_inds = []
            if plot_inds is not None and len(plot_inds) > 0:
                output_plot_inds = [p for p in plot_inds if 0 <= p < state_dim]
            if not output_plot_inds:
                output_plot_inds = [0, state_dim // 4, state_dim // 2]
        else:
            if plot_inds is not None:
                output_plot_inds = []
                for plot_idx in plot_inds:
                    if plot_idx in output_inds:
                        array_pos = output_inds.index(plot_idx)
                        output_plot_inds.append(array_pos)
                
                if len(output_plot_inds) == 0:
                    output_plot_inds = list(range(min(3, case_data['median']['y_true'].shape[-1])))
            else:
                output_plot_inds = list(range(min(3, case_data['median']['y_true'].shape[-1])))
            
            # Override for specific dimensions if hardcoded
            if len(output_inds) >= 2:  # For L96 system
                output_plot_inds = [1]

        # Plot the first two outputs, or fewer when the model predicts fewer components.
        output_plot_inds = [i for i in (0, 1) if i < case_data['median']['y_true'].shape[-1]]
        n_dims = len(output_plot_inds)
        
        # Create figure: 2 rows (median, worst), multiple columns (one per dimension)
        fig, axes = plt.subplots(2, n_dims, figsize=(7*n_dims, 5), squeeze=False)
        
        for row_idx, case_name in enumerate(['median', 'worst']):
            data = case_data[case_name]
            x_input = data['x']
            y_true = data['y_true']
            y_pred = data['y_pred']
            error = data['error']
            idx = data['idx']
            
            for col_idx, output_idx in enumerate(output_plot_inds):
                ax = axes[row_idx, col_idx]
                
                if is_ks:
                    # output_idx is a spatial grid index; plot vs time
                    ax.plot(coords_y, y_true[:, output_idx],
                           color='blue', linestyle='-', linewidth=2,
                           label='True', alpha=0.9)
                    ax.plot(coords_y, y_pred[:, output_idx],
                           color='red', linestyle='-', linewidth=2,
                           label='Predicted', alpha=0.9)
                    if row_idx == 1:
                        ax.set_xlabel('Time', fontsize=12)
                else:
                    # Get the actual system dimension being plotted
                    actual_output_dim = output_inds[output_idx] if output_idx < len(output_inds) else output_idx
                    
                    # Find corresponding input dimension (if it exists in inputs)
                    corresponding_input_idx = None
                    if actual_output_dim in input_inds:
                        corresponding_input_idx = input_inds.index(actual_output_dim)
                    
                    # Plot corresponding input trajectory if available (blue dashed)
                    if corresponding_input_idx is not None:
                        ax.plot(coords_x, x_input[:, corresponding_input_idx], 
                               color='blue', linestyle='--', linewidth=2, 
                               label='Input', alpha=0.7)
                    
                    # Plot true output (blue solid)
                    ax.plot(coords_y, y_true[:, output_idx], 
                           color='blue', linestyle='-', linewidth=2, 
                           label='True', alpha=0.9)
                    
                    # Plot predicted output (red solid)
                    ax.plot(coords_y, y_pred[:, output_idx],
                           color='red', linestyle='-', linewidth=2,
                           label='Predicted', alpha=0.9)
                    if row_idx == 1:
                        ax.set_xlabel('Time', fontsize=12)
                
                # Styling
                if col_idx == 0:
                    if row_idx == 0:
                        ax.set_ylabel(f'Median error sample', fontsize=12)
                    else:
                        ax.set_ylabel(f'Highest error sample', fontsize=12)
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=10)
                
                # Add title to top row
                if row_idx == 0:
                    ax.set_title(self._variable_label(output_idx), fontsize=13)
                    
                # Add row label on left column
                # if col_idx == 0:
                #     ylabel = f'{case_name.capitalize()}\n(idx={idx}, error={error:.6f})'
                #     ax.text(-0.25, 0.5, ylabel, transform=ax.transAxes,
                #            fontsize=12, fontweight='bold', va='center', rotation=90)
        
        # Overall title
        task = 'Forecasting' if self.dataset_config.get('prediction_horizon', 0) > 0 else 'Smoothing'
        fig.suptitle(f'{self.system_name} {task} Experiment', 
                    fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        return fig
    
    def _create_combined_plot(
        self,
        x_input: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        coords_x: np.ndarray,
        coords_y: np.ndarray,
        error_value: float,
        title: str
    ):
        """
        Create a combined plot showing input and output trajectories.
        
        Args:
            x_input: Input trajectory (seq_len, n_input_features)
            y_true: True output trajectory (seq_len, n_output_features)
            y_pred: Predicted output trajectory (seq_len, n_output_features)
            coords_x: Time coordinates for input
            coords_y: Time coordinates for output
            error_value: Relative L2 error
            title: Plot title
        """
        # Get indices to plot from config
        plot_inds = self.dataset_config.get('plot_inds', [0])
        input_inds = self.dataset_config.get('input_inds', [0])
        output_inds = self.dataset_config.get('output_inds', [0])
        
        print(f"\n=== Plotting Configuration Debug ===")
        print(f"plot_inds from config: {plot_inds}")
        print(f"input_inds: {input_inds[:5]}... (total {len(input_inds)})")
        print(f"output_inds: {output_inds}")
        print(f"y_true shape: {y_true.shape}")
        print(f"x_input shape: {x_input.shape}")
        
        # Determine which output dimensions to plot
        if plot_inds is not None:
            # Use specified plot indices (these are system dimensions, not array indices)
            output_plot_inds = []
            for plot_idx in plot_inds:
                if plot_idx in output_inds:
                    # Find position in output array
                    array_pos = output_inds.index(plot_idx)
                    output_plot_inds.append(array_pos)
            
            if len(output_plot_inds) == 0:
                # Fallback: plot first 3 outputs
                output_plot_inds = list(range(min(3, y_true.shape[-1])))
        else:
            # Default: plot first 3 output dimensions
            output_plot_inds = list(range(min(3, y_true.shape[-1])))
        output_plot_inds = [9]
        #[9,17,30]
        
        # Debug: print what will be plotted
        actual_dims_plotted = [output_inds[idx] if idx < len(output_inds) else idx for idx in output_plot_inds]
        print(f"\n✓ Will plot system dimensions: {actual_dims_plotted}")
        print(f"  (Array indices in output: {output_plot_inds})")
        print(f"=== End Debug ===\n")
        
        n_plots = len(output_plot_inds)
        
        # Create figure with subplots for selected output dimensions
        fig, axs = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), squeeze=False)
        axs = axs.flatten()
        
        for plot_idx, output_idx in enumerate(output_plot_inds):
            ax = axs[plot_idx]
            
            # Get the actual system dimension being plotted
            actual_output_dim = output_inds[output_idx] if output_idx < len(output_inds) else output_idx
            
            # Find corresponding input dimension (if it exists in inputs)
            corresponding_input_idx = None
            if actual_output_dim in input_inds:
                corresponding_input_idx = input_inds.index(actual_output_dim)
            
            # Plot corresponding input trajectory if available (blue dashed)
            if corresponding_input_idx is not None:
                ax.plot(coords_x, x_input[:, corresponding_input_idx], 
                       color='blue', linestyle='--', linewidth=2, 
                       label=f'Input (dim {actual_output_dim})', alpha=0.7)
            
            # Plot true output (blue solid)
            ax.plot(coords_y, y_true[:, output_idx], 
                   color='blue', linestyle='-', linewidth=2, 
                   label=f'True output (dim {actual_output_dim})', alpha=0.9)
            
            # Plot predicted output (red solid)
            ax.plot(coords_y, y_pred[:, output_idx], 
                   color='red', linestyle='-', linewidth=2, 
                   label=f'Predicted output (dim {actual_output_dim})', alpha=0.9)
            
            # Styling
            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel(f'Dimension {actual_output_dim}', fontsize=14)
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)
        
        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def run(self):
        """Run evaluation instead of training."""
        return self.evaluate()
