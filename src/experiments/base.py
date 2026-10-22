"""
Base experiment class for neural operator experiments on dynamical systems.
"""

from datetime import timedelta
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, EarlyStopping
import wandb
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from src.datasets.dynamical import load_dataset


class WandbRunIdInCheckpoint(Callback):
    """Record the W&B run id in each checkpoint so a resumed training logs to the same run."""

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if isinstance(trainer.logger, WandbLogger):
            checkpoint['wandb_run_id'] = trainer.logger.experiment.id


class BaseExperiment(ABC):
    """Base class for all experiments on dynamical systems."""
    
    def __init__(
        self,
        # Experiment metadata
        project_name: str = "experiment",
        experiment_name: str = "base",
        seed: int = 0,
        
        # Dataset configuration
        dataset_config: Optional[Dict[str, Any]] = None,
        
        # Model configuration  
        model_config: Optional[Dict[str, Any]] = None,
        
        # Training configuration
        training_config: Optional[Dict[str, Any]] = None,
        
        **kwargs
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.seed = seed
        
        # Set default configurations and merge with provided ones
        self.dataset_config = self._get_default_dataset_config()
        if dataset_config:
            self.dataset_config.update(dataset_config)
            
        self.model_config = self._get_default_model_config() 
        if model_config:
            self.model_config.update(model_config)
            
        self.training_config = self._get_default_training_config()
        if training_config:
            self.training_config.update(training_config)
        
        # Apply experiment-specific dataset configurations
        self._configure_experiment_specifics()
        
        # Set random seed
        seed_everything(self.seed, workers=True)
        
        # Initialize components
        self.datasets = None
        self.model = None
        self.trainer = None
        
    @abstractmethod
    def _configure_experiment_specifics(self):
        """Configure dataset parameters specific to this experiment type."""
        pass
    
    def _get_default_dataset_config(self) -> Dict[str, Any]:
        """Default dataset configuration."""
        return {
            'dyn_sys_name': 'Lorenz63',
            'size': 1000,
            'T': 2.0,
            'dt': 0.01,
            'sample_rate': 0.01,
            'input_inds': [0],
            'output_inds': [1, 2],
            'plot_inds': [1, 2],
            'prediction_horizon': 0.0,  # 0.0 = smoothing, >0.0 = forecasting
            'burn_in_time': 5.0,  # Remove transients to ensure trajectory is on attractor
            'params': {}
        }
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """Default model configuration optimized for L63/L96 experiments."""
        return {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'activation': 'gelu',
            'use_positional_encoding': True,
            'append_position_to_x': True,
            'pos_enc_coeff': 2,
            'domain_dim': 1
        }
    
    def _get_default_training_config(self) -> Dict[str, Any]:
        """Default training configuration."""
        return {
            'batch_size': 64,
            'learning_rate': 1e-3,
            'max_epochs': 100,
            'monitor_metric': 'loss/val/mse',
            'lr_scheduler_params': {'patience': 2, 'factor': 0.5},
            'gradient_clip_val': 10.0,
            'log_every_n_steps': 10
        }
    
    def prepare_datasets(self):
        """Create train/val/test datasets."""
        if self.dataset_config.get('saved_data_dir'):
            from src.datasets.saved import load_saved_split
            for split in ('train', 'val', 'test'):
                setattr(self, f'{split}_dataset', load_saved_split(self.dataset_config, split, self.seed))
            return

        # Create training dataset
        train_config = self.dataset_config.copy()
        # The model predicts one horizon; compositions are for rollout evaluation.
        train_config['compositions'] = 1
        train_config['size'] = train_config.get('n_trajectories_train', 10000)
        train_config['test'] = False
        self.train_dataset = load_dataset(**train_config)
        
        # Create validation dataset
        val_config = self.dataset_config.copy()
        # The model predicts one horizon; compositions are for rollout evaluation.
        val_config['compositions'] = 1
        val_config['size'] = val_config.get('n_trajectories_val', 2000)
        val_config['test'] = False
        self.val_dataset = load_dataset(**val_config)
        
        # Create test dataset
        test_config = self.dataset_config.copy()
        # The model predicts one horizon; compositions are for rollout evaluation.
        test_config['compositions'] = 1
        test_config['size'] = test_config.get('n_trajectories_test', 2000)
        test_config['test'] = True
        self.test_dataset = load_dataset(**test_config)
    
    def prepare_model(self):
        """Initialize the model. Should be implemented by subclasses."""
        # Import here to avoid circular imports
        try:
            from src.models.tno import SimpleEncoderModule
            
            # Calculate input/output dimensions based on experiment type
            if self.dataset_config['dyn_sys_name'] == 'KuramotoSivashinsky':
                input_dim = 128
                output_dim = 128
            else:
                input_dim = len(self.dataset_config.get('input_inds', [0]))
                output_dim = len(self.dataset_config.get('output_inds', [1, 2]))

            # Create model with experiment configuration
            model_config = self.model_config.copy()
            use_cross_attention = model_config.pop('use_cross_attention', True)

            prediction_horizon = self.dataset_config.get('prediction_horizon', 0)
            dt = self.dataset_config.get('dt', 1)
            sample_rate = self.dataset_config.get('sample_rate', dt)

            query_coords_cfg = self.dataset_config.get('query_coords', None)
            if query_coords_cfg is not None:
                # Explicit list of timestamps from config, e.g. [0.25, 0.5, 0.75, 1.0]
                query_coords = torch.tensor(query_coords_cfg, dtype=torch.float32).unsqueeze(-1)
                query_coords = query_coords / query_coords.abs().max()  # normalize to [0,1]
                query_seq_len = query_coords.shape[0]
            elif prediction_horizon > 0 and use_cross_attention:
                query_seq_len = int(prediction_horizon / sample_rate)
                # Query coords: time points in the prediction horizon, normalized to [0,1], shape (query_seq_len, 1)
                query_coords = torch.linspace(sample_rate, prediction_horizon, query_seq_len).unsqueeze(-1) / prediction_horizon
            else:
                query_seq_len = None
                query_coords = None

            # Encoder-only forecasting retains the most recent latent token for
            # each requested output step. Every retained token has attended to
            # the complete input history through self-attention.
            output_seq_len = (
                int(prediction_horizon / sample_rate)
                if prediction_horizon > 0 and not use_cross_attention
                else None
            )

            model_config.update({
                'input_dim': input_dim,
                'output_dim': output_dim,
                'learning_rate': self.training_config['learning_rate'],
                'lr_scheduler_params': self.training_config['lr_scheduler_params'],
                'loss_name': self.training_config.get('loss_name', 'l2_rel'),
                'monitor_metric': self.training_config.get('monitor_metric', 'loss/val/rel_L2'),
                'dyn_sys_name': self.dataset_config.get('dyn_sys_name', 'Lorenz63'),
                'output_inds': self.dataset_config.get('output_inds', list(range(output_dim))),
                'state_dim': self.dataset_config.get('params', {}).get('state_dim', 128),
                'query_seq_len': query_seq_len,
                'query_coords': query_coords,
                'output_seq_len': output_seq_len,
            })
            
            self.model = SimpleEncoderModule(**model_config)
            
        except ImportError as e:
            raise NotImplementedError(
                f"Model preparation failed: {e}. "
                "Make sure the TNO models are properly set up in src.models.tno"
            )
    
    def prepare_trainer(self):
        """Setup PyTorch Lightning trainer with callbacks and logging."""
        # Setup wandb logger
        run_name = f"{self.experiment_name}_{self.dataset_config['dyn_sys_name']}"
        obs_noise_std = self.dataset_config.get('obs_noise_std') or 0.0
        if obs_noise_std > 0.0:
            run_name += f"_obsnoise_{obs_noise_std}"
        # A resumed training continues its original W&B run rather than starting a new one.
        resume_id = self._resume_wandb_run_id()
        logger = WandbLogger(
            project=self.project_name,
            name=run_name,
            id=resume_id,
            resume='must' if resume_id else None,
            config={
                **self.dataset_config,
                **self.model_config, 
                **{k: v for k, v in self.training_config.items() if k != 'resume_from_checkpoint'}
            }
        )
        
        # Setup callbacks
        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            WandbRunIdInCheckpoint(),
        ]
        # A patience of None disables early stopping, e.g. to continue a run to a fixed epoch.
        patience = self.training_config.get('early_stopping_patience', 20)
        if patience is not None:
            callbacks.append(EarlyStopping(
                monitor=self.training_config['monitor_metric'],
                patience=patience,
                mode='min'
            ))
        
        # Create trainer
        self.trainer = Trainer(
            max_epochs=self.training_config['max_epochs'],
            max_time=self._session_max_time(),
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=self.training_config['gradient_clip_val'],
            log_every_n_steps=self.training_config['log_every_n_steps'],
            deterministic=True
        )
    
    def _resume_checkpoint(self):
        """The checkpoint being resumed from, loaded once on CPU; None for a fresh run."""
        resume = self.training_config.get('resume_from_checkpoint')
        if resume is None:
            return None
        if getattr(self, '_resumed', None) is None:
            self._resumed = torch.load(resume, map_location='cpu', weights_only=False)
        return self._resumed

    def _resume_wandb_run_id(self):
        """W&B run id of the training being resumed; None for a fresh run."""
        checkpoint = self._resume_checkpoint()
        if checkpoint is None:
            return None
        if 'wandb_run_id' in checkpoint:
            return checkpoint['wandb_run_id']
        # Older checkpoints: Lightning stores them under <project>/<run id>/checkpoints/.
        path = Path(self.training_config['resume_from_checkpoint']).resolve()
        if path.parent.name == 'checkpoints' and path.parent.parent.parent.name == self.project_name:
            return path.parent.parent.name
        raise ValueError(f'Cannot determine the W&B run of {path}; resume from the checkpoint '
                         f'in its original <project>/<run id>/checkpoints/ directory')

    def _session_max_time(self):
        """Per-session time limit, extended by any training time a resumed checkpoint already used.

        Lightning restores the Timer from the checkpoint and enforces max_time on the total,
        so without this a resumed run would stop as soon as it started.
        """
        max_time = self.training_config.get('max_time')
        resume = self.training_config.get('resume_from_checkpoint')
        if max_time is None or resume is None:
            return max_time
        days, hours, minutes, seconds = (int(part) for part in max_time.split(':'))
        budget = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        callbacks = self._resume_checkpoint().get('callbacks', {})
        elapsed = sum(state.get('time_elapsed', {}).get('train', 0.0)
                      for key, state in callbacks.items() if key.startswith('Timer'))
        print(f'Resuming from {resume}: {elapsed / 3600:.2f} h of training already recorded; '
              f'this session may train for {budget}', flush=True)
        return budget + timedelta(seconds=elapsed)

    def run(self):
        """Main experiment execution pipeline."""
        print(f"Running {self.experiment_name} experiment on {self.dataset_config['dyn_sys_name']}")
        
        # Prepare all components
        self.prepare_datasets()
        self.prepare_model()
        
        # FIXED: Pass normalizers to model for denormalized plotting during training
        self.model.y_normalizer = self.train_dataset.y_normalizer
        self.model.x_normalizer = getattr(self.train_dataset, 'x_normalizer', None)
        
        self.prepare_trainer()
        # Record the effective settings in the run directory, independent of W&B.
        Path('run_config.yaml').write_text(yaml.safe_dump({
            'dataset': self.dataset_config, 'model': self.model_config,
            'training': {k: v for k, v in self.training_config.items() if k != 'resume_from_checkpoint'}}))

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.training_config['batch_size'],
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False
        )
        
        # Train the model, continuing optimizer, scheduler and epoch count when resuming.
        self.trainer.fit(self.model, train_loader, val_loader,
                         ckpt_path=self.training_config.get('resume_from_checkpoint'))
        
        # Test the model
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False
        )
        self.trainer.test(self.model, test_loader)
        
        # ENHANCEMENT: Create comprehensive final plots with denormalized data
        print("🎨 Generating final prediction plots...")
        self.model.plot_final_predictions(test_loader, n_examples=4)
        
        print(f"Experiment finished.")
        return self.trainer, self.model
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment configuration."""
        return {
            'project_name': self.project_name,
            'experiment_name': self.experiment_name,
            'dataset_config': self.dataset_config,
            'model_config': self.model_config,
            'training_config': self.training_config
        } 
