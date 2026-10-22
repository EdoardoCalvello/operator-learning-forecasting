#!/usr/bin/env python3
"""
Forecasting Experiment
Handles temporal prediction tasks: predict future system states from historical data.
"""

import argparse
import yaml
from .base import BaseExperiment


class ForecastingExperiment(BaseExperiment):
    """Experiment for forecasting (temporal prediction) tasks."""
    
    def __init__(self, system_name, **kwargs):
        """Initialize forecasting experiment for specified system."""
        self.system_name = system_name.upper()
        self.task_type = 'forecasting'
        super().__init__(**kwargs)
    
    def _configure_experiment_specifics(self):
        """Configure dataset parameters specific to forecasting experiments."""
        self.setup_system_config()
    
    def setup_system_config(self):
        """Configure input/output indices based on system."""
        if self.system_name == 'L63':
            # L63: Use indices from config file (allow custom input/output configurations)
            if 'input_inds' not in self.dataset_config:
                self.dataset_config['input_inds'] = [0, 1, 2]  # Default: full state
            if 'output_inds' not in self.dataset_config:
                self.dataset_config['output_inds'] = [0, 1, 2]  # Default: full state
            
        elif self.system_name == 'L96':
            # L96: use indices from config (sparse forecasting)
            if 'input_inds' not in self.dataset_config:
                self.dataset_config['input_inds'] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
            if 'output_inds' not in self.dataset_config:
                self.dataset_config['output_inds'] = self.dataset_config['input_inds']  # Same for forecasting
            
        elif self.system_name == 'KSE':
            # KSE: use indices from config (sparse forecasting)
            if 'input_inds' not in self.dataset_config:
                self.dataset_config['input_inds'] = list(range(0, 128, 8))
            if 'output_inds' not in self.dataset_config:
                self.dataset_config['output_inds'] = self.dataset_config['input_inds']  # Same for forecasting
            
        else:
            raise ValueError(f"Unknown system: {self.system_name}")
    
    def print_experiment_info(self):
        """Print experiment configuration."""
        prediction_horizon = self.dataset_config.get('prediction_horizon', 0.0)
        print(f"{self.system_name} FORECASTING EXPERIMENT")
        print("=" * 50)
        print(f"System: {self.dataset_config['dyn_sys_name']}")
        print(f"State dimension: {len(self.dataset_config['input_inds'])}")
        print(f"Prediction horizon: {prediction_horizon}s")
        print(f"Training trajectories: {self.dataset_config.get('n_trajectories_train', self.dataset_config.get('size'))}")
        print(f"Model: {self.model_config['d_model']}d, {self.model_config['num_layers']} layers")
        print(f"Training: {self.training_config['max_epochs']} epochs")
        print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Forecasting Experiment')
    
    parser.add_argument('--system', type=str, required=True,
                       choices=['L63', 'L96', 'KSE'],
                       help='Dynamical system (L63, L96, or KSE)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--project_name', type=str, default=None,
                       help='WandB project name (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Max epochs (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main experiment execution."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.seed:
        config['experiment']['seed'] = args.seed
    if args.project_name:
        config['experiment']['project_name'] = args.project_name
    
    # Create and run forecasting experiment
    experiment = ForecastingExperiment(
        system_name=args.system,
        project_name=config['experiment']['project_name'],
        experiment_name=f"{args.system.lower()}_forecasting",
        seed=config['experiment'].get('seed', 42),
        dataset_config=config['dataset'],
        model_config=config['model'],
        training_config=config['training']
    )
    
    experiment.setup_system_config()
    experiment.print_experiment_info()
    experiment.run()
    
    print("EXPERIMENT COMPLETE")
    print("Check WandB for detailed results")


if __name__ == "__main__":
    main()