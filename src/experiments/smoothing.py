#!/usr/bin/env python3
"""
Smoothing Experiment
Handles data assimilation tasks: predict unobserved variables at the same time point.
"""

import argparse
import yaml
from .base import BaseExperiment


class SmoothingExperiment(BaseExperiment):
    """Experiment for smoothing (same-time prediction) tasks."""
    
    def __init__(self, system_name, **kwargs):
        """Initialize smoothing experiment for specified system."""
        self.system_name = system_name.upper()
        self.task_type = 'smoothing'
        super().__init__(**kwargs)
    
    def _configure_experiment_specifics(self):
        """Configure dataset parameters specific to smoothing experiments."""
        self.setup_system_config()
    
    def setup_system_config(self):
        """Configure input/output indices based on system."""
        if self.system_name == 'L63':
            # L63: defaults observe x, predict y,z (overridable from config)
            if 'input_inds' not in self.dataset_config:
                self.dataset_config['input_inds'] = [0]
            if 'output_inds' not in self.dataset_config:
                self.dataset_config['output_inds'] = [1, 2]
            self.dataset_config['prediction_horizon'] = 0.0
            
        elif self.system_name == 'L96':
            # L96: use indices from config (every 4th dimension)
            if 'input_inds' not in self.dataset_config:
                self.dataset_config['input_inds'] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
            if 'output_inds' not in self.dataset_config:
                observed = set(self.dataset_config['input_inds'])
                self.dataset_config['output_inds'] = [i for i in range(40) if i not in observed]
            self.dataset_config['prediction_horizon'] = 0.0
            
        elif self.system_name == 'KSE':
            # KSE: use indices from config (every 8th dimension)
            if 'input_inds' not in self.dataset_config:
                self.dataset_config['input_inds'] = list(range(0, 128, 8))
            if 'output_inds' not in self.dataset_config:
                observed = set(self.dataset_config['input_inds'])
                self.dataset_config['output_inds'] = [i for i in range(128) if i not in observed]
            self.dataset_config['prediction_horizon'] = 0.0
            
        else:
            raise ValueError(f"Unknown system: {self.system_name}")
    
    def print_experiment_info(self):
        """Print experiment configuration."""
        print(f"{self.system_name} SMOOTHING EXPERIMENT")
        print("=" * 50)
        print(f"System: {self.dataset_config['dyn_sys_name']}")
        print(f"Input dims: {len(self.dataset_config['input_inds'])}")
        print(f"Output dims: {len(self.dataset_config['output_inds'])}")
        print(f"Training trajectories: {self.dataset_config.get('n_trajectories_train', self.dataset_config.get('size'))}")
        print(f"Model: {self.model_config['d_model']}d, {self.model_config['num_layers']} layers")
        print(f"Training: {self.training_config['max_epochs']} epochs")
        print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Smoothing Experiment')
    
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
    
    # Create and run smoothing experiment
    experiment = SmoothingExperiment(
        system_name=args.system,
        project_name=config['experiment']['project_name'],
        experiment_name=f"{args.system.lower()}_smoothing",
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