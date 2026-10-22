#!/usr/bin/env python3
"""L63 Smoothing Script"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.experiments.smoothing import SmoothingExperiment
import yaml
import argparse


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='L63 Smoothing Experiment')
    parser.add_argument('--config', type=str, 
                       default=os.path.join(os.path.dirname(__file__), '../../../configs/l63_smoothing.yaml'),
                       help='Path to config file')
    parser.add_argument('--obs_noise_std', type=float, default=None,
                       help='Override dataset.obs_noise_std from the config')
    parser.add_argument('--resume', type=str,
                       help='Checkpoint to continue training from (model, optimizer, scheduler and epoch)')
    parser.add_argument('--max-epochs', type=int,
                       help='Total number of epochs to train to, overriding the config')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Train until --max-epochs or the time limit instead of stopping early')
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.obs_noise_std is not None:
        config['dataset']['obs_noise_std'] = args.obs_noise_std
    if args.resume:
        config['training']['resume_from_checkpoint'] = args.resume
    if args.max_epochs is not None:
        config['training']['max_epochs'] = args.max_epochs
    if args.no_early_stopping:
        config['training']['early_stopping_patience'] = None
    
    experiment = SmoothingExperiment(
        system_name='L63',
        project_name=config['experiment']['project_name'],
        experiment_name="l63_smoothing",
        seed=config['experiment']['seed'],
        dataset_config=config['dataset'],
        model_config=config['model'],
        training_config=config['training']
    )
    
    experiment.run()


if __name__ == "__main__":
    main()