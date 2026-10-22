#!/usr/bin/env python3
"""L63 Forecasting Script"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.experiments.forecasting import ForecastingExperiment
import yaml
import argparse


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='L63 Forecasting Experiment')
    parser.add_argument('--config', type=str, 
                       default=os.path.join(os.path.dirname(__file__), '../../../configs/l63_forecasting.yaml'),
                       help='Path to config file')
    parser.add_argument('--resume', type=str,
                       help='Checkpoint to continue training from (model, optimizer, scheduler and epoch)')
    parser.add_argument('--max-time', type=str,
                       help='Per-session training time limit DD:HH:MM:SS, overriding the config')
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.resume:
        config['training']['resume_from_checkpoint'] = args.resume
    if args.max_time:
        config['training']['max_time'] = args.max_time
    
    experiment = ForecastingExperiment(
        system_name='L63',
        project_name=config['experiment']['project_name'],
        experiment_name="l63_forecasting",
        seed=config['experiment']['seed'],
        dataset_config=config['dataset'],
        model_config=config['model'],
        training_config=config['training']
    )
    
    experiment.run()


if __name__ == "__main__":
    main()