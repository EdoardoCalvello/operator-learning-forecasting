#!/usr/bin/env python3
"""KSE Forecasting Script"""

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
    parser = argparse.ArgumentParser(description='KSE Forecasting Experiment')
    parser.add_argument('--config', type=str, 
                       default=os.path.join(os.path.dirname(__file__), '../../../configs/kse_forecasting.yaml'),
                       help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    experiment = ForecastingExperiment(
        system_name='KSE',
        project_name=config['experiment']['project_name'],
        experiment_name=config['experiment'].get('name', 'kse_forecasting'),
        seed=config['experiment']['seed'],
        dataset_config=config['dataset'],
        model_config=config['model'],
        training_config=config['training']
    )
    
    experiment.run()


if __name__ == "__main__":
    main()