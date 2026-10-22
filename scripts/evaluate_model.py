#!/usr/bin/env python3
"""
Evaluate a trained model on the evaluation split over one prediction horizon.

This script loads a trained model checkpoint and evaluates it on the held-out
evaluation split, computing metrics and generating visualization plots.

Usage:
    python scripts/evaluate_model.py --config configs/l63_smoothing.yaml --checkpoint path/to/checkpoint.ckpt

    # Using a W&B run's recorded config instead of a config file
    python scripts/evaluate_model.py --checkpoint path/to/checkpoint.ckpt --wandb-run-id abc123 --wandb-project L63_Forecasting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import yaml
from pathlib import Path

from src.experiments.evaluation import EvaluationExperiment


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():

    parser = argparse.ArgumentParser(
        description='Evaluate a trained model on test data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments (checkpoint can be in config file)
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (.ckpt file). If not provided, uses path from config file.'
    )
    
    # Config source (either config file or WandB)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config',
        type=str,
        help='Path to evaluation YAML config file'
    )
    config_group.add_argument(
        '--wandb-run-id',
        type=str,
        help='WandB run ID to load config from'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        help='WandB project name (required if using --wandb-run-id)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--system-name',
        type=str,
        default='L63',
        help='Name of dynamical system (default: L63)'
    )
    parser.add_argument(
        '--n-examples',
        type=int,
        default=5,
        help='Number of example plots to generate (default: 5)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip computing baseline errors (useful for smoothing experiments)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for evaluation (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.config:
        # When using config, checkpoint can come from either args or config file
        config = load_config(args.config)
        eval_config = config.get('evaluation', {})
        checkpoint_path = args.checkpoint or eval_config.get('checkpoint_path')
        
        if not checkpoint_path:
            print("Error: Checkpoint path must be provided either via --checkpoint or in the config file")
            sys.exit(1)
        
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)
    else:
        # When using WandB, checkpoint must be provided via args
        if not args.checkpoint:
            print("Error: --checkpoint is required when using --wandb-run-id")
            sys.exit(1)
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)
    
    if args.wandb_run_id and not args.wandb_project:
        print("Error: --wandb-project is required when using --wandb-run-id")
        sys.exit(1)
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Get evaluation-specific settings
        eval_config = config.get('evaluation', {})
        # The command-line checkpoint takes precedence over the config's.
        checkpoint = args.checkpoint or eval_config.get('checkpoint_path')
        system_name = eval_config.get('system_name', args.system_name)
        n_examples = eval_config.get('n_examples', args.n_examples)
        save_plots = eval_config.get('save_plots', not args.no_plots)
        compute_baseline = eval_config.get('compute_baseline', not args.no_baseline)
        seed = eval_config.get('seed', args.seed)
        
        experiment = EvaluationExperiment(
            checkpoint_path=checkpoint,
            system_name=system_name,
            project_name=config['experiment']['project_name'],
            experiment_name=f"eval_{config['experiment']['project_name']}",
            seed=seed,
            dataset_config=config['dataset'],
            model_config=config['model'],
            training_config=config['training']
        )
        
        # Store eval settings for later use
        experiment.n_examples = n_examples
        experiment.save_plots = save_plots
        experiment.compute_baseline = compute_baseline
    else:
        print(f"Loading configuration from WandB: {args.wandb_project}/{args.wandb_run_id}")
        experiment = EvaluationExperiment(
            checkpoint_path=args.checkpoint,
            wandb_run_id=args.wandb_run_id,
            wandb_project=args.wandb_project,
            system_name=args.system_name,
            project_name=args.wandb_project,
            experiment_name=f"eval_{args.wandb_project}",
            seed=args.seed
        )
        
        # Store eval settings for later use
        experiment.n_examples = args.n_examples
        experiment.save_plots = not args.no_plots
        experiment.compute_baseline = not args.no_baseline
    
    # Run evaluation
    print("\n" + "="*60)
    print(f"EVALUATING MODEL: {experiment.system_name}")
    print(f"Checkpoint: {experiment.checkpoint_path}")
    print("="*60 + "\n")
    
    results = experiment.evaluate(
        save_plots=experiment.save_plots,
        n_examples=experiment.n_examples,
        compute_baseline=experiment.compute_baseline
    )
    
    print("\n✓ Evaluation complete!")
    
    # Optionally save results to file
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    checkpoint_name = Path(experiment.checkpoint_path).stem
    results_file = results_dir / f"{experiment.system_name}_{checkpoint_name}_results.txt"
    
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"EVALUATION RESULTS: {experiment.system_name}\n")
        f.write(f"Checkpoint: {experiment.checkpoint_path}\n")
        f.write("="*60 + "\n\n")
        f.write("Model Performance:\n")
        f.write(f"  Average Relative L2 Error:  {results['avg_error']:.6f}\n")
        f.write(f"  Median Relative L2 Error:   {results['median_error']:.6f}\n")
        f.write(f"  Std Dev Relative L2 Error:  {results['std_error']:.6f}\n")
        f.write(f"  Min Relative L2 Error:      {results['min_error']:.6f}\n")
        f.write(f"  Max Relative L2 Error:      {results['max_error']:.6f}\n")
        
        if results['avg_baseline_error'] is not None:
            f.write(f"\nBaseline Performance (constant from last input):\n")
            f.write(f"  Average Relative L2 Error:  {results['avg_baseline_error']:.6f}\n")
            f.write(f"  Median Relative L2 Error:   {results['median_baseline_error']:.6f}\n")
            f.write(f"\nImprovement over Baseline:   {results['improvement_over_baseline']:.2f}%\n")
            f.write(f"\nTrajectory Statistics Comparison:\n")
            f.write(f"  Mean Error (Learned):       {results['mean_error_learned']:.6f}\n")
            f.write(f"  Mean Error (Baseline):      {results['mean_error_baseline']:.6f}\n")
            f.write(f"  Variance Error (Learned):   {results['var_error_learned']:.6f}\n")
            f.write(f"  Variance Error (Baseline):  {results['var_error_baseline']:.6f}\n")
        else:
            f.write(f"\nTrajectory Statistics:\n")
            f.write(f"  Mean Error (Learned):       {results['mean_error_learned']:.6f}\n")
            f.write(f"  Variance Error (Learned):   {results['var_error_learned']:.6f}\n")
        
        f.write("="*60 + "\n")
    
    print(f"✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
