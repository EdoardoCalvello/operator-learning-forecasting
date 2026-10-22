#!/usr/bin/env python3
"""Evaluate every trained L63 smoothing checkpoint at each observation noise level.

Checkpoints are discovered in the run directories under --runs-dir (default
$TNO_OUTPUT_ROOT/l63_smoothing, where scripts/L63/smoothing/run.sh writes them); each
run's training noise level is read from its recorded W&B config. Outputs (plots, per-case
results, summary.csv) are written to the current directory.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.experiments.evaluation import EvaluationExperiment
# Full float32 matmuls: the container enables TF32, whose rounding shifts these small errors
# by a few percent and makes them depend on batch size.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

EVAL_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1, 1.0]


def recorded_config(job_dir):
    """Flat {key: value} of a training run's settings, from run_config.yaml or else its W&B config."""
    path = job_dir / 'run_config.yaml'
    if path.exists():
        sections = yaml.safe_load(path.read_text())
        return {k: v for section in sections.values() for k, v in section.items()}
    path = next(job_dir.glob('wandb/*run-*/files/config.yaml'), None)
    if path is None:
        raise FileNotFoundError(f'{job_dir}: no run_config.yaml or W&B config.yaml')
    raw = yaml.safe_load(path.read_text())
    return {k: v['value'] for k, v in raw.items() if isinstance(v, dict) and 'value' in v}


def discover(root, model_config):
    runs = []
    for job_dir in sorted(root.iterdir()):
        checkpoints = sorted(job_dir.glob('*/*/checkpoints/*.ckpt'))
        if not checkpoints:
            continue
        recorded = recorded_config(job_dir)
        mismatched = {k: (recorded.get(k), v) for k, v in model_config.items()
                      if k in recorded and recorded[k] != v}
        if mismatched:
            raise ValueError(f'{job_dir}: architecture differs from config: {mismatched}')
        runs.append(dict(job=job_dir.name, checkpoint=checkpoints[-1],
                         train_noise=float(recorded.get('obs_noise_std') or 0.0)))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), '../configs/l63_smoothing.yaml'))
    parser.add_argument('--runs-dir', type=Path,
                        default=Path(os.environ.get('TNO_OUTPUT_ROOT', Path(__file__).resolve().parents[1] / 'outputs')) / 'l63_smoothing',
                        help='Directory containing one subdirectory per L63 smoothing training run')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    out = Path.cwd()
    runs = discover(args.runs_dir.resolve(), config['model'])
    print(f'Found {len(runs)} L63 smoothing checkpoints:', flush=True)
    for run in runs:
        print(f"  job {run['job']}  train noise {run['train_noise']}  {run['checkpoint']}", flush=True)

    rows = []
    for run in runs:
        for eval_noise in EVAL_NOISE_LEVELS:
            case = out / f"train_noise_{run['train_noise']}_job{run['job']}" / f'eval_noise_{eval_noise}'
            case.mkdir(parents=True, exist_ok=True)
            os.chdir(case)  # evaluation plots are written relative to the cwd
            print(f"\n### job {run['job']} (train noise {run['train_noise']}) at eval noise {eval_noise}", flush=True)
            experiment = EvaluationExperiment(
                checkpoint_path=str(run['checkpoint']),
                system_name='L63',
                project_name=config['experiment']['project_name'],
                experiment_name='eval_l63_smoothing',
                seed=config['evaluation']['seed'],
                dataset_config=dict(config['dataset'], obs_noise_std=eval_noise),
                model_config=config['model'],
                training_config=config['training'],
            )
            results = experiment.evaluate(save_plots=True, n_examples=config['evaluation']['n_examples'],
                                          compute_baseline=False)
            # The evaluation loads weights non-strictly; refuse silently partial loads.
            saved = torch.load(run['checkpoint'], map_location='cpu')['state_dict'].keys()
            if set(saved) != set(experiment.model.state_dict().keys()):
                raise RuntimeError(f"{run['checkpoint']}: checkpoint keys do not match the model")
            # Relative L2 in physical units, alongside the normalized metric used in training.
            normalizer = experiment.test_dataset.y_normalizer
            pred = normalizer.decode(torch.from_numpy(results['predictions'][:, 0]))
            true = normalizer.decode(torch.from_numpy(results['ground_truths'][:, 0]))
            physical = (torch.linalg.vector_norm(pred - true, dim=(-2, -1))
                        / torch.linalg.vector_norm(true, dim=(-2, -1))).numpy()
            row = dict(job=run['job'], train_noise=run['train_noise'], eval_noise=eval_noise,
                       rel_l2_normalized_mean=float(results['avg_error']),
                       rel_l2_normalized_median=float(results['median_error']),
                       rel_l2_physical_mean=float(np.mean(physical)),
                       rel_l2_physical_median=float(np.median(physical)),
                       checkpoint=str(run['checkpoint']))
            (case / 'results.json').write_text(json.dumps(row, indent=2))
            rows.append(row)

    os.chdir(out)
    with open('summary.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print('\nMean relative L2 on the eval split (normalized / physical units); rows = training noise')
    print('train \\ eval  ' + ''.join(f'{e:>18}' for e in EVAL_NOISE_LEVELS))
    for run in runs:
        cells = {r['eval_noise']: r for r in rows if r['job'] == run['job']}
        print(f"{run['train_noise']:<13} " + ''.join(
            f"{cells[e]['rel_l2_normalized_mean']:>9.4f}/{cells[e]['rel_l2_physical_mean']:<8.4f}"
            for e in EVAL_NOISE_LEVELS))
    print(f'\nSummary: {out / "summary.csv"}')


if __name__ == '__main__':
    main()
