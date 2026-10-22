#!/usr/bin/env python3
"""Compare trained forecasting models against constant and polynomial-extrapolation baselines.

For each system the polynomial degree and fit window are chosen on the validation
split (median relative L2); the chosen baseline, the constant baseline and the
model are then scored on the held-out eval split over one prediction horizon.
All errors are relative L2 in physical units against the true trajectories.
Results are written to the current directory.
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
from src.datasets.saved import load_saved_split
from src.experiments.evaluation import EvaluationExperiment
from src.utils.baselines import polynomial_extrapolation, relative_l2_per_trajectory, trajectory_statistics_errors

REPO = Path(__file__).resolve().parents[1]
# system: config of the trained model
SYSTEMS = {
    'L63': 'configs/l63_forecasting.yaml',
    'L96': 'configs/l96_forecasting.yaml',
    'KSE': 'configs/kse_forecasting.yaml',
}
DEGREES = [0, 1, 2, 3, 4, 5, 6]
WINDOWS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]


def candidates(n_in):
    return [(p, m) for p in DEGREES for m in WINDOWS if p + 1 <= m <= n_in]


def model_predictions(config, checkpoint, test):
    experiment = EvaluationExperiment(
        checkpoint_path=str(checkpoint), system_name=config['evaluation']['system_name'],
        project_name=config['experiment']['project_name'], experiment_name='forecast_baselines',
        seed=config['evaluation']['seed'], dataset_config=config['dataset'],
        model_config=config['model'], training_config=config['training'])
    model = experiment.load_model_from_checkpoint()
    # The evaluation loads weights non-strictly; refuse silently partial loads.
    saved = torch.load(checkpoint, map_location='cpu')['state_dict'].keys()
    if set(saved) != set(model.state_dict().keys()):
        raise RuntimeError(f'{checkpoint}: checkpoint keys do not match the model')
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
    outputs = []
    with torch.no_grad():
        for x, y, coords_x, coords_y in loader:
            outputs.append(model(x.to(device), y.to(device), coords_x.to(device), coords_y.to(device)).cpu())
    return test.y_normalizer.decode(torch.cat(outputs)).numpy()


def summarize(errors):
    return dict(median=float(np.median(errors)), mean=float(np.mean(errors)))


def latest_checkpoint(path):
    """A checkpoint file, or the newest checkpoint under a training run directory."""
    if path.is_file():
        return path
    checkpoints = sorted(path.glob('**/checkpoints/*.ckpt'), key=lambda c: c.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints under {path}')
    return checkpoints[-1]


def evaluate_system(name, config_path, checkpoint=None):
    config = yaml.safe_load((REPO / config_path).read_text())
    data = config['dataset']
    with_model = checkpoint is not None
    val = load_saved_split(data, 'val', config['experiment']['seed'])
    test = load_saved_split(data, 'eval', config['evaluation']['seed'], compositions=1)
    times_x, times_y = val.times_x[:, 0].numpy(), val.times_y[:, 0].numpy()
    if not (np.allclose(times_x, test.times_x[:, 0].numpy()) and np.allclose(times_y, test.times_y[:, 0].numpy())):
        raise ValueError(f'{name}: val and eval time grids differ')
    val_x, val_y = val.raw_x.numpy(), val.raw_y.numpy()
    print(f'\n=== {name}: {config_path}, horizon {data["prediction_horizon"]} '
          f'({len(times_y)} steps), checkpoint {checkpoint}', flush=True)

    tuning = []
    for degree, window in candidates(len(times_x)):
        errors = relative_l2_per_trajectory(polynomial_extrapolation(val_x, times_x, times_y, degree, window), val_y)
        tuning.append(dict(system=name, degree=degree, window=window, **{f'val_{k}': v for k, v in summarize(errors).items()}))
        print(f'  val  degree {degree} window {window:>2}: median {tuning[-1]["val_median"]:.4g}', flush=True)
    best = min(tuning, key=lambda row: row['val_median'])

    test_x, test_y = test.raw_x.numpy(), test.raw_y.numpy()
    forecasts = {
        'constant': polynomial_extrapolation(test_x, times_x, times_y, 0, 1),
        'polynomial': polynomial_extrapolation(test_x, times_x, times_y, best['degree'], best['window']),
        # Without a model the model columns are NaN, so baselines can be scored before training ends.
        'model': model_predictions(config, checkpoint, test) if with_model else np.full(test_y.shape, np.nan),
    }
    result = dict(system=name, horizon=data['prediction_horizon'], horizon_steps=len(times_y),
                  n_eval=len(test_y), poly_degree=best['degree'], poly_window=best['window'],
                  checkpoint=str(checkpoint) if checkpoint else '')
    for label, forecast in forecasts.items():
        for stat, value in summarize(relative_l2_per_trajectory(forecast, test_y)).items():
            result[f'{label}_{stat}'] = value
        result[f'{label}_time_mean_error'], result[f'{label}_time_var_error'] = trajectory_statistics_errors(forecast, test_y)
    # Relative improvement of the model over each baseline, as reported for the constant baseline.
    for baseline in ('constant', 'polynomial'):
        for stat in ('mean', 'median'):
            result[f'improvement_vs_{baseline}_{stat}_pct'] = 100 * (1 - result[f'model_{stat}'] / result[f'{baseline}_{stat}'])
    print(f'  eval median rel L2: model {result["model_median"]:.4g}, constant {result["constant_median"]:.4g}, '
          f'polynomial (degree {best["degree"]}, window {best["window"]}) {result["polynomial_median"]:.4g}', flush=True)
    return result, tuning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Unused; accepted for scripts/launch.sh')
    parser.add_argument('--systems', nargs='+', default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument('--checkpoints', nargs='+', default=[], metavar='SYSTEM=PATH',
                        help='Trained model per system: a checkpoint file or a training run directory '
                             '(its newest checkpoint is used). Systems without one are scored on the baselines only.')
    args = parser.parse_args()
    checkpoints = {}
    for item in args.checkpoints:
        name, sep, path = item.partition('=')
        if not sep or name not in SYSTEMS:
            parser.error(f'--checkpoints expects SYSTEM=PATH with SYSTEM in {list(SYSTEMS)}, got {item!r}')
        checkpoints[name] = latest_checkpoint(Path(path).resolve())
    results, tuning = [], []
    for name in args.systems:
        result, rows = evaluate_system(name, SYSTEMS[name], checkpoint=checkpoints.get(name))
        results.append(result)
        tuning.extend(rows)
    for filename, rows in (('baseline_results.csv', results), ('baseline_tuning.csv', tuning)):
        with open(filename, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    Path('baseline_results.json').write_text(json.dumps(results, indent=2))

    print('\nRelative L2 on the eval split over one horizon (median / mean, physical units)')
    print(f'{"system":<9}{"model":>22}{"constant":>22}{"polynomial":>22}  chosen (degree, window)')
    for r in results:
        cells = ''.join(f'{r[k + "_median"]:>11.4g} /{r[k + "_mean"]:>9.4g}' for k in ('model', 'constant', 'polynomial'))
        print(f'{r["system"]:<9}{cells}  ({r["poly_degree"]}, {r["poly_window"]})')
    print('\nModel improvement in relative L2 (mean / median) and trajectory statistics errors (physical units)')
    for r in results:
        print(f'{r["system"]}:')
        for baseline in ('constant', 'polynomial'):
            print(f'  vs {baseline:<10}: {r[f"improvement_vs_{baseline}_mean_pct"]:6.2f}% / {r[f"improvement_vs_{baseline}_median_pct"]:6.2f}%')
        for label in ('model', 'constant', 'polynomial'):
            print(f'  {label:<10} time-mean error {r[f"{label}_time_mean_error"]:.6f}   time-variance error {r[f"{label}_time_var_error"]:.6f}')
    print(f'\nResults: {Path.cwd()}')


if __name__ == '__main__':
    main()
