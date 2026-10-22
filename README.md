# TNO Data Assimilation

Code and configurations for the experiments in our paper on Transformer Neural Operators (TNO)
for data assimilation in chaotic dynamical systems. Experiments cover **smoothing**
(reconstructing unobserved state components from partial observations) and **forecasting**
(predicting future states) on the Lorenz-63 (L63), Lorenz-96 (L96) and Kuramoto–Sivashinsky (KSE)
systems.

## Installation

Python 3.10+ and a CUDA GPU are recommended. The paper experiments ran with PyTorch 2.8.

```bash
pip install -r requirements.txt
```

Training logs to [Weights & Biases](https://wandb.ai). Log in with `wandb login`, or set
`WANDB_MODE=offline` (or `disabled`) to run without an account.

## Repository layout

```
configs/                 One YAML file per paper experiment (dataset, model, training, evaluation)
scripts/
  generate_data.py       Generate and save the train/val/test/eval trajectories
  L63/ L96/ KSE/         Training entry points (<system>/<task>/<system>_<task>.py) and run.sh launchers
  launch.sh              Runs an entry point in its own output directory (used by every run.sh)
  evaluate_model.py              Errors and plots over one prediction horizon
  evaluate_model_statistics.py   Long forecast rollouts: errors and trajectory statistics
  evaluate_forecast_baselines.py Comparison with constant and polynomial-extrapolation forecasts
  evaluate_l63_smoothing.py      L63 smoothing models evaluated at several observation noise levels
src/
  datasets/              Dynamical systems, ODE/PDE solvers and saved-dataset loading
  models/tno/            TNO model (PyTorch) and its PyTorch Lightning module
  experiments/           Training and evaluation pipelines
  utils/                 Metrics, plotting and forecasting baselines
tests/                   Small CPU tests of data generation and loading
```

## Experiments

| Task | System | Config | Notes |
| --- | --- | --- | --- |
| Smoothing | L63 | `l63_smoothing.yaml` | Observe *x*, reconstruct *y*, *z*; trained at several noise levels (see below) |
| Smoothing | L96 | `l96_smoothing.yaml` | 40 variables; observe 30, reconstruct 10 |
| Smoothing | KSE | `kse_smoothing.yaml` | 128 grid points |
| Forecasting | L63 | `l63_forecasting.yaml` | Observe and forecast *x*; horizon 2.0, cross-attention decoder, dropout 0.1 |
| Forecasting | L96 | `l96_forecasting.yaml` | Horizon 0.2, cross-attention decoder |
| Forecasting | KSE | `kse_forecasting.yaml` | Horizon 5, cross-attention decoder |

## Reproducing the experiments

All commands are run from the repository root.

### 1. Generate the datasets

Each experiment trains and evaluates on trajectories that are generated once and saved,
so every run sees exactly the same data:

```bash
python scripts/generate_data.py                                   # all configs
python scripts/generate_data.py --configs configs/l63_smoothing.yaml   # a subset
```

Data is written to `$TNO_DATA_DIR` (default `./data`), under the `saved_data_dir` given in each
config, e.g. `data/Lorenz63/forecasting/{train,val,test,eval}/`. Completed datasets are skipped,
and a dataset generated with different settings is never overwritten.

Each split contains raw `float32` arrays (`x.npy`, `y.npy`, `times_x.npy`, `times_y.npy`,
`initial_conditions.npy`), the source config, and a `manifest.json` that records the settings,
seed, initial-condition hash, array shapes, solver and normalization statistics.

- **Seeds.** Train, validation and test initial conditions are drawn sequentially from
  `experiment.seed`. The evaluation split uses `evaluation.seed`, continuing the same random stream
  when the two seeds are equal, so the splits never overlap.
- **Evaluation split.** The `eval` split is a held-out set of `n_trajectories_test` trajectories.
  For forecasting it extends over `dataset.compositions` consecutive horizons for the rollout
  evaluation; training, validation and test use one horizon.
- **Normalization.** Every split is normalized with the training split's statistics.
- **Solver.** Trajectories are integrated with `torchdiffeq` (dopri5, rtol 1e-7, atol 1e-9) in
  batches of `data_generation.batch_size`. Results can differ slightly with batch size or hardware,
  so both are recorded in the manifest.

Generating the large forecasting datasets requires a GPU (`--device cuda`, the default).

### 2. Train

Each `run.sh` launcher takes an optional config (default: the system's main config) followed by extra
arguments for the training script:

```bash
bash scripts/L63/smoothing/run.sh
bash scripts/L63/forecasting/run.sh
bash scripts/KSE/smoothing/run.sh configs/kse_smoothing.yaml
```

The same launchers can be submitted to Slurm (add your account, partition and GPU constraints):

```bash
sbatch scripts/L96/forecasting/run.sh
```

Each run writes its checkpoints, W&B files, plots and a `run_config.yaml` to
`$TNO_OUTPUT_ROOT/<entry point>/<run id>/` (default `./outputs/`). The run id is the Slurm job
id, or a timestamp when run locally.

**L63 smoothing with observation noise.** The paper trains one L63 smoothing model per
observation noise level, e.g.

```bash
for noise in 0.0 0.001 0.01 0.1 1.0; do
    bash scripts/L63/smoothing/run.sh configs/l63_smoothing.yaml --obs_noise_std $noise
done
```

Noise is added to the observations at load time, reproducibly per split, while the targets
stay clean.

**Training limits.** Some configs set `training.max_time`, the wall-clock training limit used
for the paper runs. The L63 training scripts can continue a run from a checkpoint with
`--resume /absolute/path/to/checkpoint.ckpt`. The L63 forecasting model was trained over several
such sessions, each limited with `--max-time DD:HH:MM:SS`.

### 3. Evaluate

Evaluation scripts can be run directly or through `scripts/launch.sh` (which gives each evaluation its
own output directory). Use absolute checkpoint paths with `launch.sh`.

```bash
# Errors and example plots over one horizon (smoothing and forecasting)
python scripts/evaluate_model.py --config configs/l96_smoothing.yaml --checkpoint path/to/model.ckpt

# Forecast rollouts over `dataset.compositions` horizons: errors and trajectory statistics
python scripts/evaluate_model_statistics.py --config configs/l63_forecasting.yaml --checkpoint path/to/model.ckpt

# Constant and polynomial-extrapolation baselines. Each checkpoint may be a .ckpt file or a training run
# directory (its newest checkpoint is used); systems without one are scored on the baselines only.
python scripts/evaluate_forecast_baselines.py \
    --checkpoints L63=outputs/l63_forecasting/<run id> L96=outputs/l96_forecasting/<run id> KSE=outputs/kse_forecasting/<run id>

# Every L63 smoothing run in outputs/l63_smoothing/, evaluated at each observation noise level
python scripts/evaluate_l63_smoothing.py
```

Results (tables, `summary.csv`, `baseline_results.csv`, plots) are written to the current directory.

## Tests

```bash
python -m unittest discover -s tests
```
