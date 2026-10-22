#!/usr/bin/env python3
"""Generate and save the config-driven L63/L96/KSE train/val/test/eval trajectories.

Uses the existing adaptive torchdiffeq solver and dataset slicing. Raw arrays
are saved before normalization; each split retains the existing per-split
normalization convention, with statistics saved for exact reuse.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml

from src.datasets.dynamical import Lorenz63, Lorenz63Dataset, Lorenz96, Lorenz96Dataset, KuramotoSivashinsky, KSDataset
from src.datasets.transforms import UnitGaussianNormalizer
from src.datasets.saved import SavedDataset, data_path


class RawMixin:
    def __init__(self, initial_states, **kwargs):
        self.initial_states = initial_states
        super().__init__(**kwargs)

    def _load_system(self):
        system = super()._load_system()
        system.get_inits = lambda size: self.initial_states
        return system

    def _normalize_data(self):
        pass


class RawLorenz63Dataset(RawMixin, Lorenz63Dataset):
    pass


class RawKSDataset(RawMixin, KSDataset):
    pass


class RawLorenz96Dataset(RawMixin, Lorenz96Dataset):
    pass


SYSTEMS = {'Lorenz63': (Lorenz63, RawLorenz63Dataset),
           'Lorenz96': (Lorenz96, RawLorenz96Dataset),
           'KuramotoSivashinsky': (KuramotoSivashinsky, RawKSDataset)}


def split_initial_states(config):
    """Use configured seeds, continuing the stream to prevent split overlap."""
    data = config['dataset']
    system = SYSTEMS[data['dyn_sys_name']][0](**data['params'])
    training_seed = int(config['experiment']['seed'])
    evaluation_seed = int(config['evaluation']['seed'])
    states = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(training_seed)
        for split in ('train', 'val', 'test'):
            states[split] = system.get_inits(data[f'n_trajectories_{split}'])
        # When both seeds match, continue past train/val/test instead of
        # resetting the stream and reusing training initial conditions.
        if evaluation_seed != training_seed:
            torch.manual_seed(evaluation_seed)
        states['eval'] = system.get_inits(data['n_trajectories_test'])
    combined = torch.cat(list(states.values()))
    if torch.unique(combined, dim=0).shape[0] != combined.shape[0]:
        raise ValueError('Initial conditions overlap between splits')
    return states


def generate(config, source, task, split, initial, output, batch_size, device):
    effective = dict(config['dataset'])
    effective.pop('saved_data_dir', None)
    # Observation noise is added when loading, so clean arrays are saved for every noise level.
    effective.pop('obs_noise_std', None)
    effective.update(size=len(initial), test=split in ('test', 'eval'))
    effective['compositions'] = effective.get('compositions', 1) if split == 'eval' else 1
    seed = config['evaluation' if split == 'eval' else 'experiment']['seed']
    identity = {'dataset_config': effective, 'seed': seed, 'split': split,
                'batch_size': batch_size, 'device': device,
                'initial_conditions_sha256': hashlib.sha256(initial.numpy().tobytes()).hexdigest()}
    directory = output / effective['dyn_sys_name'] / task / split
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous[k] != v for k, v in identity.items()):
            raise ValueError(f'{directory} already contains a different dataset; use another output directory')
        SavedDataset(directory, effective, seed)
        print(f'REUSE {directory}', flush=True)
        return
    pending = directory / 'generation.json'
    if pending.exists() and json.loads(pending.read_text()) != identity:
        raise ValueError(f'{directory} contains an incompatible unfinished generation')
    pending.write_text(json.dumps(identity, indent=2))
    np.save(directory / 'initial_conditions.npy', initial.numpy())
    (directory / 'source_config.yaml').write_text(source.read_text())
    arrays = {}
    begin = time.monotonic()
    with torch.no_grad(), torch.device(device):
        for start in range(0, len(initial), batch_size):
            stop = min(start + batch_size, len(initial))
            chunk_config = dict(effective, size=stop-start)
            dataset = SYSTEMS[effective['dyn_sys_name']][1](initial[start:stop].to(device), **chunk_config)
            for name in ('x', 'y'):
                value = getattr(dataset, name).detach().cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError(f'Nonfinite {task}/{split}/{name} at {start}')
                if name not in arrays:
                    shape = (len(initial), *value.shape[1:])
                    arrays[name] = np.lib.format.open_memmap(directory / f'{name}.npy', mode='w+', dtype=np.float32, shape=shape)
                arrays[name][start:stop] = value
                arrays[name].flush()
            for name in ('times_x', 'times_y'):
                value = getattr(dataset, name).detach().cpu().numpy()
                if start == 0:
                    np.save(directory / f'{name}.npy', value)
                elif not np.array_equal(value, np.load(directory / f'{name}.npy')):
                    raise ValueError('Inconsistent time grids between chunks')
            del dataset
            print(f'{task}/{split}: {stop}/{len(initial)} ({time.monotonic()-begin:.1f}s)', flush=True)
    stats = {}
    for name, array in arrays.items():
        tensor = torch.from_numpy(array)
        if name == 'y' and effective.get('prediction_horizon', 0) > 0:
            tensor = tensor[:, :int(effective['prediction_horizon'] / effective['dt'])]
        norm = UnitGaussianNormalizer(tensor.reshape(-1, tensor.shape[-1]))
        stats[name] = {'mean': norm.mean.tolist(), 'std': norm.std.tolist(), 'eps': norm.eps}
    metadata = dict(identity, format_version=1, normalizers=stats,
                    shapes={name: list(array.shape) for name, array in arrays.items()},
                    source_config=str(source.resolve()),
                    torch_version=str(torch.__version__), solver='torchdiffeq dopri5, rtol=1e-7, atol=1e-9',
                    normalization='per split, matching BaseDataset; y statistics use the first horizon',
                    seed_policy='train/val/test draw sequentially from experiment.seed; eval uses evaluation.seed, continuing the stream if equal',
                    elapsed_seconds=time.monotonic()-begin)
    temporary = directory / 'manifest.json.tmp'
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(manifest)
    saved = SavedDataset(directory, effective, seed)
    assert len(saved) == len(initial)
    assert torch.isfinite(saved.x).all() and torch.isfinite(saved.y).all()
    print(f'SAVED {directory}: x={tuple(saved.x.shape)}, y={tuple(saved.y.shape)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--configs', nargs='+', type=Path, default=sorted(root.glob('configs/*.yaml')),
                        help='Configs whose datasets to generate (default: all)')
    parser.add_argument('--output', type=Path, default=data_path('${TNO_DATA_DIR}'),
                        help='Data root (default: $TNO_DATA_DIR, or <repo>/data)')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test', 'eval'], default=['train', 'val', 'test', 'eval'])
    parser.add_argument('--batch-size', type=int,
                        help='Trajectories solved at once (default: data_generation.batch_size in each config). '
                             'Results can differ slightly between batch sizes, so the batch size is part of a dataset\'s identity.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.device == 'cuda' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Expected exactly one visible GPU')
    torch.set_num_threads(1)
    for source in args.configs:
        config = yaml.safe_load(source.read_text())
        if config['dataset']['dyn_sys_name'] not in SYSTEMS:
            raise ValueError(f'This generator supports {sorted(SYSTEMS)} only')
        task = 'forecasting' if config['dataset'].get('prediction_horizon', 0) else 'smoothing'
        if config['dataset'].get('saved_data_dir'):
            # Write where the config will load from, e.g. .../KuramotoSivashinsky/forecasting_h5.
            target = data_path(config['dataset']['saved_data_dir']).resolve()
            if target.parent != args.output.resolve() / config['dataset']['dyn_sys_name']:
                raise ValueError(f'{source}: saved_data_dir {target} is not under {args.output}')
            task = target.name
        batch_size = args.batch_size or config.get('data_generation', {}).get('batch_size', 50000)
        states = split_initial_states(config)
        for split in args.splits:
            generate(config, source, task, split, states[split], args.output, batch_size, args.device)
    print('ALL REQUESTED DATASETS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
