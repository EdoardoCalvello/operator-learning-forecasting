"""Load persistent, raw trajectory arrays without rerunning the ODE solver."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import UnitGaussianNormalizer

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_path(path):
    """Expand environment variables in a data path; TNO_DATA_DIR defaults to <repo>/data."""
    os.environ.setdefault('TNO_DATA_DIR', str(REPO_ROOT / 'data'))
    return Path(os.path.expandvars(str(path)))


class SavedDataset(Dataset):
    def __init__(self, directory, expected_config=None, expected_seed=None, compositions=None,
                 normalizer_manifest=None):
        directory = data_path(directory)
        with (directory / 'manifest.json').open() as stream:
            self.metadata = json.load(stream)
        if self.metadata['format_version'] != 1:
            raise ValueError(f'Unsupported dataset format in {directory}')
        actual = self.metadata['dataset_config']
        defaults = {'obs_noise_std': 0.0, 'T_in': actual.get('T'), 'compositions': 1}
        # Smoothing observation noise is applied at load time to the clean saved arrays.
        smoothing = actual.get('prediction_horizon', 0) == 0
        noise_std = ((expected_config or {}).get('obs_noise_std') or 0.0) if smoothing else 0.0
        if expected_config is not None:
            for key, value in expected_config.items():
                if key == 'compositions' and compositions is not None:
                    continue
                if key == 'obs_noise_std' and smoothing:
                    continue
                recorded = actual.get(key, defaults.get(key))
                if (key in actual or key in defaults) and recorded != value:
                    raise ValueError(f'Saved dataset {directory}: {key}={recorded!r}, requested {value!r}')
        if expected_seed is not None and self.metadata['seed'] != expected_seed:
            raise ValueError(f'Saved dataset {directory} has a different seed')
        self.size = actual['size']
        for key, value in actual.items():
            setattr(self, key, value)
        self.raw_x = torch.from_numpy(np.load(directory / 'x.npy', mmap_mode='c'))
        self.raw_y = torch.from_numpy(np.load(directory / 'y.npy', mmap_mode='c'))
        for name in ('times_x', 'times_y'):
            setattr(self, name, torch.from_numpy(np.load(directory / f'{name}.npy')))
        if list(self.raw_x.shape) != self.metadata['shapes']['x'] or list(self.raw_y.shape) != self.metadata['shapes']['y']:
            raise ValueError(f'Saved array shape mismatch in {directory}')
        if self.times_x.shape != (self.raw_x.shape[1], 1) or self.times_y.shape != (self.raw_y.shape[1], 1):
            raise ValueError(f'Saved time-grid shape mismatch in {directory}')
        if self.raw_x.shape[0] != self.size or self.raw_y.shape[0] != self.size:
            raise ValueError(f'Saved trajectory count mismatch in {directory}')
        if compositions is not None and actual.get('prediction_horizon', 0) > 0:
            if not 1 <= compositions <= actual['compositions']:
                raise ValueError('Requested forecast compositions are unavailable')
            steps = self.raw_y.shape[1] // actual['compositions'] * compositions
            self.raw_y = self.raw_y[:, :steps]
            self.times_y = self.times_y[:steps]
            self.compositions = compositions
        self.obs_noise_std = noise_std
        if noise_std > 0.0:
            # Noisy observations, clean targets; fixed per split so runs are reproducible.
            split_index = ('train', 'val', 'test', 'eval').index(self.metadata['split'])
            generator = torch.Generator().manual_seed(self.metadata['seed'] + 1000 * split_index)
            self.raw_x = self.raw_x + noise_std * torch.randn(self.raw_x.shape, generator=generator)
        # Every split must be encoded with the training statistics the model was fitted on.
        normalizers = self.metadata['normalizers']
        if normalizer_manifest is not None:
            normalizers = json.loads(Path(normalizer_manifest).read_text())['normalizers']
        for name in ('x', 'y'):
            stats = normalizers[name]
            normalizer = UnitGaussianNormalizer.__new__(UnitGaussianNormalizer)
            normalizer.mean = torch.tensor(stats['mean'], dtype=torch.float32)
            normalizer.std = torch.tensor(stats['std'], dtype=torch.float32)
            normalizer.eps = stats['eps']
            setattr(self, f'{name}_normalizer', normalizer)
            setattr(self, name, normalizer.encode(getattr(self, f'raw_{name}')))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.times_x, self.times_y


def load_saved_split(config, split, seed, compositions=None, train_dir=None):
    """Load a named split, normalized with the train split's statistics.

    Rejects stale settings instead of regenerating. `train_dir` overrides where the
    training statistics are read from (default: the sibling `train` split).
    """
    expected = dict(config)
    expected['size'] = config['n_trajectories_test' if split == 'eval' else f'n_trajectories_{split}']
    expected['test'] = split in ('test', 'eval')
    expected['compositions'] = config.get('compositions', 1) if split == 'eval' else 1
    root = data_path(config['saved_data_dir'])
    directory = root / split
    normalizer_manifest = None
    if split != 'train':
        normalizer_manifest = Path(train_dir or root / 'train') / 'manifest.json'
        if not normalizer_manifest.exists():
            raise FileNotFoundError(f'Training statistics for normalization not found: {normalizer_manifest}')
    return SavedDataset(directory, expected, seed, compositions=compositions,
                        normalizer_manifest=normalizer_manifest)
