"""Small CPU checks of saved-dataset generation and loading."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from scripts.generate_data import generate, split_initial_states, RawLorenz63Dataset
from src.datasets.saved import SavedDataset, load_saved_split


class SavedDataTests(unittest.TestCase):
    def test_generation_roundtrip_and_split_independence(self):
        for horizon in (0.0, 0.1):
            config = {'dataset': dict(dyn_sys_name='Lorenz63', T=0.1, T_in=0.1,
                dt=0.01, sample_rate=0.01, prediction_horizon=horizon,
                compositions=2, burn_in_time=0.1, input_inds=[0], output_inds=[1, 2],
                n_trajectories_train=4, n_trajectories_val=2, n_trajectories_test=2,
                params=dict(sigma=10., rho=28., beta=2.667)),
                'experiment': {'seed': 42}, 'evaluation': {'seed': 42}}
            states = split_initial_states(config)
            repeated = split_initial_states(config)
            for split in states:
                torch.testing.assert_close(states[split], repeated[split])
            all_states = torch.cat(list(states.values()))
            self.assertEqual(len(torch.unique(all_states, dim=0)), len(all_states))
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root/'config.yaml'
                source.write_text(yaml.safe_dump(config))
                for split in ('train', 'eval'):
                    generate(config, source, 'task', split, states[split], root, 2, 'cpu')
                    directory = root/'Lorenz63/task'/split
                    saved = SavedDataset(directory)
                    effective = dict(saved.metadata['dataset_config'], size=2)
                    raw = RawLorenz63Dataset(states[split][:2], **effective)
                    # Raw persisted values must match the existing solver/slicing.
                    torch.testing.assert_close(saved.raw_x[:2], raw.x)
                    torch.testing.assert_close(saved.raw_y[:2], raw.y)
                    np.testing.assert_array_equal(np.load(directory/'initial_conditions.npy'), states[split].numpy())
                    torch.testing.assert_close(saved.x_normalizer.decode(saved.x), saved.raw_x)
                    torch.testing.assert_close(saved.y_normalizer.decode(saved.y), saved.raw_y)
                    self.assertEqual(saved.metadata['dataset_config']['compositions'], 1 if split == 'train' else 2)
                    # Identical repeated requests reuse the completed files.
                    before = (directory/'x.npy').stat().st_mtime_ns
                    generate(config, source, 'task', split, states[split], root, 2, 'cpu')
                    self.assertEqual(before, (directory/'x.npy').stat().st_mtime_ns)
                    configured = dict(config['dataset'], saved_data_dir=str(directory.parent))
                    reused = load_saved_split(configured, split, 42)
                    torch.testing.assert_close(reused.raw_x, saved.raw_x)
                    # Every split is encoded with the training statistics.
                    train = SavedDataset(root/'Lorenz63/task'/'train')
                    torch.testing.assert_close(reused.x, train.x_normalizer.encode(saved.raw_x))
                    if split == 'eval' and horizon > 0:
                        prefix = load_saved_split(configured, split, 42, compositions=1)
                        torch.testing.assert_close(prefix.raw_y, saved.raw_y[:, :10])
                        torch.testing.assert_close(prefix.y, train.y_normalizer.encode(saved.raw_y[:, :10]))
                        torch.testing.assert_close(prefix.times_y, saved.times_y[:10])
                    with self.assertRaises(ValueError):
                        SavedDataset(directory, expected_seed=0)
                    with self.assertRaises(ValueError):
                        SavedDataset(directory, expected_config={'dt': .02})
                    if horizon > 0:
                        with self.assertRaises(ValueError):
                            SavedDataset(directory, expected_config={'obs_noise_std': .1})
                    else:
                        # Smoothing: noise perturbs observations only, reproducibly.
                        noisy = SavedDataset(directory, expected_config={'obs_noise_std': .1})
                        again = SavedDataset(directory, expected_config={'obs_noise_std': .1})
                        torch.testing.assert_close(noisy.raw_x, again.raw_x)
                        torch.testing.assert_close(noisy.y, saved.y)
                        self.assertFalse(torch.equal(noisy.raw_x, saved.raw_x))
                        torch.testing.assert_close(noisy.x, noisy.x_normalizer.encode(noisy.raw_x))


if __name__ == '__main__':
    unittest.main()
