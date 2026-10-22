"""Small CPU checks of saved L96 dataset generation and loading."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml

from scripts.generate_data import generate, split_initial_states
from src.datasets.dynamical import Lorenz96Dataset
from src.datasets.saved import load_saved_split


class SavedL96Tests(unittest.TestCase):
    def test_forecasting_rollout(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / 'configs/l96_forecasting.yaml').read_text())
        data = config['dataset']
        data.update(T=0.1, T_in=0.1, burn_in_time=0.1,
                    n_trajectories_train=4, n_trajectories_val=2, n_trajectories_test=2)
        states = split_initial_states(config)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            data['saved_data_dir'] = str(output / 'Lorenz96/forecasting')
            source = output / 'config.yaml'
            source.write_text(yaml.safe_dump(config))
            for split in states:
                generate(config, source, 'forecasting', split, states[split], output, 2, 'cpu')
                seed = config['evaluation' if split == 'eval' else 'experiment']['seed']
                saved = load_saved_split(data, split, seed)
                steps = 500 if split == 'eval' else 10
                self.assertEqual(tuple(saved.x.shape), (len(states[split]), 6, 30))
                self.assertEqual(tuple(saved.y.shape), (len(states[split]), steps, 30))
                self.assertTrue(torch.isfinite(saved.y).all())
                self.assertTrue(torch.all(saved.times_y[1:] > saved.times_y[:-1]))
                if split == 'eval':
                    short = load_saved_split(data, split, seed, compositions=1)
                    torch.testing.assert_close(short.y, saved.y[:, :10])
                    torch.testing.assert_close(short.times_y, saved.times_y[:10])

    def test_smoothing_roundtrip(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / 'configs/l96_smoothing.yaml').read_text())
        data = config['dataset']
        data.update(T=0.1, burn_in_time=0.1, n_trajectories_train=4,
                    n_trajectories_val=2, n_trajectories_test=2)
        states = split_initial_states(config)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            source = output / 'config.yaml'
            source.write_text(yaml.safe_dump(config))
            data['saved_data_dir'] = str(output / 'Lorenz96/smoothing')
            for split in states:
                generate(config, source, 'smoothing', split, states[split], output, 2, 'cpu')
                seed = config['evaluation' if split == 'eval' else 'experiment']['seed']
                saved = load_saved_split(data, split, seed)
                self.assertEqual(tuple(saved.x.shape), (len(states[split]), 5, 30))
                self.assertEqual(tuple(saved.y.shape), (len(states[split]), 5, 10))
                np.testing.assert_array_equal(
                    np.load(Path(data['saved_data_dir']) / split / 'initial_conditions.npy'),
                    states[split].numpy())
                torch.testing.assert_close(saved.x_normalizer.decode(saved.x), saved.raw_x)
                torch.testing.assert_close(saved.y_normalizer.decode(saved.y), saved.raw_y)
            # Independently reproduce the first training batch with the normal dataset.
            torch.manual_seed(config['experiment']['seed'])
            direct = Lorenz96Dataset(**dict(data, size=2, test=False))
            train = load_saved_split(data, 'train', config['experiment']['seed'])
            torch.testing.assert_close(direct.x_normalizer.decode(direct.x), train.raw_x[:2])
            torch.testing.assert_close(direct.y_normalizer.decode(direct.y), train.raw_y[:2])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
