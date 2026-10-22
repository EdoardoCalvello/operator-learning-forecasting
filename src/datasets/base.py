"""
Base classes for dynamical systems and datasets.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from torchdiffeq import odeint
from typing import Tuple, Optional, Dict, Any


class DynSys(ABC):
    """Base class for dynamical systems."""
    
    def __init__(self, state_dim: int = 1):
        self.state_dim = state_dim
    
    @abstractmethod
    def rhs(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Right-hand side function of the dynamical system."""
        pass
    
    @abstractmethod
    def get_inits(self, size: int) -> torch.Tensor:
        """Generate initial conditions."""
        pass
    
    def solve(self, N_traj: int, T: float, dt: float, sample_rate: float,
              test: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the dynamical system using ODE solver.
        
        Args:
            N_traj: Number of trajectories
            T: Total integration time
            dt: Integration time step (controls ODE solver accuracy)
            sample_rate: Data collection time step (controls output frequency)
            test: Whether this is for testing
        
        Returns:
            xyz: (seq_len, N_traj, state_dim) - trajectories sampled at sample_rate
            times: (seq_len,) - time points sampled at sample_rate
        """
        # Create fine time grid for accurate ODE integration
        times_fine = torch.arange(0, T, dt)
        xyz0 = self.get_inits(N_traj)
        
        # Solve ODE with fine time step for accuracy
        xyz_fine = odeint(self.rhs, xyz0, times_fine)
        
        # Sample at the desired collection rate
        sample_step = int(sample_rate / dt)
        if sample_step < 1:
            raise ValueError(f"sample_rate ({sample_rate}) must be >= dt ({dt})")
        
        # Sample every sample_step points
        xyz = xyz_fine[::sample_step]
        times = times_fine[::sample_step]
        
        return xyz, times


class BaseDataset(Dataset, ABC):
    """Base dataset class for dynamical systems."""
    
    def __init__(
        self,
        size: int = 1000,
        T: float = 1.0,
        T_in: Optional[float] = None,  # If None, use T for input period
        compositions: int = 1,
        dt: float = 0.01,
        sample_rate: float = 0.01,
        dyn_sys_name: str = 'Lorenz63',
        input_inds: list = [0],
        output_inds: list = [-1],
        plot_inds: list = [-1],
        prediction_horizon: float = 0.0,  # Time ahead to predict (0.0 = smoothing, >0.0 = forecasting)
        burn_in_time: float = 5.0,  # Time to run solver before collecting data (removes transients)
        obs_noise_std: float = 0.0,  # Std of Gaussian noise added to input (observations) in smoothing
        test: bool = False,
        params: dict = None,
        **kwargs
    ):
        self.size = size
        self.T = T
        self.T_in = T_in or T
        self.compositions = compositions or 1
        self.dt = dt
        self.sample_rate = sample_rate
        self.dyn_sys_name = dyn_sys_name
        self.input_inds = input_inds
        self.output_inds = output_inds
        self.plot_inds = plot_inds
        self.prediction_horizon = prediction_horizon
        self.burn_in_time = burn_in_time
        self.obs_noise_std = obs_noise_std
        self.test = test
        self.params = params or {}
        
        # Initialize system and generate data
        self.dynsys = self._load_system()
        self.generate_data()
    
    @abstractmethod
    def _load_system(self) -> DynSys:
        """Load the appropriate dynamical system."""
        pass
    
    def generate_data(self):
        """Generate the input and output data with burn-in period to remove transients."""
        # Total time needed: burn-in + data collection + prediction horizon
        T_total = self.burn_in_time + self.T + self.prediction_horizon

        if self.prediction_horizon > 0.0:
            # For forecasting: solve in two phases with different time steps
            # Phase 1: Burn-in + input period with normal dt
            T_phase1 = self.burn_in_time + self.T
            xyz_phase1, times_phase1 = self.dynsys.solve(
                N_traj=self.size, 
                T=T_phase1+ self.dt, 
                dt=self.dt,
                sample_rate=self.sample_rate,
                test=self.test
            )
            
            # Phase 2: Prediction horizon with scaled dt to maintain same number of steps
            input_steps = int(self.T_in / self.sample_rate)
            # Scale dt to get same number of steps in prediction horizon as in T
            dt_scaled = self.prediction_horizon / input_steps #* (self.sample_rate / self.dt)
            #
            dt_scaled = self.dt
            #
            # Use final state from phase 1 as initial condition for phase 2.
            # Integrate one composition at a time so each segment is identical
            # regardless of total number of compositions (avoids adaptive-solver
            # inconsistency when the full time span changes).
            one_horizon = self.prediction_horizon  # unscaled, single-horizon duration
            self.prediction_horizon = one_horizon * self.compositions
            n_steps_per = int(one_horizon / dt_scaled)
            times_one = torch.linspace(0, one_horizon, n_steps_per + 1)

            xyz0_seg = xyz_phase1[-1, :, :]
            segments = []
            for _ in range(self.compositions):
                seg = odeint(self.dynsys.rhs, xyz0_seg, times_one)
                segments.append(seg[1:])  # drop t=0 (endpoint of previous segment)
                xyz0_seg = seg[-1]
            xyz_phase2 = torch.cat(segments, dim=0)

            # Combine phases
            times_phase2 = torch.cat([
                one_horizon * c + times_one[1:] for c in range(self.compositions)
            ])
            xyz_full = torch.cat([xyz_phase1, xyz_phase2], dim=0)
            times_full = torch.cat([times_phase1, times_phase1[-1] + times_phase2])

        else:
            # Smoothing: solve once with normal dt
            xyz_full, times_full = self.dynsys.solve(
                N_traj=self.size, 
                T=T_total, 
                dt=self.dt,
                sample_rate=self.sample_rate,
                test=self.test
            )
        
        # Remove burn-in period - only keep data after transients have settled
        burn_in_steps = int(self.burn_in_time / self.sample_rate)
        xyz = xyz_full[burn_in_steps:, :, :]  # Remove first burn_in_steps
        times = times_full[burn_in_steps:] - self.burn_in_time  # Shift time to start at 0
        
        if self.prediction_horizon > 0.0:
            # Forecasting: split input/output in time
            input_steps = int(self.T_in / self.sample_rate)+1
            if self.dyn_sys_name=='KuramotoSivashinsky':
                # For KS: zero out high modes not in input_inds/output_inds, then irfft
                # back to physical space (low-pass filtered solution).
                # xyz shape: (seq_len, N_traj, state_dim)
                state_dim = xyz.shape[2]
                n_complex = state_dim // 2 + 1

                xyz_fft = torch.fft.rfft(xyz, dim=2)  # (seq_len, N_traj, n_complex) complex

                def _filter_and_irfft(data_fft, mode_inds):
                    """Zero out complex modes not in mode_inds, then irfft."""
                    filtered = torch.zeros_like(data_fft)
                    filtered[..., mode_inds] = data_fft[..., mode_inds]
                    return torch.fft.irfft(filtered, n=state_dim, dim=-1)  # (..., state_dim)

                xyz_x_filtered = _filter_and_irfft(xyz_fft[:input_steps], self.input_inds)
                xyz_y_filtered = _filter_and_irfft(xyz_fft[input_steps:], self.output_inds)

                self.x = xyz_x_filtered.permute(1, 0, 2)  # (N_traj, input_steps, state_dim)
                self.y = xyz_y_filtered.permute(1, 0, 2)  # (N_traj, output_steps, state_dim)
            else:
                self.x = xyz[:input_steps, :, self.input_inds].permute(1, 0, 2) 
                self.y = xyz[input_steps:, :, self.output_inds].permute(1, 0, 2) 
            self.times_x = times[:input_steps].unsqueeze(-1)
            self.times_y = times[input_steps:].unsqueeze(-1)
        else:
            if self.dyn_sys_name=='KuramotoSivashinsky':
                # For KS: zero out high modes not in input_inds/output_inds, then irfft
                # back to physical space (low-pass filtered solution).
                # xyz shape: (seq_len, N_traj, state_dim)
                state_dim = xyz.shape[2]
                n_complex = state_dim // 2 + 1

                xyz_fft = torch.fft.rfft(xyz, dim=2)  # (seq_len, N_traj, n_complex) complex

                def _filter_and_irfft(data_fft, mode_inds):
                    """Zero out complex modes not in mode_inds, then irfft."""
                    filtered = torch.zeros_like(data_fft)
                    filtered[..., mode_inds] = data_fft[..., mode_inds]
                    return torch.fft.irfft(filtered, n=state_dim, dim=-1)  # (..., state_dim)

                xyz_x_filtered = _filter_and_irfft(xyz_fft, self.input_inds)
                xyz_y_filtered = _filter_and_irfft(xyz_fft, self.output_inds)

                self.x = xyz_x_filtered.permute(1, 0, 2)  # (N_traj, input_steps, state_dim)
                self.y = xyz_y_filtered.permute(1, 0, 2)  # (N_traj, output_steps, state_dim)
                self.times_x = times.unsqueeze(-1)
                self.times_y = times.unsqueeze(-1)
            else:
                # Smoothing: same time for input/output, different variables
                self.x = xyz[:, :, self.input_inds].permute(1, 0, 2)
                self.y = xyz[:, :, self.output_inds].permute(1, 0, 2)
                if self.obs_noise_std > 0.0:
                    self.x = self.x + self.obs_noise_std * torch.randn_like(self.x)
                self.times_x = times.unsqueeze(-1)
                self.times_y = times.unsqueeze(-1)

        
        # Normalize data
        self._normalize_data()
    
    def _normalize_data(self):
        """Normalize input and output data.

        Normalizer stats are computed from the first prediction horizon only
        so they are invariant to the number of compositions.
        """
        from .transforms import UnitGaussianNormalizer

        one_horizon_steps = int((self.prediction_horizon/self.compositions) / self.dt)
        y_steps = one_horizon_steps if one_horizon_steps > 1 else self.y.shape[1]

        self.x_normalizer = UnitGaussianNormalizer(
            self.x.reshape(-1, self.x.shape[-1])
        )
        self.y_normalizer = UnitGaussianNormalizer(
            self.y[:, :y_steps, :].reshape(-1, self.y.shape[-1])
        )

        self.x = self.x_normalizer.encode(self.x)
        self.y = self.y_normalizer.encode(self.y)
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.times_x, self.times_y