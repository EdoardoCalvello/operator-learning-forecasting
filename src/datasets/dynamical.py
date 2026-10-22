"""
Specific dynamical systems: Lorenz63, Lorenz96, KS equation.
"""

import torch
import numpy as np
from .base import DynSys, BaseDataset


class Lorenz63(DynSys):
    """Lorenz 63 system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz"""
    
    def __init__(self, state_dim: int = 3, sigma: float = 10, rho: float = 28, beta: float = 8/3, 
                 init_ranges: list = None):
        super().__init__(state_dim=state_dim)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
        # Default initial condition ranges: [x_min, x_max, y_min, y_max, z_min, z_max]
        if init_ranges is None:
            self.init_ranges = [-15, 15, -15, 15, 0, 40]
        else:
            if len(init_ranges) != 6:
                raise ValueError("init_ranges for Lorenz63 must have 6 values: [x_min, x_max, y_min, y_max, z_min, z_max]")
            self.init_ranges = init_ranges

    def rhs(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Lorenz63 right-hand side."""
        x_coord, y_coord, z_coord = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        dx = self.sigma * (y_coord - x_coord)
        dy = x_coord * (self.rho - z_coord) - y_coord
        dz = x_coord * y_coord - self.beta * z_coord
        return torch.cat([dx, dy, dz], dim=1)

    def get_inits(self, size: int) -> torch.Tensor:
        """Generate random initial conditions for Lorenz63 with configurable ranges."""
        x_min, x_max, y_min, y_max, z_min, z_max = self.init_ranges
        
        x0 = torch.empty(size, 1).uniform_(x_min, x_max)
        y0 = torch.empty(size, 1).uniform_(y_min, y_max)
        z0 = torch.empty(size, 1).uniform_(z_min, z_max)
        
        return torch.cat([x0, y0, z0], dim=1)


class Lorenz96(DynSys):
    """Lorenz 96 system: dX_i/dt = (X_{i+1} - X_{i-2})X_{i-1} - X_i + F"""
    
    def __init__(self, state_dim: int = 40, forcing: float = 8.0, init_ranges: list = None):
        super().__init__(state_dim=state_dim)
        self.forcing = forcing
        self.N = state_dim
        
        # Default initial condition ranges: [min_deviation, max_deviation] around forcing
        if init_ranges is None:
            self.init_ranges = [-1, 1]  # Default: forcing + uniform(-1, 1)
        else:
            if len(init_ranges) != 2:
                raise ValueError("init_ranges for Lorenz96 must have 2 values: [min_deviation, max_deviation] from forcing")
            self.init_ranges = init_ranges

    def rhs(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Lorenz96 right-hand side with periodic boundary conditions."""
        # x shape: (batch_size, N)
        batch_size = x.shape[0]
        
        # Create shifted versions for periodic boundary conditions
        x_plus1 = torch.roll(x, -1, dims=1)  # X_{i+1}
        x_minus1 = torch.roll(x, 1, dims=1)  # X_{i-1}  
        x_minus2 = torch.roll(x, 2, dims=1)  # X_{i-2}
        
        # Lorenz96 equation: (X_{i+1} - X_{i-2})X_{i-1} - X_i + F
        dxdt = (x_plus1 - x_minus2) * x_minus1 - x + self.forcing
        
        return dxdt

    def get_inits(self, size: int) -> torch.Tensor:
        """Generate random initial conditions for Lorenz96 with configurable ranges."""
        min_dev, max_dev = self.init_ranges
        
        x0 = torch.empty(size, self.N).uniform_(min_dev, max_dev) + self.forcing
        
        return x0


class KuramotoSivashinsky(DynSys):
    """Kuramoto-Sivashinsky equation: u_t + u*u_x + u_xx + u_xxxx = 0
    
    Uses pseudo-spectral method with 2/3 dealiasing rule to prevent aliasing errors.
    """
    
    def __init__(self, state_dim: int = 64, domain_length: float = 32.0, init_ranges: list = None):
        super().__init__(state_dim=state_dim)
        self.N = state_dim
        self.L = domain_length
        
        # Spatial domain and wavenumbers
        self.x = torch.linspace(0, self.L, self.N)
        self.dx = self.L / self.N
        
        # Wavenumber for Fourier transform (assuming periodic BC)
        k = 2 * np.pi * torch.fft.fftfreq(self.N, d=self.dx)
        self.k = k
        
        # Dealiasing mask using 2/3 rule
        # Zero out modes with |k| > 2/3 * k_max to prevent aliasing
        k_max = np.pi / self.dx  # Maximum resolved wavenumber
        self.dealias_mask = (torch.abs(self.k) <= (2./3.) * k_max).float()
        
        # Default initial condition parameters: [max_modes, amplitude_scale, decay_rate] 
        if init_ranges is None:
            self.init_ranges = [5, 1.0, 1.0]  # 5 modes, amplitude 1.0, decay rate 1.0
        else:
            if len(init_ranges) != 3:
                raise ValueError("init_ranges for KS must have 3 values: [max_modes, amplitude_scale, decay_rate]")
            self.init_ranges = init_ranges

    def rhs(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """KS equation right-hand side using pseudo-spectral method with dealiasing."""
        # x shape: (batch_size, N)
        batch_size = x.shape[0]
        
        # Vectorized computation for all samples in batch
        # Fourier transform for all samples
        u_hat = torch.fft.fft(x, dim=1)  # (batch_size, N)
        
        # Apply dealiasing mask to input
        u_hat = u_hat * self.dealias_mask.unsqueeze(0)  # (batch_size, N)
        
        # Derivatives in Fourier space
        k = self.k.unsqueeze(0)  # (1, N) for broadcasting
        u_x_hat = 1j * k * u_hat
        u_xx_hat = -(k**2) * u_hat
        u_xxxx_hat = (k**4) * u_hat
        
        # Nonlinear term: u*u_x (computed in physical space with proper dealiasing)
        u = torch.real(torch.fft.ifft(u_hat, dim=1))  # (batch_size, N)
        u_x = torch.real(torch.fft.ifft(u_x_hat, dim=1))  # (batch_size, N)
        
        nonlinear = u * u_x  # (batch_size, N)
        nonlinear_hat = torch.fft.fft(nonlinear, dim=1)  # (batch_size, N)
        
        # Apply dealiasing to nonlinear term to prevent aliasing errors
        nonlinear_hat = nonlinear_hat * self.dealias_mask.unsqueeze(0)
        
        # KS equation: u_t = -u*u_x - u_xx - u_xxxx
        dudt_hat = -nonlinear_hat - u_xx_hat - u_xxxx_hat
        
        # Apply final dealiasing to result
        dudt_hat = dudt_hat * self.dealias_mask.unsqueeze(0)
        
        # Transform back to physical space
        dxdt = torch.real(torch.fft.ifft(dudt_hat, dim=1))
        
        return dxdt

    def get_inits(self, size: int) -> torch.Tensor:
        """Generate random initial conditions for KS equation with configurable parameters.
        
        Uses Fourier mode initialization that respects dealiasing constraints.
        """
        max_modes, amplitude_scale, decay_rate = self.init_ranges
        max_modes = int(max_modes)
        
        # Ensure max_modes respects dealiasing constraint (2/3 rule)
        k_max_dealias = int((2./3.) * self.N // 2)
        max_modes = min(max_modes, k_max_dealias)
        
        u0 = torch.zeros(size, self.N)
        
        # Generate initial conditions in Fourier space for better control
        for i in range(size):
            u_hat = torch.zeros(self.N, dtype=torch.complex64)
            
            # Add random modes with controllable decay
            for mode in range(1, max_modes + 1):  # Skip DC component (mode 0)
                amplitude = amplitude_scale * np.exp(-decay_rate * (mode - 1))
                
                # Random complex amplitude with uniform phase
                real_part = torch.randn(1) * amplitude
                imag_part = torch.randn(1) * amplitude
                complex_amp = real_part + 1j * imag_part
                
                # Set positive and negative frequencies (maintain reality condition)
                u_hat[mode] = complex_amp
                u_hat[-mode] = torch.conj(complex_amp)  # Hermitian symmetry for real signal
            
            # Apply dealiasing mask to initial conditions
            u_hat = u_hat * self.dealias_mask
            
            # Convert to physical space
            u0[i] = torch.real(torch.fft.ifft(u_hat))
        
        return u0


# Dataset classes for each system

class Lorenz63Dataset(BaseDataset):
    """Dataset for Lorenz63 system."""
    
    def _load_system(self) -> DynSys:
        return Lorenz63(**self.params)


class Lorenz96Dataset(BaseDataset):
    """Dataset for Lorenz96 system."""
    
    def _load_system(self) -> DynSys:
        # Default to 40-dimensional system
        if 'state_dim' not in self.params:
            self.params['state_dim'] = 40
        return Lorenz96(**self.params)


class KSDataset(BaseDataset):
    """Dataset for Kuramoto-Sivashinsky equation."""
    
    def _load_system(self) -> DynSys:
        # Default parameters for KS
        if 'state_dim' not in self.params:
            self.params['state_dim'] = 64
        return KuramotoSivashinsky(**self.params)


# Factory function to load datasets
def load_dataset(dyn_sys_name: str, **kwargs) -> BaseDataset:
    """Factory function to create datasets by name."""
    dataset_classes = {
        'Lorenz63': Lorenz63Dataset,
        'Lorenz96': Lorenz96Dataset, 
        'KS': KSDataset,
        'KuramotoSivashinsky': KSDataset,  # Alias
    }
    
    if dyn_sys_name not in dataset_classes:
        raise ValueError(f"Dataset '{dyn_sys_name}' not found. Available: {list(dataset_classes.keys())}")
    
    return dataset_classes[dyn_sys_name](dyn_sys_name=dyn_sys_name, **kwargs)