"""Initial condition generators."""

from typing import Dict
import numpy as np
import cupy as cp
from ..simulation.domain import Domain
from ..simulation.initial_conditions import InitialConditions


class FlatNoisy(InitialConditions):
    """
    Flat field with optional Gaussian noise.
    """
    
    def __init__(self, phi0: float, noise_amplitude: float = 0.0, seed: int = None):
        """
        Args:
            phi0: Average density
            noise_amplitude: Noise level
            seed: Random seed for reproducibility
        """
        self.phi0 = phi0
        self.noise_amplitude = noise_amplitude
        self.seed = seed
    
    def __call__(self, domain: Domain) -> Dict[str, np.ndarray]:
        """Generate flat field with noise."""
        if self.seed is not None:
            cp.random.seed(self.seed)
        
        # Create flat field
        phi = self.phi0 * cp.ones(domain.shape, dtype=cp.float64)
        
        # Add noise if requested
        if self.noise_amplitude > 0:
            noise = cp.random.normal(0, 1, domain.shape)
            noise *= self.noise_amplitude / cp.sqrt((noise**2).mean())
            phi += noise
        
        # Normalize to maintain average
        phi_hat = cp.fft.fft2(phi)
        phi_hat[0, 0] = domain.shape[0] * domain.shape[1] * self.phi0
        phi = cp.fft.ifft2(phi_hat).real
        
        return {'phi': phi}


class TriangularLattice(InitialConditions):
    """
    Triangular lattice initial condition for 2D PFC.
    """
    
    def __init__(
        self, 
        phi0: float, 
        amplitude: float = None, 
        noise_amplitude: float = 0.0,
        seed: int = None
    ):
        """
        Args:
            phi0: Average density
            amplitude: Amplitude of density modulation (default: phi0/3)
            noise_amplitude: Noise level
            seed: Random seed
        """
        self.phi0 = phi0
        self.amplitude = amplitude if amplitude is not None else phi0 / 3.0
        self.noise_amplitude = noise_amplitude
        self.seed = seed
    
    def __call__(self, domain: Domain) -> Dict[str, np.ndarray]:
        """Generate triangular lattice."""
        if self.seed is not None:
            cp.random.seed(self.seed)
        
        # Get coordinates from domain
        x_cpu, y_cpu = domain.get_coordinates()
        x = cp.asarray(x_cpu)
        y = cp.asarray(y_cpu)
        
        # Create meshgrid on GPU
        X, Y = cp.meshgrid(x, y)
        
        # Three wave vectors for triangular lattice
        q1 = cp.array([-np.sqrt(3)*0.5, -0.5], dtype=cp.float64)
        q2 = cp.array([0.0, 1.0], dtype=cp.float64)
        q3 = cp.array([np.sqrt(3)*0.5, -0.5], dtype=cp.float64)
        
        # Stack coordinates
        r = cp.stack([X, Y], axis=-1)
        
        # Create lattice
        phi = (
            2 * self.amplitude * (
                cp.cos(r.dot(q1)) +
                cp.cos(r.dot(q2)) +
                cp.cos(r.dot(q3))
            ) + self.phi0
        )
        
        # Normalize average
        phi_hat = cp.fft.fft2(phi)
        phi_hat[0, 0] = domain.shape[0] * domain.shape[1] * self.phi0
        phi = cp.fft.ifft2(phi_hat).real
        
        # Add noise if requested
        if self.noise_amplitude > 0:
            noise = cp.random.normal(0, 1, domain.shape)
            noise *= self.noise_amplitude / cp.sqrt((noise**2).mean())
            phi += noise
            
            # Re-normalize average
            phi_hat = cp.fft.fft2(phi)
            phi_hat[0, 0] = domain.shape[0] * domain.shape[1] * self.phi0
            phi = cp.fft.ifft2(phi_hat).real
        
        return {'phi': phi}
