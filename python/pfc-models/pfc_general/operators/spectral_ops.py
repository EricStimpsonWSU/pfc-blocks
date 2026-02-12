"""Spectral operators using FFT."""

from typing import Dict, Union, Tuple
import numpy as np
import cupy as cp
from .base_ops import Operators


class SpectralOperators2D(Operators):
    """
    Spectral (FFT-based) differential operators for 2D periodic domains.
    
    Uses FFT for all derivatives. Assumes periodic boundary conditions.
    Optimized for GPU (CuPy).
    """
    
    def __init__(self, domain=None):
        """
        Args:
            domain: Domain instance with shape, spacing, wavenumbers
        """
        super().__init__()
        self._domain = domain
        self._kx = None
        self._ky = None
        self._k2 = None
        self._k4 = None
        self._k6 = None
        
        if domain is not None:
            self._setup_kernels(domain)
    
    def _setup_kernels(self, domain):
        """Pre-compute k-space kernels."""
        # Get wavenumbers from domain
        kx_cpu, ky_cpu = domain.get_wavenumbers()
        
        # Transfer to GPU
        kx = cp.asarray(kx_cpu)
        ky = cp.asarray(ky_cpu)
        
        # Create 2D meshgrid
        self._kx, self._ky = cp.meshgrid(kx, ky)
        
        # Compute k^2 with spectral anomaly correction
        # Standard: k2 = kx^2 + ky^2
        # Corrected for finite grid:
        dx, dy = domain.spacing
        self._k2 = 2 * (
            (1 / dx**2) * (1 - cp.cos(self._kx * dx)) +
            (1 / dy**2) * (1 - cp.cos(self._ky * dy))
        )
        
        self._k4 = self._k2**2
        self._k6 = self._k2**3
    
    @property
    def method(self) -> str:
        """Spectral method."""
        return 'spectral'
    
    @property
    def laplacian_kernel(self) -> cp.ndarray:
        """K-space Laplacian kernel (-k^2)."""
        return -self._k2
    
    @property
    def k2(self) -> cp.ndarray:
        """K^2 kernel."""
        return self._k2
    
    @property
    def k4(self) -> cp.ndarray:
        """K^4 kernel."""
        return self._k4
    
    @property
    def k6(self) -> cp.ndarray:
        """K^6 kernel."""
        return self._k6
    
    def laplacian(self, field: cp.ndarray) -> cp.ndarray:
        """
        Compute Laplacian using FFT.
        
        Args:
            field: 2D array (GPU)
        
        Returns:
            Laplacian as 2D array (GPU)
        """
        field_hat = cp.fft.fft2(field)
        laplacian_hat = -self._k2 * field_hat
        return cp.fft.ifft2(laplacian_hat).real
    
    def bilaplacian(self, field: cp.ndarray) -> cp.ndarray:
        """
        Compute bilaplacian using FFT.
        
        Args:
            field: 2D array (GPU)
        
        Returns:
            Bilaplacian as 2D array (GPU)
        """
        field_hat = cp.fft.fft2(field)
        bilaplacian_hat = self._k4 * field_hat
        return cp.fft.ifft2(bilaplacian_hat).real
    
    def gradient(self, field: cp.ndarray, axis: int) -> cp.ndarray:
        """
        Compute gradient along axis using FFT.
        
        Args:
            field: 2D array (GPU)
            axis: 0 for x, 1 for y
        
        Returns:
            Gradient as 2D array (GPU)
        """
        field_hat = cp.fft.fft2(field)
        if axis == 0:
            grad_hat = 1j * self._kx * field_hat
        elif axis == 1:
            grad_hat = 1j * self._ky * field_hat
        else:
            raise ValueError("axis must be 0 or 1 for 2D")
        return cp.fft.ifft2(grad_hat).real
    
    def configure(self, config: Dict) -> None:
        """
        Configure operators with domain information.
        
        Args:
            config: dict with 'domain' key
        """
        super().configure(config)
        if 'domain' in config:
            self._domain = config['domain']
            self._setup_kernels(self._domain)
