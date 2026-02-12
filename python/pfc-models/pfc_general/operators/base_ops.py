"""Base class for spatial differential operators."""

from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple
import numpy as np


class Operators(ABC):
    """
    Abstract base class for spatial differential operators.
    
    Provides spatial differential operations (Laplacian, gradient, etc.) in either
    spectral (FFT) or finite-difference (FD) form. Abstracts away numerical method.
    """
    
    def __init__(self):
        self._config: Dict = {}
    
    @property
    @abstractmethod
    def method(self) -> str:
        """'spectral' or 'finite_difference' or hybrid identifier."""
        pass
    
    @property
    @abstractmethod
    def laplacian_kernel(self) -> Union[np.ndarray, Tuple]:
        """
        For spectral: pre-computed k-space kernel (1D/2D/3D).
        For FD: tuple of stencil arrays (for different dimensions).
        """
        pass
    
    @abstractmethod
    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian of field (del^2 field).
        
        Args:
            field: n-d array (dtype float or complex)
        
        Returns:
            Laplacian as n-d array (same dtype/shape)
        """
        pass
    
    @abstractmethod
    def bilaplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute bilaplacian of field (del^4 field).
        
        Args:
            field: n-d array
        
        Returns:
            Bilaplacian as n-d array (same dtype/shape)
        """
        pass
    
    @abstractmethod
    def gradient(self, field: np.ndarray, axis: int) -> np.ndarray:
        """
        Compute gradient along axis (d/dx_axis).
        
        Args:
            field: n-d array
            axis: direction (0=x, 1=y, 2=z, ...)
        
        Returns:
            Gradient as n-d array (same dtype/shape)
        """
        pass
    
    def configure(self, config: Dict) -> None:
        """
        Accept configuration (domain shape, boundary conditions, etc.).
        Pre-compute kernels/stencils if needed.
        """
        self._config = config
