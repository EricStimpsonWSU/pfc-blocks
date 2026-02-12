"""Base class for dynamics (time evolution)."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..operators.base_ops import Operators
    from ..models.base_model import PFCModel


class Dynamics(ABC):
    """
    Abstract base class for dynamics (time evolution).
    
    Defines time evolution equations (first-order: d/dt, second-order: d2/dt2 + d/dt).
    Dynamics is pure: no FFT/FD details beyond calling Operators.
    """
    
    def __init__(self):
        self._config: Dict = {}
    
    @property
    @abstractmethod
    def order(self) -> int:
        """1 for first-order (d/dt), 2 for second-order (d2/dt2 + d/dt)."""
        pass
    
    @abstractmethod
    def compute_fields_next(
        self,
        fields: Dict[str, np.ndarray],
        dt: float,
        operators: 'Operators',
        model: 'PFCModel',
        noise_fn: Optional[Callable] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute field values at next timestep (fields -> fields_next).
        
        Args:
            fields: dict mapping field names to current field arrays
            dt: timestep size
            operators: Operators instance (provides Laplacian, etc.)
            model: PFCModel instance (provides functional_derivative, etc.)
            noise_fn: optional callable for noise (amplitude, dtype, shape)
        
        Returns:
            dict mapping field names to updated field arrays (same dtype/shape)
        
        Example for first-order: 
            fields_next = fields + dt * L(fields) + dt * N(fields) + noise
            where L is linear (spectral or FD), N is nonlinear, noise is optional.
        """
        pass
    
    @abstractmethod
    def compute_fields_velocity(
        self,
        fields: Dict[str, np.ndarray],
        operators: 'Operators',
        model: 'PFCModel'
    ) -> Dict[str, np.ndarray]:
        """
        Compute velocity (d fields / dt) at current state (used for output/diagnostics).
        
        Args:
            fields: dict mapping field names to current field arrays
            operators: Operators instance
            model: PFCModel instance
        
        Returns:
            dict mapping field names to velocity arrays (same dtype/shape as fields)
        """
        pass
    
    def configure(self, config: Dict) -> None:
        """Accept configuration dictionary (e.g., noise amplitude, friction, etc.)."""
        self._config = config
