"""Base class for computational backends."""

from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..simulation.domain import Domain
    from ..models.base_model import PFCModel
    from ..dynamics.base_dynamics import Dynamics
    from ..operators.base_ops import Operators


class Backend(ABC):
    """
    Abstract base class for computational backends (CPU/GPU).
    
    Executes time evolution kernels. Owns data layout, dtype, memory, and performance.
    Presents a unified interface to Simulation despite internal CPU/GPU differences.
    """
    
    def __init__(self):
        self._config: Dict = {}
    
    @property
    @abstractmethod
    def device(self) -> str:
        """'cpu' or 'gpu' identifier."""
        pass
    
    @abstractmethod
    def initialize_fields(
        self, 
        domain: 'Domain', 
        initial_conditions_fn
    ) -> Dict[str, np.ndarray]:
        """
        Initialize fields on device.
        
        Args:
            domain: Domain instance (shape, dtype, etc.)
            initial_conditions_fn: callable that returns {field_name: array}
        
        Returns:
            dict mapping field names to device arrays (backend's internal layout/dtype)
        """
        pass
    
    @abstractmethod
    def timestep(
        self,
        fields: Dict[str, np.ndarray],
        dt: float,
        model: 'PFCModel',
        dynamics: 'Dynamics',
        operators: 'Operators',
        noise_amplitude: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Perform one timestep on device.
        
        Args:
            fields: dict of device arrays (current state)
            dt: timestep size
            model: PFCModel instance
            dynamics: Dynamics instance
            operators: Operators instance (pre-configured for device)
            noise_amplitude: scalar noise level
        
        Returns:
            dict of device arrays (updated state)
        
        Note: For GPU, this may involve cupy/pycuda arrays or similar.
        """
        pass
    
    @abstractmethod
    def to_numpy(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transfer fields from device to CPU as numpy arrays for I/O."""
        pass
    
    @abstractmethod
    def from_numpy(self, fields_np: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transfer fields from numpy (CPU) to device."""
        pass
    
    def configure(self, config: Dict) -> None:
        """Accept backend-specific settings (thread count, memory limits, etc.)."""
        self._config = config
