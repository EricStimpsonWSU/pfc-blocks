"""Base class for PFC models."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np


class PFCModel(ABC):
    """
    Abstract base class for phase field crystal models.
    
    Defines free energy expression and its functional derivatives for one or more
    density fields. Model is pure: no domain/time/IO logic.
    """
    
    def __init__(self):
        self._field_shape: Optional[Tuple[int, ...]] = None
        self._config: Dict = {}
    
    @property
    @abstractmethod
    def num_fields(self) -> int:
        """Number of density fields (1 for single-field, >1 for multi-field)."""
        pass
    
    @property
    def field_shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of a single field in (nx, ny, nz, ...) format. Set by domain."""
        return self._field_shape
    
    @abstractmethod
    def functional_derivative(
        self, 
        fields: Dict[str, np.ndarray], 
        mode_data: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute functional derivatives (d energy / d field).
        
        Args:
            fields: dict mapping field names to n-d arrays (e.g., {'phi': phi_array})
            mode_data: dict with precomputed mode info if needed (single/multi-mode)
        
        Returns:
            dict mapping field names to functional derivative arrays (same shape/dtype)
        """
        pass
    
    @abstractmethod
    def free_energy(
        self, 
        fields: Dict[str, np.ndarray], 
        mode_data: Optional[Dict] = None
    ) -> float:
        """
        Compute total free energy.
        
        Args:
            fields: dict mapping field names to n-d arrays
            mode_data: dict with precomputed mode info if needed
        
        Returns:
            scalar free energy value
        """
        pass
    
    def set_field_shape(self, shape: Tuple[int, ...]) -> None:
        """Called by simulation to inform model of spatial grid dimensions."""
        self._field_shape = shape
    
    def configure(self, config: Dict) -> None:
        """
        Accept configuration dictionary (e.g., parameters like temperature, etc.).
        Subclasses override to extract and store model parameters.
        """
        self._config = config
