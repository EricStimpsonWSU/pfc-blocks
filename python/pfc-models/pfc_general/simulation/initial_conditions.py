"""Initial condition generators for fields."""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from .domain import Domain


class InitialConditions(ABC):
    """
    Abstract base for initial condition generators.
    
    Generates initial field configurations.
    """
    
    @abstractmethod
    def __call__(self, domain: Domain) -> Dict[str, np.ndarray]:
        """
        Generate initial fields.
        
        Args:
            domain: Domain instance (shape, spacing, etc.)
        
        Returns:
            dict mapping field names to n-d arrays (initialized to domain shape)
        """
        pass
