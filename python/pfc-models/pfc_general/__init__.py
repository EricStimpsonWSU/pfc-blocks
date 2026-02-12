"""
PFC General Library

A refactored phase field crystal (PFC) simulation library with clean separation
of model definition, dynamics, and simulation concerns.

Supports:
- 1D/2D/3D density fields
- CPU (Numba) and GPU (custom kernels) backends
- Spectral and finite difference operators
- Configurable boundary conditions
- First and second-order dynamics
"""

__version__ = "0.1.0"

from .models.base_model import PFCModel
from .dynamics.base_dynamics import Dynamics
from .operators.base_ops import Operators
from .backends.base_backend import Backend
from .simulation.domain import Domain
from .simulation.runner import Simulation
from .simulation.initial_conditions import InitialConditions
from .compatibility import PFC2D_Vacancy, PFC2D_Vacancy_Parms, PFC2D_Standard, PFC2D_Standard_Parms

__all__ = [
    "PFCModel",
    "Dynamics",
    "Operators",
    "Backend",
    "Domain",
    "Simulation",
    "InitialConditions",
    "PFC2D_Vacancy",
    "PFC2D_Vacancy_Parms",
    "PFC2D_Standard",
    "PFC2D_Standard_Parms",
]
