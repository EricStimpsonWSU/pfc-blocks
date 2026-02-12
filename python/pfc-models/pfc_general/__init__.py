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

# Lazy imports to avoid loading GPU dependencies unless needed
def __getattr__(name):
    """Lazy import of pfc_general components."""
    if name == "PFCModel":
        from .models.base_model import PFCModel
        return PFCModel
    elif name == "Dynamics":
        from .dynamics.base_dynamics import Dynamics
        return Dynamics
    elif name == "Operators":
        from .operators.base_ops import Operators
        return Operators
    elif name == "Backend":
        from .backends.base_backend import Backend
        return Backend
    elif name == "Domain":
        from .simulation.domain import Domain
        return Domain
    elif name == "Simulation":
        from .simulation.runner import Simulation
        return Simulation
    elif name == "InitialConditions":
        from .simulation.initial_conditions import InitialConditions
        return InitialConditions
    elif name == "PFC2D_Vacancy":
        from .compatibility import PFC2D_Vacancy
        return PFC2D_Vacancy
    elif name == "PFC2D_Vacancy_Parms":
        from .compatibility import PFC2D_Vacancy_Parms
        return PFC2D_Vacancy_Parms
    elif name == "PFC2D_Standard":
        from .compatibility import PFC2D_Standard
        return PFC2D_Standard
    elif name == "PFC2D_Standard_Parms":
        from .compatibility import PFC2D_Standard_Parms
        return PFC2D_Standard_Parms
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
