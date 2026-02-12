"""Simulation orchestration and utilities."""

from .domain import Domain

# Lazy imports to avoid loading cupy dependencies
def __getattr__(name):
    """Lazy import of simulation components."""
    if name == "Simulation":
        from .runner import Simulation
        return Simulation
    elif name == "InitialConditions":
        from .initial_conditions import InitialConditions
        return InitialConditions
    elif name == "FlatNoisy":
        from .ics import FlatNoisy
        return FlatNoisy
    elif name == "TriangularLattice":
        from .ics import TriangularLattice
        return TriangularLattice
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Domain", "Simulation", "InitialConditions", "FlatNoisy", "TriangularLattice"]
