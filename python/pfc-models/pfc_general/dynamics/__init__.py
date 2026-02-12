"""Dynamics (time evolution) definitions."""

from .base_dynamics import Dynamics

# Lazy import to avoid loading cupy dependencies
def __getattr__(name):
    """Lazy import of dynamics components."""
    if name == "FirstOrderDynamics":
        from .first_order import FirstOrderDynamics
        return FirstOrderDynamics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Dynamics", "FirstOrderDynamics"]
