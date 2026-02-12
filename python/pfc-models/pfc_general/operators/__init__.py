"""Spatial differential operators."""

from .base_ops import Operators

# Lazy import to avoid loading cupy dependencies
def __getattr__(name):
    """Lazy import of operators."""
    if name == "SpectralOperators2D":
        from .spectral_ops import SpectralOperators2D
        return SpectralOperators2D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Operators", "SpectralOperators2D"]
