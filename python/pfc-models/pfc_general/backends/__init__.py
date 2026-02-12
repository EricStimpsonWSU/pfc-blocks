"""Computational backends for CPU and GPU."""

from .base_backend import Backend
from .cpu.backend import CPUBackend

# Lazy import for GPU backend to avoid cupy dependency unless needed
def __getattr__(name):
    """Lazy import of GPU backend."""
    if name == "GPUBackendPBC2DKernel":
        from .gpu_kernels import GPUBackendPBC2DKernel
        return GPUBackendPBC2DKernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Backend", "CPUBackend", "GPUBackendPBC2DKernel"]
