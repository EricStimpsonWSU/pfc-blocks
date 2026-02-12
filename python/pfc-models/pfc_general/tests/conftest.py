"""
Pytest configuration and shared fixtures for PFC General tests.
"""

import pytest
import numpy as np
import cupy as cp
import sys
from pathlib import Path

# Add parent directory to path for imports
test_dir = Path(__file__).parent
pfc_general_dir = test_dir.parent
pfc_models_dir = pfc_general_dir.parent
sys.path.insert(0, str(pfc_models_dir))


@pytest.fixture
def small_domain():
    """Fixture: Small 2D domain for fast testing."""
    from pfc_general.simulation.domain import Domain
    return Domain(shape=(32, 32), box_size=(10.0, 10.0))


@pytest.fixture
def medium_domain():
    """Fixture: Medium 2D domain."""
    from pfc_general.simulation.domain import Domain
    return Domain(shape=(64, 64), box_size=(20.0, 20.0))


@pytest.fixture
def log_model():
    """Fixture: Standard log PFC model."""
    from pfc_general.models.free_energy.log import LogPFCModel2D
    return LogPFCModel2D(
        epsilon=-0.25,
        beta=1.0,
        g=0.0,
        v0=1.0,
        Hln=1.0,
        Hng=0.5,
        phi0=0.35
    )


@pytest.fixture
def spectral_ops(small_domain):
    """Fixture: Spectral operators configured for small domain."""
    from pfc_general.operators.spectral_ops import SpectralOperators2D
    ops = SpectralOperators2D()
    ops.configure({'domain': small_domain})
    return ops


@pytest.fixture
def first_order_dynamics():
    """Fixture: First-order dynamics with no noise."""
    from pfc_general.dynamics.first_order import FirstOrderDynamics
    return FirstOrderDynamics(noise_amplitude=0.0)


@pytest.fixture
def gpu_backend():
    """Fixture: GPU backend."""
    from pfc_general.backends.gpu_kernels.backend import GPUBackend
    return GPUBackend()


@pytest.fixture
def flat_ic():
    """Fixture: Flat noisy initial condition."""
    from pfc_general.simulation.ics import FlatNoisy
    return FlatNoisy(phi0=0.35, noise_amplitude=0.01, seed=42)


@pytest.fixture
def triangular_ic():
    """Fixture: Triangular lattice initial condition."""
    from pfc_general.simulation.ics import TriangularLattice
    return TriangularLattice(phi0=0.35, seed=42)
