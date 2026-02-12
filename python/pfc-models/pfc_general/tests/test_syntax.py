"""
Syntax and import tests - can run without GPU.

These tests check that files import correctly and have proper syntax.
Run with: pytest test_syntax.py -v
"""

import pytest
import sys
from pathlib import Path

# Check if cupy is available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except (ImportError, Exception):
    CUPY_AVAILABLE = False

# Add parent directory to path for imports
test_dir = Path(__file__).parent
pfc_general_dir = test_dir.parent
pfc_models_dir = pfc_general_dir.parent
sys.path.insert(0, str(pfc_models_dir))


def test_import_domain():
    """Test importing Domain class."""
    from pfc_general.simulation.domain import Domain
    assert Domain is not None


def test_import_base_model():
    """Test importing PFCModel base class."""
    from pfc_general.models.base_model import PFCModel
    assert PFCModel is not None


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_import_log_model():
    """Test importing LogPFCModel2D."""
    from pfc_general.models.free_energy.log import LogPFCModel2D
    assert LogPFCModel2D is not None


def test_import_base_operators():
    """Test importing Operators base class."""
    from pfc_general.operators.base_ops import Operators
    assert Operators is not None


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_import_spectral_operators():
    """Test importing SpectralOperators2D."""
    from pfc_general.operators.spectral_ops import SpectralOperators2D
    assert SpectralOperators2D is not None


def test_import_base_dynamics():
    """Test importing Dynamics base class."""
    from pfc_general.dynamics.base_dynamics import Dynamics
    assert Dynamics is not None


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_import_first_order_dynamics():
    """Test importing FirstOrderDynamics."""
    from pfc_general.dynamics.first_order import FirstOrderDynamics
    assert FirstOrderDynamics is not None


def test_import_base_backend():
    """Test importing Backend base class."""
    from pfc_general.backends.base_backend import Backend
    assert Backend is not None


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_import_gpu_backend():
    """Test importing GPUBackend."""
    from pfc_general.backends.gpu_kernels.backend import GPUBackend
    assert GPUBackend is not None


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_import_ics():
    """Test importing initial condition classes."""
    from pfc_general.simulation.ics import FlatNoisy, TriangularLattice
    assert FlatNoisy is not None
    assert TriangularLattice is not None


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_import_simulation():
    """Test importing Simulation class."""
    from pfc_general.simulation import Simulation
    assert Simulation is not None


def test_domain_instantiation():
    """Test creating Domain instance."""
    from pfc_general.simulation.domain import Domain
    
    domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
    
    assert domain.ndim == 2
    assert domain.shape == (32, 32)
    assert domain.box_size == (10.0, 10.0)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_log_model_instantiation():
    """Test creating LogPFCModel2D instance."""
    from pfc_general.models.free_energy.log import LogPFCModel2D
    
    model = LogPFCModel2D(
        epsilon=-0.25,
        beta=1.0,
        g=0.0,
        v0=1.0,
        Hln=1.0,
        Hng=0.5,
        phi0=0.35
    )
    
    assert model.num_fields == 1


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_spectral_operators_instantiation():
    """Test creating SpectralOperators2D instance."""
    from pfc_general.operators.spectral_ops import SpectralOperators2D
    
    ops = SpectralOperators2D()
    
    assert ops.method == 'spectral'


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_first_order_dynamics_instantiation():
    """Test creating FirstOrderDynamics instance."""
    from pfc_general.dynamics.first_order import FirstOrderDynamics
    
    dynamics = FirstOrderDynamics(noise_amplitude=0.01)
    
    assert dynamics.order == 1
    assert dynamics.noise_amplitude == 0.01


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_gpu_backend_instantiation():
    """Test creating GPUBackend instance."""
    from pfc_general.backends.gpu_kernels.backend import GPUBackend
    
    backend = GPUBackend()
    
    assert backend.device == 'gpu'


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_flat_noisy_instantiation():
    """Test creating FlatNoisy IC."""
    from pfc_general.simulation.ics import FlatNoisy
    
    ic = FlatNoisy(phi0=0.35, seed=42)
    
    assert callable(ic)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_triangular_lattice_instantiation():
    """Test creating TriangularLattice IC."""
    from pfc_general.simulation.ics import TriangularLattice
    
    ic = TriangularLattice(phi0=0.35, seed=42)
    
    assert callable(ic)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
