"""Test specialized 2D PBC GPU kernel backend.

The GPUBackendPBC2DKernel provides optimized computation for 2D PBC systems
using CuPy, with support for Log PFC and variants.
"""

import numpy as np
import cupy as cp
import pytest
from pathlib import Path

import sys
pfc_models_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(pfc_models_path))

from pfc_general.backends import GPUBackendPBC2DKernel
from pfc_general.models.free_energy.log import LogPFCModel2D
from pfc_general.operators.spectral_ops import SpectralOperators2D
from pfc_general.simulation.domain import Domain
from pfc_general.dynamics.first_order import FirstOrderDynamics


@pytest.mark.gpu
class TestGPUBackendPBC2DKernel:
    """Test suite for GPUBackendPBC2DKernel."""
    
    @pytest.fixture
    def setup(self):
        """Set up common test fixtures."""
        # Domain
        domain = Domain(
            shape=(64, 64),
            box_size=(16.0, 16.0),
            dtype=np.float64,
            bc='periodic'
        )
        
        # Model parameters
        model = LogPFCModel2D(
            epsilon=-0.25,
            beta=1.0,
            g=0.5,
            v0=0.2,
            Hln=0.25,
            Hng=0.0,
            a=0.0,
            phi0=0.5,
            min_log=-10.0
        )
        
        # Operators
        operators = SpectralOperators2D(domain)
        
        # Dynamics
        dynamics = FirstOrderDynamics(noise_amplitude=0.0)
        
        # Backend
        backend = GPUBackendPBC2DKernel(max_phi=5.0, precompute_buffers=True)
        
        return {
            'domain': domain,
            'model': model,
            'operators': operators,
            'dynamics': dynamics,
            'backend': backend
        }
    
    def test_backend_initialization(self, setup):
        """Test backend initializes correctly."""
        backend = setup['backend']
        assert backend.device == 'gpu'
        assert backend._kernel is not None
    
    def test_nonlinear_term_output_shape(self, setup):
        """Test nonlinear term has correct shape."""
        domain = setup['domain']
        model = setup['model']
        operators = setup['operators']
        backend = setup['backend']
        
        # Create test field
        phi = cp.ones(domain.shape, dtype=cp.float64) * 0.5
        fields = {'phi': phi}
        
        # Compute nonlinear term
        result = backend.nonlinear_term(fields, model, operators)
        
        assert 'phi' in result
        assert result['phi'].shape == phi.shape
        assert isinstance(result['phi'], cp.ndarray)
    
    def test_nonlinear_term_values_valid(self, setup):
        """Test nonlinear term values are finite and reasonable."""
        domain = setup['domain']
        model = setup['model']
        operators = setup['operators']
        backend = setup['backend']
        
        # Create test field with variation
        phi = cp.ones(domain.shape, dtype=cp.float64) * 0.3
        phi[10:20, 10:20] = 0.8
        fields = {'phi': phi}
        
        # Compute nonlinear term
        result = backend.nonlinear_term(fields, model, operators)
        N = result['phi']
        
        # Check for NaNs or Infs (convert to numpy for safety)
        N_np = cp.asnumpy(N)
        assert not np.isnan(N_np).any()
        assert not np.isinf(N_np).any()
        
        # Check values are within reasonable range
        assert float(np.min(N_np)) > -100.0
        assert float(np.max(N_np)) < 100.0
    
    def test_precomputed_buffers(self, setup):
        """Test precomputed buffers are accessible."""
        domain = setup['domain']
        model = setup['model']
        operators = setup['operators']
        backend = setup['backend']
        
        phi = cp.ones(domain.shape, dtype=cp.float64) * 0.5
        fields = {'phi': phi}
        
        # Compute with buffers
        backend.nonlinear_term(fields, model, operators)
        
        # Check buffers exist
        assert 'phi2' in backend._buffer
        assert 'phi3' in backend._buffer
        assert 'phivac' in backend._buffer
        assert 'philn' in backend._buffer
        
        # Verify values
        expected_phi2 = 0.5 * 0.5
        actual_phi2 = float(cp.asnumpy(backend._buffer['phi2'][0, 0]))
        assert np.isclose(actual_phi2, expected_phi2)
    
    def test_timestep_with_kernel_integration(self, setup):
        """Test that timestep uses the custom kernel (not vanilla model)."""
        domain = setup['domain']
        model = setup['model']
        operators = setup['operators']
        dynamics = setup['dynamics']
        backend_kernel = setup['backend']
        
        # Also create vanilla backend for comparison
        from pfc_general.backends.gpu_kernels.backend import GPUBackend
        backend_vanilla = GPUBackend(max_phi=5.0)
        
        # Same initial field
        phi = cp.ones(domain.shape, dtype=cp.float64) * 0.35
        phi[10:20, 10:20] += 0.1  # Add some variation
        fields = {'phi': cp.array(phi)}
        
        dt = 0.01
        
        # Kernel backend timestep
        fields_kernel = backend_kernel.timestep(
            {'phi': cp.array(phi)}, dt, model, dynamics, operators, noise_amplitude=0.0
        )
        
        # Vanilla backend timestep (should use different nonlinear path)
        fields_vanilla = backend_vanilla.timestep(
            {'phi': cp.array(phi)}, dt, model, dynamics, operators, noise_amplitude=0.0
        )
        
        # Results should be very close (both correct implementations)
        # but may differ slightly due to numerical precision and minor
        # differences in computation order (kernel vs pure Python ops)
        diff = float(cp.mean(cp.abs(fields_kernel['phi'] - fields_vanilla['phi'])))
        assert diff < 1e-3, f"Kernel and vanilla backends differ by {diff}"
    
    def test_field_transfer_to_from_numpy(self, setup):
        """Test CPU<->GPU field transfers."""
        backend = setup['backend']
        
        # Create CPU arrays
        phi_cpu = np.random.randn(64, 64).astype(np.float64)
        fields_cpu = {'phi': phi_cpu}
        
        # Transfer to GPU
        fields_gpu = backend.from_numpy(fields_cpu)
        assert isinstance(fields_gpu['phi'], cp.ndarray)
        
        # Transfer back to CPU
        fields_cpu2 = backend.to_numpy(fields_gpu)
        assert isinstance(fields_cpu2['phi'], np.ndarray)
        
        # Check values match
        assert np.allclose(fields_cpu['phi'], fields_cpu2['phi'])
    
    def test_large_phi_safety_check(self, setup, capsys):
        """Test that large phi values trigger safety warning."""
        domain = setup['domain']
        model = setup['model']
        operators = setup['operators']
        dynamics = setup['dynamics']
        backend = setup['backend']
        
        # Create field with value exceeding max_phi
        phi = cp.ones(domain.shape, dtype=cp.float64) * 10.0  # > max_phi=5.0
        fields = {'phi': phi}
        
        dt = 0.01
        
        # Should print a warning but not raise error
        backend.timestep(
            fields, dt, model, dynamics, operators, noise_amplitude=0.0
        )
        
        # Verify warning was printed
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "exceeded max value" in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
