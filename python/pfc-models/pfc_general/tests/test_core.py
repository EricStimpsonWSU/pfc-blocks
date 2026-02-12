"""
Unit tests for PFC General library core components.

Run with: pytest test_core.py -v
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


class TestDomain:
    """Test Domain class functionality."""
    
    def test_domain_creation_2d(self):
        """Test creating a 2D domain."""
        from pfc_general.simulation.domain import Domain
        
        domain = Domain(
            shape=(64, 64),
            box_size=(10.0, 10.0),
            dtype=np.float64,
            bc='periodic'
        )
        
        assert domain.ndim == 2
        assert domain.shape == (64, 64)
        assert domain.box_size == (10.0, 10.0)
        assert domain.bc == 'periodic'
        
    def test_domain_spacing(self):
        """Test grid spacing calculation."""
        from pfc_general.simulation.domain import Domain
        
        domain = Domain(shape=(100, 200), box_size=(10.0, 20.0))
        
        assert domain.spacing[0] == pytest.approx(0.1)
        assert domain.spacing[1] == pytest.approx(0.1)
    
    def test_domain_wavenumbers(self):
        """Test wavenumber generation."""
        from pfc_general.simulation.domain import Domain
        
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        kx, ky = domain.get_wavenumbers()
        
        assert len(kx) == 32
        assert len(ky) == 32
        assert kx[0] == pytest.approx(0.0)
        
    def test_domain_coordinates(self):
        """Test coordinate array generation."""
        from pfc_general.simulation.domain import Domain
        
        domain = Domain(shape=(16, 16), box_size=(8.0, 8.0))
        x, y = domain.get_coordinates()
        
        assert len(x) == 16
        assert len(y) == 16
        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(7.5)
        
    def test_domain_shape_boxsize_mismatch(self):
        """Test that mismatched dimensions raise error."""
        from pfc_general.simulation.domain import Domain
        
        with pytest.raises(ValueError, match="same dimension"):
            Domain(shape=(64, 64), box_size=(10.0,))


class TestLogPFCModel:
    """Test LogPFCModel2D class."""
    
    def test_model_creation(self):
        """Test creating a log PFC model."""
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
        assert float(model.epsilon) == pytest.approx(-0.25)
        assert float(model.beta) == pytest.approx(1.0)
        
    def test_functional_derivative(self):
        """Test functional derivative computation."""
        from pfc_general.models.free_energy.log import LogPFCModel2D
        
        model = LogPFCModel2D(
            epsilon=-0.25, beta=1.0, g=0.0, v0=1.0,
            Hln=1.0, Hng=0.5, phi0=0.35
        )
        
        model.set_field_shape((32, 32))
        
        # Create small test field
        phi = cp.random.randn(32, 32) * 0.1 + 0.35
        
        deriv = model.functional_derivative({'phi': phi})
        
        assert 'phi' in deriv
        assert deriv['phi'].shape == (32, 32)
        assert isinstance(deriv['phi'], cp.ndarray)
        
    def test_free_energy(self):
        """Test free energy computation."""
        from pfc_general.models.free_energy.log import LogPFCModel2D
        
        model = LogPFCModel2D(
            epsilon=-0.25, beta=1.0, g=0.0, v0=1.0,
            Hln=1.0, Hng=0.5, phi0=0.35
        )
        
        phi = cp.ones((32, 32)) * 0.35
        
        energy = model.free_energy({'phi': phi})
        
        assert isinstance(energy, float)
        assert not np.isnan(energy)
        assert not np.isinf(energy)


class TestSpectralOperators:
    """Test SpectralOperators2D class."""
    
    def test_operators_creation(self):
        """Test creating spectral operators."""
        from pfc_general.operators.spectral_ops import SpectralOperators2D
        from pfc_general.simulation.domain import Domain
        
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        operators = SpectralOperators2D(domain)
        
        assert operators.method == 'spectral'
        assert operators.k2 is not None
        
    def test_operators_configure(self):
        """Test operators configuration."""
        from pfc_general.operators.spectral_ops import SpectralOperators2D
        from pfc_general.simulation.domain import Domain
        
        operators = SpectralOperators2D()
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        
        operators.configure({'domain': domain})
        
        assert operators.k2 is not None
        assert operators.k2.shape == (32, 32)
        
    def test_laplacian(self):
        """Test Laplacian computation."""
        from pfc_general.operators.spectral_ops import SpectralOperators2D
        from pfc_general.simulation.domain import Domain
        
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        operators = SpectralOperators2D(domain)
        
        # Create simple test field
        phi = cp.random.randn(32, 32)
        
        lap = operators.laplacian(phi)
        
        assert lap.shape == phi.shape
        assert isinstance(lap, cp.ndarray)
        
    def test_gradient(self):
        """Test gradient computation."""
        from pfc_general.operators.spectral_ops import SpectralOperators2D
        from pfc_general.simulation.domain import Domain
        
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        operators = SpectralOperators2D(domain)
        
        phi = cp.random.randn(32, 32)
        
        grad_x = operators.gradient(phi, axis=0)
        grad_y = operators.gradient(phi, axis=1)
        
        assert grad_x.shape == phi.shape
        assert grad_y.shape == phi.shape
        
    def test_gradient_invalid_axis(self):
        """Test gradient with invalid axis raises error."""
        from pfc_general.operators.spectral_ops import SpectralOperators2D
        from pfc_general.simulation.domain import Domain
        
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        operators = SpectralOperators2D(domain)
        
        phi = cp.random.randn(32, 32)
        
        with pytest.raises(ValueError, match="axis must be"):
            operators.gradient(phi, axis=2)


class TestFirstOrderDynamics:
    """Test FirstOrderDynamics class."""
    
    def test_dynamics_creation(self):
        """Test creating first-order dynamics."""
        from pfc_general.dynamics.first_order import FirstOrderDynamics
        
        dynamics = FirstOrderDynamics(noise_amplitude=0.01)
        
        assert dynamics.order == 1
        assert dynamics.noise_amplitude == pytest.approx(0.01)

    def test_compute_fields_next(self):
        """Test timestep computation."""
        from pfc_general.dynamics.first_order import FirstOrderDynamics
        from pfc_general.models.free_energy.log import LogPFCModel2D
        from pfc_general.operators.spectral_ops import SpectralOperators2D
        from pfc_general.simulation.domain import Domain

        # Small system for fast test
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        operators = SpectralOperators2D(domain)

        model = LogPFCModel2D(
            epsilon=-0.25, beta=1.0, g=0.0, v0=1.0,
            Hln=1.0, Hng=0.5, phi0=0.35
        )
        model.set_field_shape(domain.shape)

        dynamics = FirstOrderDynamics(noise_amplitude=0.0)

        phi_init = cp.ones((32, 32)) * 0.35 + cp.random.randn(32, 32) * 0.01

        fields_next = dynamics.compute_fields_next(
            fields={'phi': phi_init},
            dt=0.01,
            operators=operators,
            model=model
        )

        assert 'phi' in fields_next
        assert fields_next['phi'].shape == (32, 32)
        assert not np.isnan(cp.asnumpy(fields_next['phi'])).any()


class TestGPUBackend:
    """Test GPUBackend class."""
    
    def test_backend_creation(self):
        """Test creating GPU backend."""
        from pfc_general.backends.gpu_kernels.backend import GPUBackend
        
        backend = GPUBackend(max_phi=5.0)
        
        assert backend.device == 'gpu'
        assert backend.max_phi == pytest.approx(5.0)
        
    def test_to_numpy(self):
        """Test GPU to CPU transfer."""
        from pfc_general.backends.gpu_kernels.backend import GPUBackend
        
        backend = GPUBackend()
        
        fields_gpu = {'phi': cp.ones((32, 32))}
        fields_cpu = backend.to_numpy(fields_gpu)
        
        assert isinstance(fields_cpu['phi'], np.ndarray)
        assert fields_cpu['phi'].shape == (32, 32)
        
    def test_from_numpy(self):
        """Test CPU to GPU transfer."""
        from pfc_general.backends.gpu_kernels.backend import GPUBackend
        
        backend = GPUBackend()
        
        fields_cpu = {'phi': np.ones((32, 32))}
        fields_gpu = backend.from_numpy(fields_cpu)
        
        assert isinstance(fields_gpu['phi'], cp.ndarray)
        assert fields_gpu['phi'].shape == (32, 32)


class TestInitialConditions:
    """Test initial condition generators."""
    
    def test_flat_noisy(self):
        """Test FlatNoisy initial condition."""
        from pfc_general.simulation.ics import FlatNoisy
        from pfc_general.simulation.domain import Domain
        
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        ic = FlatNoisy(phi0=0.35, noise_amplitude=0.01, seed=42)
        
        fields = ic(domain)
        
        assert 'phi' in fields
        assert fields['phi'].shape == (32, 32)
        
        # Check average is close to phi0
        phi_mean = float(cp.mean(fields['phi']))
        assert phi_mean == pytest.approx(0.35, abs=0.01)
        
    def test_triangular_lattice(self):
        """Test TriangularLattice initial condition."""
        from pfc_general.simulation.ics import TriangularLattice
        from pfc_general.simulation.domain import Domain
        
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        ic = TriangularLattice(phi0=0.35, amplitude=0.1, seed=42)
        
        fields = ic(domain)
        
        assert 'phi' in fields
        assert fields['phi'].shape == (32, 32)
        
        # Check average is close to phi0
        phi_mean = float(cp.mean(fields['phi']))
        assert phi_mean == pytest.approx(0.35, abs=0.01)
        
        # Check there is structure (not flat)
        phi_std = float(cp.std(fields['phi']))
        assert phi_std > 0.05


class TestSimulation:
    """Test Simulation class."""
    
    def test_simulation_creation(self):
        """Test creating a simulation."""
        from pfc_general.simulation import Domain, Simulation
        from pfc_general.simulation.ics import FlatNoisy
        from pfc_general.models.free_energy.log import LogPFCModel2D
        from pfc_general.dynamics.first_order import FirstOrderDynamics
        from pfc_general.operators.spectral_ops import SpectralOperators2D
        from pfc_general.backends.gpu_kernels.backend import GPUBackend
        
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        model = LogPFCModel2D(
            epsilon=-0.25, beta=1.0, g=0.0, v0=1.0,
            Hln=1.0, Hng=0.5, phi0=0.35
        )
        operators = SpectralOperators2D()
        operators.configure({'domain': domain})
        dynamics = FirstOrderDynamics(noise_amplitude=0.0)
        backend = GPUBackend()
        ic = FlatNoisy(phi0=0.35, seed=42)
        
        sim = Simulation(domain, model, dynamics, backend, operators, ic)
        
        assert sim.current_step == 0
        assert sim.current_time == pytest.approx(0.0)
        assert sim.fields is not None
        
    def test_simulation_run(self):
        """Test running simulation for a few steps."""
        from pfc_general.simulation import Domain, Simulation
        from pfc_general.simulation.ics import FlatNoisy
        from pfc_general.models.free_energy.log import LogPFCModel2D
        from pfc_general.dynamics.first_order import FirstOrderDynamics
        from pfc_general.operators.spectral_ops import SpectralOperators2D
        from pfc_general.backends.gpu_kernels.backend import GPUBackend
        
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        model = LogPFCModel2D(
            epsilon=-0.25, beta=1.0, g=0.0, v0=1.0,
            Hln=1.0, Hng=0.5, phi0=0.35
        )
        operators = SpectralOperators2D()
        operators.configure({'domain': domain})
        dynamics = FirstOrderDynamics(noise_amplitude=0.0)
        backend = GPUBackend()
        ic = FlatNoisy(phi0=0.35, noise_amplitude=0.01, seed=42)
        
        sim = Simulation(domain, model, dynamics, backend, operators, ic)
        
        # Run for 10 steps
        sim.run(num_steps=10, dt=0.01)
        
        assert sim.current_step == 10
        assert sim.current_time == pytest.approx(0.1)
        
    def test_simulation_checkpoint(self, tmp_path):
        """Test checkpoint save/load."""
        from pfc_general.simulation import Domain, Simulation
        from pfc_general.simulation.ics import FlatNoisy
        from pfc_general.models.free_energy.log import LogPFCModel2D
        from pfc_general.dynamics.first_order import FirstOrderDynamics
        from pfc_general.operators.spectral_ops import SpectralOperators2D
        from pfc_general.backends.gpu_kernels.backend import GPUBackend
        
        domain = Domain(shape=(32, 32), box_size=(10.0, 10.0))
        model = LogPFCModel2D(
            epsilon=-0.25, beta=1.0, g=0.0, v0=1.0,
            Hln=1.0, Hng=0.5, phi0=0.35
        )
        operators = SpectralOperators2D()
        operators.configure({'domain': domain})
        dynamics = FirstOrderDynamics(noise_amplitude=0.0)
        backend = GPUBackend()
        ic = FlatNoisy(phi0=0.35, seed=42)
        
        sim = Simulation(domain, model, dynamics, backend, operators, ic)
        sim.run(num_steps=5, dt=0.01)
        
        # Save checkpoint
        checkpoint_file = tmp_path / "test_checkpoint.npz"
        sim.save_checkpoint(str(checkpoint_file))
        
        assert checkpoint_file.exists()
        
        # Create new simulation and load checkpoint
        sim2 = Simulation(domain, model, dynamics, backend, operators, ic)
        sim2.load_checkpoint(str(checkpoint_file))
        
        assert sim2.current_step == 5
        assert sim2.current_time == pytest.approx(0.05)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_small(self):
        """Test complete workflow with small system."""
        from pfc_general.simulation import Domain, Simulation
        from pfc_general.simulation.ics import TriangularLattice
        from pfc_general.models.free_energy.log import LogPFCModel2D
        from pfc_general.dynamics.first_order import FirstOrderDynamics
        from pfc_general.operators.spectral_ops import SpectralOperators2D
        from pfc_general.backends.gpu_kernels.backend import GPUBackend

        # Setup (mirrors beta-log-crystal workflow, scaled down)
        n = 64
        ppu = 30
        dt = 0.01

        mx = int(np.round(n / ppu))
        my = int(np.round((2 * n / (np.sqrt(3) * ppu)) / 2) * 2)
        lx = 4 * np.pi * mx / np.sqrt(3)
        ly = 2 * np.pi * my
        domain = Domain(shape=(n, n), box_size=(lx, ly))

        model = LogPFCModel2D(
            epsilon=-1.6,
            beta=1.0,
            g=-1.0,
            v0=1.0,
            Hln=1.0,
            Hng=0.0,
            a=0.0,
            phi0=0.6,
            min_log=-12.0,
        )
        operators = SpectralOperators2D()
        operators.configure({'domain': domain})
        dynamics = FirstOrderDynamics(noise_amplitude=0.0)
        backend = GPUBackend(max_phi=5.0)
        ic = TriangularLattice(phi0=0.6, amplitude=None, noise_amplitude=0.1, seed=123)

        sim = Simulation(domain, model, dynamics, backend, operators, ic)

        # Run a few steps with adaptive sub-stepping like the notebook
        energies = []

        for _ in range(5):
            success = False
            for factor in (1, 2, 4, 8, 16):
                sub_dt = dt / factor
                try:
                    for _ in range(factor):
                        sim.fields = backend.timestep(sim.fields, sub_dt, model, dynamics, operators)
                    success = True
                    break
                except RuntimeError:
                    continue

            if not success:
                raise RuntimeError("Adaptive stepping failed in test.")

            sim.current_step += 1
            sim.current_time += dt

            energy = model.free_energy({'phi': sim.fields['phi']})
            energies.append(float(energy))

        # Verify
        assert len(energies) == 5
        assert all(not np.isnan(e) for e in energies)
        assert all(not np.isinf(e) for e in energies)

        # Energy should be stable or decreasing (gradient flow)
        # Allow some tolerance for numerical noise
        assert energies[-1] <= energies[0] + 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
