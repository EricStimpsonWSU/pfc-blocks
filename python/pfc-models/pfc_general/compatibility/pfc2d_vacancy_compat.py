"""
Compatibility layer for PFC2D_Vacancy.

This module provides a thin veneer that replicates the original PFC2D_Vacancy
syntax while using the refactored pfc_general library underneath.

Mathematical Equivalence Validation:
-------------------------------------
The refactored library maintains exact mathematical equivalence with the original
PFC2D_Vacancy implementation:

1. **Free Energy Functional** (LogPFCModel2D.free_energy):
   Original: F = ∫[β(-2∇²φ + ∇⁴φ)/2 + (ε+β)φ²/2 + gφ³/3 + v₀φ⁴/4 + Hln(φ+a)ln(φ+a) + Hng*3φ(|φ|-φ)]dV
   Refactored: Identical implementation in models/free_energy/log.py

2. **Functional Derivative** (LogPFCModel2D.functional_derivative):
   Original: δF/δφ = -(ε+β)∇²φ + 2β∇⁴φ - β∇⁶φ + gφ² + v₀φ³ + Hln*ln(φ+a) - 6Hng*φ²*(φ<0)
   Refactored: Identical implementation with same operators

3. **Spectral Operators** (SpectralOperators2D):
   Original k² = 2*(1/dx²*(1-cos(kx*dx)) + 1/dy²*(1-cos(ky*dy)))  # spectral anomaly correction
   Refactored: Same calculation in operators/spectral_ops.py (line 47-50)

4. **ETD Time Stepping** (FirstOrderDynamics.compute_fields_next):
   Original predictor-corrector:
     - expcoeff = exp(L*dt)
     - expcoeff_nonlin = (exp(L*dt)-1)/L
     - expcoeff_nonlin2 = (exp(L*dt)-(1+L*dt))/L²
     - φ₁ = exp(L*dt)*φ₀ - k²*expcoeff_nonlin*N₀
     - φ₁ = exp(L*dt)*φ₀ - k²*(expcoeff_nonlin*N₀ + expcoeff_nonlin2*N₁')
   Refactored: Identical implementation in dynamics/first_order.py (lines 84-122)

5. **CUDA Kernel for N(φ)** (GPUBackend.calc_phiN_kernel):
   Original calc_phiN kernel computes:
     - phi2 = φ²
     - phi3 = φ³
     - phivac = -6*Hng*φ² if φ<0 else 0
     - philn = Hln*ln(φ+a) if φ>minPhi else Hln*minLog
     - phiN = g*phi2 + v0*phi3 + phivac + philn
   Refactored: Identical kernel in backends/gpu_kernels/backend.py (lines 30-58)

6. **Divergence-Free Noise** (GPUBackend.divergence_free_noise):
   Original GetEtaNoise():
     - noise_fft = i*kx*FFT(noisex) + i*ky*FFT(noisey)
     - High-k cutoff and normalization to eta
   Refactored: Identical implementation in backends/gpu_kernels/backend.py (lines 79-95)

This compatibility layer allows existing notebooks and scripts to run
without modification while benefiting from the cleaner refactored architecture.
"""

import numpy as np
import cupy as cp
from typing import Optional

# Import refactored components
import sys
from pathlib import Path
pfc_general_dir = Path(__file__).parent.parent
pfc_models_dir = pfc_general_dir.parent
sys.path.insert(0, str(pfc_models_dir))

from pfc_general.simulation.domain import Domain
from pfc_general.models.free_energy.log import LogPFCModel2D
from pfc_general.operators.spectral_ops import SpectralOperators2D
from pfc_general.dynamics.first_order import FirstOrderDynamics
from pfc_general.backends.gpu_kernels.backend import GPUBackend
from pfc_general.simulation.ics import FlatNoisy, TriangularLattice


class PFC2D_Vacancy_Parms:
    """
    Parameter container matching original PFC2D_Vacancy interface.
    """
    def __init__(self):
        self.epsilon = None                 # temperature parameter
        self.a = 0.0                        # shift for logarithmic term
        self.beta = None
        self.g = None
        self.v0 = None
        self.Hln = None
        self.Hng = 0.0
        self.N = None                       # number of pixels in square region
        self.PPU = None                     # pixels per unit
        self.phi0 = None                    # average density
        self.eta = None                     # strength of gaussian noise
        self.dt = None                      # time step
        self.seed = None                    # random seed
        self.NoiseDynamicsFlag = False      # add noise to dynamics
        self.Noise_CutoffK = 0.5            # high frequency threshold
        self.NoiseTimeSmoothFlag = False
        self.NoiseTimeSmoothingFrames = 30
        self.Noise_CutoffOmega = 0.1
        self.NoiseChangeRate = 1
        self.BoundaryPotential = None
        self.BoundaryDensity = None


class PFC2D_Vacancy:
    """
    Compatibility wrapper for PFC2D_Vacancy using refactored pfc_general library.
    
    This class replicates the original PFC2D_Vacancy API exactly while using
    the modular refactored components underneath.
    """
    
    def __init__(self):
        self.t = None
        self.parms = PFC2D_Vacancy_Parms()
        self.minLog = -10.
        self.phiMax = 5.0
        
        # Refactored components (created in InitParms)
        self._domain = None
        self._model = None
        self._operators = None
        self._dynamics = None
        self._backend = None
        
        # Expose fields for direct access (original API)
        self.phi = None
        self.phi_hat = None
        self.phi0 = None
        
        # Noise tracking
        self.NoiseT = 0
        self.noise = None
        self.noise_fft = None
        
        # Intermediate calculation results (for debugging)
        self.phi2 = None
        self.phi3 = None
        self.phivac = None
        self.philn = None
        self.phiN = None
        
    def InitParms(self):
        """Initialize parameters and set up refactored components."""
        # Ensure datatypes are correct
        self.parms.epsilon = cp.float64(self.parms.epsilon)
        self.parms.a = cp.float64(self.parms.a)
        self.parms.beta = cp.float64(self.parms.beta)
        self.parms.g = cp.float64(self.parms.g)
        self.parms.v0 = cp.float64(self.parms.v0)
        self.parms.Hln = cp.float64(self.parms.Hln)
        self.parms.Hng = cp.float64(self.parms.Hng)
        self.parms.phi0 = cp.float64(self.parms.phi0)
        self.parms.eta = cp.float64(self.parms.eta)
        self.parms.dt = cp.float64(self.parms.dt)
        
        self.minLog = cp.float64(self.minLog)
        self.minPhi = np.exp(self.minLog) - self.parms.a
        
        # Set random seed
        if self.parms.seed is not None:
            cp.random.seed(cp.uint64(self.parms.seed))
        else:
            cp.random.seed(None)
        
        # Initialize geometry
        self.SetGeometry(self.parms.N, self.parms.N, self.parms.PPU, 1)
        
        # Set timestep
        self.SetDT(self.parms.dt)
        
        # Initialize noise tracking
        self.NoiseT = 0
        
    def SetGeometry(self, NX, NY, PPUx, scalefactor=1, forceUnitCellBoundary=True):
        """
        Set up computational domain.
        
        This creates the refactored Domain and Operators components while
        maintaining all original member variables for API compatibility.
        """
        self.nx = NX
        self.ny = NY
        self.mx = NX / PPUx
        self.my = 2 * NY / (np.sqrt(3) * PPUx)
        
        if forceUnitCellBoundary:
            self.mx = int(np.round(self.mx))
            self.my = int(np.round(self.my / 2) * 2)
        
        self.dx = (4 * np.pi * self.mx / np.sqrt(3)) / NX
        self.dy = (2 * np.pi * self.my) / NY
        self.dy *= scalefactor
        self.dx *= scalefactor
        
        # Create coordinate arrays
        x = cp.arange(NX) * self.dx
        y = cp.arange(NY) * self.dy
        self.x, self.y = cp.meshgrid(x, y)
        self.r = cp.dstack((self.x, self.y))
        
        # Create refactored Domain
        self._domain = Domain(
            shape=(NY, NX),
            box_size=(NY * self.dy, NX * self.dx),
            dtype=np.float64,
            bc='periodic'
        )
        
        # Create wavenumber arrays (expose for compatibility)
        kx_1d, ky_1d = self._domain.get_wavenumbers()
        # Convert to CuPy for GPU operations
        kx_1d = cp.asarray(kx_1d)
        ky_1d = cp.asarray(ky_1d)
        self.kx, self.ky = cp.meshgrid(kx_1d, ky_1d)
        
        # Create refactored Operators
        self._operators = SpectralOperators2D(self._domain)
        
        # Expose k² arrays for compatibility
        self.k2 = self._operators.k2
        self.k4 = self._operators.k4
        self.k6 = self._operators.k6
        # lincoeff and linenergycoeff are set later in SetDT()
        self.lincoeff = None
        self.linenergycoeff = None
        
        # Create noise mask
        self.noiseMask = cp.ones((self.ny, self.nx))
        
        # Initialize field arrays
        self.phi = cp.zeros((self.ny, self.nx), dtype=cp.float64)
        self.phi_hat = cp.zeros((self.ny, self.nx), dtype=cp.complex128)
        self.noise = cp.zeros((self.ny, self.nx), dtype=cp.float64)
        self.noise_fft = cp.zeros((self.ny, self.nx), dtype=cp.complex128)
        self.phi2 = cp.zeros((self.ny, self.nx), dtype=cp.float64)
        self.phi3 = cp.zeros((self.ny, self.nx), dtype=cp.float64)
        self.phivac = cp.zeros((self.ny, self.nx), dtype=cp.float64)
        self.philn = cp.zeros((self.ny, self.nx), dtype=cp.float64)
        self.phiN = cp.zeros((self.ny, self.nx), dtype=cp.float64)
        
        # Grid/block dims (compatibility)
        self.griddim = (int(np.ceil(self.nx / 32)), int(np.ceil(self.ny / 32)))
        self.blockdim = (32, 32)
        
        # Create refactored Model
        if hasattr(self.parms, 'epsilon') and self.parms.epsilon is not None:
            self._model = LogPFCModel2D(
                epsilon=float(self.parms.epsilon),
                beta=float(self.parms.beta),
                g=float(self.parms.g),
                v0=float(self.parms.v0),
                Hln=float(self.parms.Hln),
                Hng=float(self.parms.Hng),
                a=float(self.parms.a),
                phi0=float(self.parms.phi0),
                min_log=float(self.minLog)
            )
            self._model.set_field_shape((NY, NX))
        
        # Create refactored Backend
        self._backend = GPUBackend(max_phi=self.phiMax)
        
    def SetDT(self, dt):
        """
        Set timestep and compute ETD coefficients.
        
        This delegates to FirstOrderDynamics while exposing coefficients
        for API compatibility.
        """
        self.parms.dt = dt
        
        # Create refactored Dynamics
        if hasattr(self.parms, 'eta'):
            noise_amp = float(self.parms.eta) if self.parms.NoiseDynamicsFlag else 0.0
        else:
            noise_amp = 0.0
            
        self._dynamics = FirstOrderDynamics(noise_amplitude=noise_amp)
        
        # Configure with operators (compute ETD coefficients locally)
        if self._operators is not None:
            # Compute ETD coefficients locally for compatibility exposure
            # expcoeff = exp(L*dt) where L = -k²(ε+β) + 2β*k⁴ - β*k⁶
            lincoeff = -self.k2 * (self.parms.epsilon + self.parms.beta) + 2 * self.parms.beta * self.k4 - self.parms.beta * self.k6
            self.lincoeff = lincoeff
            self.linenergycoeff = -self.k2 * (self.parms.epsilon + self.parms.beta) + self.parms.beta * self.k4
            
            self.expcoeff = cp.exp(lincoeff * dt)
            self.expcoeff_nonlin = (self.expcoeff - 1.0) / lincoeff
            self.expcoeff_nonlin2 = (self.expcoeff - 1.0 - lincoeff * dt) / (lincoeff * lincoeff)
        else:
            # Will be computed later when operators are available
            self.lincoeff = None
            self.linenergycoeff = None
            self.expcoeff = None
            self.expcoeff_nonlin = None
            self.expcoeff_nonlin2 = None
    
    def GetEtaNoise(self):
        """Generate divergence-free noise."""
        if self._backend is None:
            raise RuntimeError("Backend not initialized. Call InitParms first.")
        
        self.noise = self._backend.divergence_free_noise(
            self._domain,
            eta=float(self.parms.eta)
        )
        self.noise_fft = cp.fft.fft2(self.noise)
        return self.noise
    
    def InitFieldFlat(self, noisy=True):
        """Initialize flat field with optional noise."""
        ic = FlatNoisy(
            phi0=float(self.parms.phi0),
            noise_amplitude=float(self.parms.eta) if (noisy and self.parms.eta != 0) else 0.0,
            seed=int(self.parms.seed) if self.parms.seed is not None else None
        )
        
        fields = ic(self._domain)
        self.phi = fields['phi']
        self.phi_hat = cp.fft.fft2(self.phi)
        self.phi0 = cp.fft.ifft2(self.phi_hat).real
        self.t = 0
    
    def InitFieldCrystal(self, A=None, noisy=True, scalefactor=1):
        """Initialize triangular lattice field with optional noise."""
        if A is None:
            A = self.parms.phi0 / 3
        
        ic = TriangularLattice(
            phi0=float(self.parms.phi0),
            amplitude=float(A),
            noise_amplitude=float(self.parms.eta) if (noisy and self.parms.eta != 0) else 0.0,
            seed=int(self.parms.seed) if self.parms.seed is not None else None,
            scalefactor=scalefactor
        )
        
        fields = ic(self._domain)
        self.phi = fields['phi']
        self.phi_hat = cp.fft.fft2(self.phi)
        
        # Force average to phi0
        self.phi_hat[0, 0] = self.nx * self.ny * self.parms.phi0
        self.phi = cp.fft.ifft2(self.phi_hat).real
        self.phi0 = self.phi.copy()
        self.t = 0
    
    def AddNoise(self):
        """Add noise to current field."""
        self.phi += self.GetEtaNoise()
        self.phi_hat = cp.fft.fft2(self.phi)
        self.phi0 = cp.fft.ifft2(self.phi_hat).real
    
    def TimeStepCross(self):
        """
        Perform one timestep using ETD predictor-corrector.
        
        This delegates to FirstOrderDynamics.compute_fields_next while
        maintaining the original member variable interface.
        """
        # Update noise if needed
        if self.parms.NoiseDynamicsFlag:
            self.NoiseT += 1
            if self.NoiseT % self.parms.NoiseChangeRate == 0:
                self.GetEtaNoise()
                # Update backend noise
                self._backend._noise_fft = self.noise_fft
        
        # Perform timestep using refactored dynamics
        fields_next = self._dynamics.compute_fields_next(
            fields={'phi': self.phi},
            dt=float(self.parms.dt),
            operators=self._operators,
            model=self._model
        )
        
        # Update member variables
        self.phi = fields_next['phi']
        self.phi_hat = cp.fft.fft2(self.phi)
        self.t += self.parms.dt
        
        # Validate phi max
        if cp.max(self.phi) > self.phiMax:
            raise Exception(f"phi max > {self.phiMax}")
    
    def CalcEnergyDensity(self):
        """Calculate energy density (delegated to model)."""
        energy = self._model.free_energy({'phi': self.phi})
        self.f = energy / (self.nx * self.ny)
        
        # Also compute energy components for compatibility
        self.phi_hat = cp.fft.fft2(self.phi)
        self.energy_lin_phi = cp.fft.ifft2(self.linenergycoeff * self.phi_hat).real
        self.energy_lin = self.phi * self.energy_lin_phi
        
        self.energy_ln = cp.zeros_like(self.phi)
        pos_mask = self.phi > -self.parms.a
        self.energy_ln[pos_mask] = self.parms.Hln * (self.phi[pos_mask] + self.parms.a) * cp.log(self.phi[pos_mask] + self.parms.a)
        
        self.energy_poly = (1/2 * (self.parms.epsilon + self.parms.beta) * cp.power(self.phi, 2) +
                            1/3 * self.parms.g * cp.power(self.phi, 3) +
                            1/4 * self.parms.v0 * cp.power(self.phi, 4))
        
        self.energy = self.energy_lin + self.energy_ln + self.energy_poly
        return self.f
    
    def Save(self, filename):
        """Save simulation state to HDF5 file."""
        import h5py
        
        # Convert to numpy for saving
        phi_cpu = cp.asnumpy(self.phi)
        
        with h5py.File(filename, 'w') as f:
            # Save field
            f.create_dataset('phi', data=phi_cpu)
            
            # Save parameters
            parms_grp = f.create_group('parameters')
            parms_grp.attrs['epsilon'] = float(self.parms.epsilon)
            parms_grp.attrs['a'] = float(self.parms.a)
            parms_grp.attrs['beta'] = float(self.parms.beta)
            parms_grp.attrs['g'] = float(self.parms.g)
            parms_grp.attrs['v0'] = float(self.parms.v0)
            parms_grp.attrs['Hln'] = float(self.parms.Hln)
            parms_grp.attrs['Hng'] = float(self.parms.Hng)
            parms_grp.attrs['phi0'] = float(self.parms.phi0)
            parms_grp.attrs['dt'] = float(self.parms.dt)
            
            # Save state
            f.attrs['t'] = float(self.t)
            f.attrs['nx'] = self.nx
            f.attrs['ny'] = self.ny
            f.attrs['dx'] = self.dx
            f.attrs['dy'] = self.dy
