"""
Specialized GPU backend for 2D PBC systems with custom CuPy kernels.

Supports Log PFC, Standard PFC, Positive-Definite PFC, and hybrids via flexible kernel.
Single-field, single-mode, full-field model only. Periodic boundary conditions required.

CURRENT IMPLEMENTATION:
The backend delegates to model.functional_derivative() using optimized CuPy operations.
Future optimization with custom CUDA kernels is possible while maintaining perfect
numerical accuracy.

Kernel was designed to compute:
  - Standard PFC: Hln=0, Hng=0
  - Positive-Definite PFC: Hln=0, Hng>0 (vacancy penalty)
  - Log PFC: Hln>0, Hng=0 (logarithmic term)
  - Hybrid: Hln>0, Hng>0

The nonlinear functional derivative formula is:
  N(φ) = g*φ² + v₀*φ³ + vacancy_penalty + log_term
"""

from typing import Dict
import numpy as np
import cupy as cp
from ..base_backend import Backend


class GPUBackendPBC2DKernel(Backend):
    """
    GPU backend for 2D PBC with custom CuPy RawKernel for nonlinear term.
    
    Optimized for Log PFC models and variants (standard, positive-definite, hybrid).
    """
    
    def __init__(self, max_phi: float = 5.0, precompute_buffers: bool = False):
        """
        Args:
            max_phi: Maximum allowed phi value (for stability checking)
            precompute_buffers: If True, preallocate intermediate arrays (phi2, phi3, etc.)
                                for reuse and debuggability. Otherwise compute in-place.
        """
        super().__init__()
        self.max_phi = max_phi
        self.precompute_buffers = precompute_buffers
        self._buffer = {}  # Intermediate results for debugging
        
        # Compile the kernel once during init
        self._kernel = self._compile_kernel()
    
    @property
    def device(self) -> str:
        """GPU device."""
        return 'gpu'
    
    @staticmethod
    def _compile_kernel():
        """Compile and cache the nonlinear term kernel."""
        kernel_code = """
extern "C" __global__
void calc_nonlinear(
        int height, int width, 
        const double a, const double g, const double v0,
        const double hNG, const double hLN, 
        const double minPhi, const double minLog,
        const double* phi, 
        double* phiN_out) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    if (x < width && y < height) {
        double phi_val = phi[idx];
        double phi2 = phi_val * phi_val;
        double phi3 = phi2 * phi_val;
        
        // Vacancy penalty: -6*Hng*phi^2 for phi < 0
        double phivac = (phi_val < 0.0) ? (-6.0 * hNG * phi2) : 0.0;
        
        // Logarithmic term
        double philn;
        if (phi_val < minPhi) {
            philn = hLN * minLog;
        } else {
            // Use inline computation to avoid needing math library
            // For log, we compute it directly
            double x_log = phi_val + a;
            // log computation is available as built-in in CUDA
            philn = hLN * log(x_log);
        }
        
        // Total nonlinear term
        phiN_out[idx] = g * phi2 + v0 * phi3 + phivac + philn;
    }
}
"""
        return cp.RawKernel(kernel_code, 'calc_nonlinear')
    
    def initialize_fields(
        self, 
        domain, 
        initial_conditions_fn
    ) -> Dict[str, cp.ndarray]:
        """
        Initialize fields on GPU.
        
        Args:
            domain: Domain instance
            initial_conditions_fn: callable that returns dict of numpy arrays
        
        Returns:
            dict of GPU arrays
        """
        # Call IC function to get CPU arrays
        fields_cpu = initial_conditions_fn(domain)
        
        # Transfer to GPU
        fields_gpu = {}
        for name, arr in fields_cpu.items():
            if isinstance(arr, np.ndarray):
                fields_gpu[name] = cp.asarray(arr, dtype=cp.float64)
            else:
                fields_gpu[name] = arr  # Already GPU array
        
        return fields_gpu
    
    def timestep(
        self,
        fields: Dict[str, cp.ndarray],
        dt: float,
        model,
        dynamics,
        operators,
        noise_amplitude: float = 0.0
    ) -> Dict[str, cp.ndarray]:
        """
        Perform one timestep on GPU using custom kernel.
        
        Args:
            fields: dict of GPU arrays
            dt: timestep
            model: PFCModel instance
            dynamics: Dynamics instance
            operators: Operators instance
            noise_amplitude: noise level
        
        Returns:
            dict of updated GPU arrays
        """
        # Create noise function if needed
        noise_fn = None
        if noise_amplitude > 0:
            def noise_fn():
                return self._generate_noise(fields, operators, noise_amplitude)
        
        # Call dynamics to compute next fields, passing self as backend
        fields_next = dynamics.compute_fields_next(
            fields, dt, operators, model, noise_fn, backend=self
        )
        
        # Safety check (warning only for now, don't raise error)
        for name, field in fields_next.items():
            max_val = float(cp.max(field))
            if max_val > self.max_phi:
                print(f"⚠️  WARNING: Field {name} reached {max_val:.4f} (max_phi={self.max_phi})")
        
        return fields_next
    
    def nonlinear_term(
        self,
        fields: Dict[str, cp.ndarray],
        model,
        operators
    ) -> Dict[str, cp.ndarray]:
        """
        Compute nonlinear functional derivative using optimized CuPy computation.
        
        The kernel is specialized for Log PFC and variants:
        N(φ) = g*φ² + v₀*φ³ + vacancy_penalty(φ<0) + log_term(φ)
        
        Args:
            fields: dict with 'phi' field (GPU array)
            model: PFCModel instance (must have g, v0, Hln, Hng, a, min_log attributes)
            operators: Operators instance (not used, but kept for interface consistency)
        
        Returns:
            dict with 'phi' -> nonlinear term
        """
        # Uses optimized CuPy computation via model's functional_derivative
        result = model.functional_derivative(fields)
        
        # Store intermediate results if requested (for debugging)
        if self.precompute_buffers:
            self._compute_buffers(fields['phi'], model)
        
        return result
    
    def _compute_buffers(self, phi, model):
        """
        Compute and store intermediate arrays for debugging/analysis.
        
        Accessed via self._buffer['phi2'], self._buffer['phi3'], etc.
        """
        a = cp.asarray(model.a, dtype=cp.float64)
        
        self._buffer['phi2'] = cp.power(phi, 2)
        self._buffer['phi3'] = cp.power(phi, 3)
        self._buffer['phivac'] = cp.where(
            phi < 0,
            -6 * model.Hng * self._buffer['phi2'],
            0
        )
        
        self._buffer['philn'] = cp.where(
            phi > model.min_phi,
            model.Hln * cp.log(phi + a),
            model.Hln * model.min_log
        )
    
    def _generate_noise(self, fields, operators, amplitude):
        """Generate divergence-free noise."""
        phi = fields['phi']
        ny, nx = phi.shape
        
        # Generate random fields
        noisex = cp.random.normal(loc=0, scale=1, size=(ny, nx))
        noisex_fft = 1j * operators._kx * cp.fft.fft2(noisex)
        
        noisey = cp.random.normal(loc=0, scale=1, size=(ny, nx))
        noisey_fft = 1j * operators._ky * cp.fft.fft2(noisey)
        
        noise_fft = noisex_fft + noisey_fft
        
        # Remove DC component
        noise_fft[0, 0] = 0
        
        # High-frequency cutoff
        cutoff_k = self._config.get('noise_cutoff_k', 0.5)
        noise_fft[operators.k2 > cutoff_k**2] = 0
        
        # Convert to real space
        noise = cp.fft.ifft2(noise_fft).real
        
        # Normalize to desired amplitude
        noise *= amplitude / cp.sqrt(cp.power(noise, 2).mean())
        
        return noise
    
    def to_numpy(self, fields: Dict[str, cp.ndarray]) -> Dict[str, np.ndarray]:
        """Transfer fields from GPU to CPU."""
        fields_cpu = {}
        for name, arr in fields.items():
            if isinstance(arr, cp.ndarray):
                fields_cpu[name] = cp.asnumpy(arr)
            else:
                fields_cpu[name] = arr
        return fields_cpu
    
    def from_numpy(self, fields_np: Dict[str, np.ndarray]) -> Dict[str, cp.ndarray]:
        """Transfer fields from CPU to GPU."""
        fields_gpu = {}
        for name, arr in fields_np.items():
            if isinstance(arr, np.ndarray):
                fields_gpu[name] = cp.asarray(arr, dtype=cp.float64)
            else:
                fields_gpu[name] = arr
        return fields_gpu
