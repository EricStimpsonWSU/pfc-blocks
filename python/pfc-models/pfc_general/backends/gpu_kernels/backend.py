"""GPU backend using CuPy."""

from typing import Dict, Callable
import numpy as np
import cupy as cp
from ..base_backend import Backend


class GPUBackend(Backend):
    """
    GPU backend using CuPy arrays and kernels.
    
    Manages GPU memory and executes timesteps on GPU.
    """
    
    def __init__(self, max_phi: float = 5.0):
        """
        Args:
            max_phi: Maximum allowed phi value (for stability checking)
        """
        super().__init__()
        self.max_phi = max_phi
    
    @property
    def device(self) -> str:
        """GPU device."""
        return 'gpu'
    
    def initialize_fields(
        self, 
        domain, 
        initial_conditions_fn: Callable
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
        Perform one timestep on GPU.
        
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
        
        # Call dynamics to compute next fields
        fields_next = dynamics.compute_fields_next(
            fields, dt, operators, model, noise_fn
        )
        
        # Safety check
        for name, field in fields_next.items():
            if cp.max(field) > self.max_phi:
                raise RuntimeError(f"Field {name} exceeded max value {self.max_phi}")
        
        return fields_next
    
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
        
        # High-frequency cutoff (k < 0.5 by default)
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
