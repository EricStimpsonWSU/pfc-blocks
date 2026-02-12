"""First-order dynamics (d/dt)."""

from typing import Dict, Optional, Callable
import cupy as cp
from .base_dynamics import Dynamics


class FirstOrderDynamics(Dynamics):
    """
    First-order dynamics: d(phi)/dt = -L(phi) + N(phi) + noise
    
    where L is linear operator, N is nonlinear functional derivative.
    Uses exponential time differencing for stability.
    """
    
    def __init__(self, noise_amplitude: float = 0.0):
        """
        Args:
            noise_amplitude: Strength of stochastic noise
        """
        super().__init__()
        self.noise_amplitude = noise_amplitude
    
    @property
    def order(self) -> int:
        """First-order dynamics."""
        return 1
    
    def compute_fields_next(
        self,
        fields: Dict[str, cp.ndarray],
        dt: float,
        operators,
        model,
        noise_fn: Optional[Callable] = None,
        backend = None
    ) -> Dict[str, cp.ndarray]:
        """
        Compute next timestep using predictor-corrector with ETD.
        
        This implements the exponential time differencing scheme from PFC2D_Vacancy:
        1. Predictor step with current nonlinear term
        2. Corrector step with linearized correction
        
        Args:
            fields: dict with 'phi' field (GPU array)
            dt: timestep
            operators: SpectralOperators2D instance with k2, k4, k6
            model: LogPFCModel2D instance
            noise_fn: optional noise generator
            backend: optional Backend instance; if provided, uses backend.nonlinear_term()
                     instead of model.functional_derivative() for custom kernel support
        
        Returns:
            dict with updated 'phi' field
        """
        phi = fields['phi']
        
        # Get k-space coefficients from operators
        k2 = operators.k2
        k4 = operators.k4
        k6 = operators.k6
        
        # Compute linear coefficient in k-space
        # From PFC2D_Vacancy: lincoeff = -k2*(eps + beta) + 2*beta*k4 - beta*k6
        lincoeff = -k2 * (model.epsilon + model.beta) + 2 * model.beta * k4 - model.beta * k6
        
        # ETD coefficients
        expcoeff = cp.exp(lincoeff * dt)
        
        expcoeff_nonlin = cp.ones_like(phi) * dt
        expcoeff_nonlin[lincoeff != 0] = (
            (expcoeff[lincoeff != 0] - 1) / lincoeff[lincoeff != 0]
        )
        
        expcoeff_nonlin2 = cp.zeros_like(phi)
        expcoeff_nonlin2[lincoeff != 0] = (
            (expcoeff[lincoeff != 0] - (1 + lincoeff[lincoeff != 0] * dt)) /
            cp.power(lincoeff[lincoeff != 0], 2)
        )
        
        # FFT of current field
        phi_hat = cp.fft.fft2(phi)
        
        # Compute nonlinear term N0 using backend hook if available, else model
        if backend is not None and backend.nonlinear_term is not None:
            N0_dict = backend.nonlinear_term(fields, model, operators)
            N0 = N0_dict['phi']
        else:
            func_deriv = model.functional_derivative({'phi': phi})
            N0 = func_deriv['phi']
        
        # Add noise if requested
        if noise_fn is not None:
            noise = noise_fn()
            noise_fft = cp.fft.fft2(noise)
        else:
            noise_fft = 0
        
        N0_hat = cp.fft.fft2(N0) + noise_fft
        
        # Predictor step
        phi_hat0 = expcoeff * phi_hat + (-k2 * expcoeff_nonlin * N0_hat)
        phi0 = cp.fft.ifft2(phi_hat0).real
        
        # Compute N1 (time derivative of nonlinear term)
        if backend is not None and backend.nonlinear_term is not None:
            N1_dict = backend.nonlinear_term({'phi': phi0}, model, operators)
            N1 = N1_dict['phi']
        else:
            func_deriv1 = model.functional_derivative({'phi': phi0})
            N1 = func_deriv1['phi']
        N1_hat = (cp.fft.fft2(N1) + noise_fft - N0_hat) / dt
        
        # Corrector step
        phi_hat1 = expcoeff * phi_hat + (-k2 * (expcoeff_nonlin * N0_hat + expcoeff_nonlin2 * N1_hat))
        phi1 = cp.fft.ifft2(phi_hat1).real
        
        # Optional: check convergence
        delta_phi = phi1 - phi0
        if delta_phi.max() - delta_phi.min() > 0.01:
            # Refine corrector
            phi0 = phi1
            if backend is not None and backend.nonlinear_term is not None:
                N1_dict = backend.nonlinear_term({'phi': phi0}, model, operators)
                N1 = N1_dict['phi']
            else:
                func_deriv1 = model.functional_derivative({'phi': phi0})
                N1 = func_deriv1['phi']
            N1_hat = (cp.fft.fft2(N1) + noise_fft - N0_hat) / dt
            
            phi_hat1 = expcoeff * phi_hat + (-k2 * (expcoeff_nonlin * N0_hat + expcoeff_nonlin2 * N1_hat))
            phi1 = cp.fft.ifft2(phi_hat1).real
            
            delta_phi = phi1 - phi0
            if delta_phi.max() - delta_phi.min() > 0.01:
                raise RuntimeError("Predictor-corrector failed to converge")
        
        return {'phi': phi1}
    
    def compute_fields_velocity(
        self,
        fields: Dict[str, cp.ndarray],
        operators,
        model
    ) -> Dict[str, cp.ndarray]:
        """
        Compute velocity (d phi / dt) at current state.
        
        Args:
            fields: dict with 'phi'
            operators: SpectralOperators2D
            model: LogPFCModel2D
        
        Returns:
            dict with 'phi' -> velocity
        """
        phi = fields['phi']
        phi_hat = cp.fft.fft2(phi)
        
        k2 = operators.k2
        k4 = operators.k4
        k6 = operators.k6
        
        # Linear coefficient
        lincoeff = -k2 * (model.epsilon + model.beta) + 2 * model.beta * k4 - model.beta * k6
        
        # Linear part
        linear_hat = lincoeff * phi_hat
        linear_real = cp.fft.ifft2(linear_hat).real
        
        # Nonlinear part
        func_deriv = model.functional_derivative({'phi': phi})
        nonlinear = -k2 * func_deriv['phi']  # Apply -k2 in real space or k-space
        
        velocity = linear_real + cp.fft.ifft2(cp.fft.fft2(nonlinear)).real
        
        return {'phi': velocity}
