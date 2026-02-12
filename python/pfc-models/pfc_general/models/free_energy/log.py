"""Logarithmic PFC model (single-field, full-field)."""

from typing import Dict, Optional
import numpy as np
import cupy as cp
from ..base_model import PFCModel


class LogPFCModel2D(PFCModel):
    """
    Logarithmic PFC model for 2D single-field systems.
    
    Free energy includes:
    - Linear terms: beta * (-2*k^2 + k^4) / 2
    - Polynomial terms: epsilon*phi^2/2 + g*phi^3/3 + v0*phi^4/4
    - Logarithmic term: Hln * (phi + a) * log(phi + a)
    - Vacancy penalty: -6*Hng*phi^2 (for phi < 0)
    
    This is a full-field, single-field, single-mode model.
    """
    
    def __init__(
        self,
        epsilon: float,
        beta: float,
        g: float,
        v0: float,
        Hln: float,
        Hng: float = 0.0,
        a: float = 0.0,
        phi0: float = 0.0,
        min_log: float = -10.0
    ):
        """
        Args:
            epsilon: Temperature-related parameter
            beta: Elastic constant
            g: Cubic nonlinearity coefficient
            v0: Quartic nonlinearity coefficient
            Hln: Logarithmic term strength
            Hng: Vacancy penalty strength
            a: Shift for logarithmic term
            phi0: Average density
            min_log: Minimum log value to prevent numerical issues
        """
        super().__init__()

        if float(Hln) > 0.0 and float(phi0) <= 0.0:
            raise ValueError("phi0 must be positive when Hln > 0")
        
        # Convert all parameters to GPU arrays
        self.epsilon = cp.float64(epsilon)
        self.beta = cp.float64(beta)
        self.g = cp.float64(g)
        self.v0 = cp.float64(v0)
        self.Hln = cp.float64(Hln)
        self.Hng = cp.float64(Hng)
        self.a = cp.float64(a)
        self.phi0 = cp.float64(phi0)
        self.min_log = cp.float64(min_log)
        self.min_phi = cp.exp(self.min_log) - self.a
    
    @property
    def num_fields(self) -> int:
        """Single field (phi)."""
        return 1
    
    def functional_derivative(
        self, 
        fields: Dict[str, np.ndarray], 
        mode_data: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute nonlinear part of functional derivative.
        
        Returns:
            dict with 'phi' -> nonlinear functional derivative
        """
        phi = fields['phi']
        if isinstance(phi, np.ndarray):
            phi = cp.asarray(phi)
        if isinstance(phi, np.ndarray):
            phi = cp.asarray(phi)
        
        # Quadratic term
        phi2 = self.g * cp.power(phi, 2)
        
        # Cubic term
        phi3 = self.v0 * cp.power(phi, 3)
        
        # Vacancy penalty: -6*Hng*phi^2 for phi < 0
        phivac = -6 * self.Hng * cp.power(phi, 2) * (phi < 0)
        
        # Logarithmic term with safe evaluation
        philn = cp.ones_like(phi) * self.min_log
        pos_mask = phi > self.min_phi
        philn[pos_mask] = self.Hln * cp.log(phi[pos_mask] + self.a)
        
        # Total nonlinear functional derivative
        dF_dphi = phi2 + phi3 + phivac + philn
        
        return {'phi': dF_dphi}
    
    def free_energy(
        self, 
        fields: Dict[str, np.ndarray], 
        mode_data: Optional[Dict] = None
    ) -> float:
        """
        Compute total free energy (requires k-space info from mode_data).
        
        Args:
            fields: dict with 'phi' field
            mode_data: dict with 'phi_hat', 'k2', 'k4' for linear energy
        
        Returns:
            Total free energy per site
        """
        phi = fields['phi']
        
        # Linear energy (requires Fourier space)
        if mode_data is not None and 'phi_hat' in mode_data:
            phi_hat = mode_data['phi_hat']
            k2 = mode_data['k2']
            k4 = mode_data['k4']
            if isinstance(phi_hat, np.ndarray):
                phi_hat = cp.asarray(phi_hat)
            if isinstance(k2, np.ndarray):
                k2 = cp.asarray(k2)
            if isinstance(k4, np.ndarray):
                k4 = cp.asarray(k4)
            linenergycoeff = self.beta * (-2 * k2 + k4) / 2
            energy_lin_phi = cp.fft.ifft2(linenergycoeff * phi_hat).real
            energy_lin = phi * energy_lin_phi
        else:
            energy_lin = 0
        
        # Logarithmic energy
        energy_ln = cp.zeros_like(phi)
        pos_mask = phi > -self.a
        energy_ln[pos_mask] = self.Hln * (phi[pos_mask] + self.a) * cp.log(phi[pos_mask] + self.a)
        
        # Polynomial energy
        energy_poly = (
            0.5 * (self.epsilon + self.beta) * cp.power(phi, 2) +
            (1.0/3.0) * self.g * cp.power(phi, 3) +
            0.25 * self.v0 * cp.power(phi, 4)
        )
        
        # Total energy density
        energy = energy_lin + energy_ln + energy_poly
        
        # Average energy per site
        nx, ny = phi.shape
        f = energy.sum() / (nx * ny)
        
        return float(f)
