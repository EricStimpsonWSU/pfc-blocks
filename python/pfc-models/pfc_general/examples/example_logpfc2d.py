"""
Example: Run refactored PFC2D (Log model) simulation.

This demonstrates the new modular structure with separated concerns:
- Model: LogPFCModel2D
- Dynamics: FirstOrderDynamics
- Operators: SpectralOperators2D
- Backend: GPUBackend
- Simulation: orchestrates everything
"""

import sys
sys.path.insert(0, 'd:/GitHubWSU/pfc-blocks/python/pfc-models/pfc-general')

import numpy as np
import cupy as cp
from simulation import Domain, Simulation, TriangularLattice
from models.free_energy import LogPFCModel2D
from dynamics import FirstOrderDynamics
from operators import SpectralOperators2D
from backends.gpu_kernels import GPUBackend


def run_pfc2d_vacancy_equivalent():
    """
    Run a simulation equivalent to PFC2D_Vacancy with the new structure.
    """
    
    # Parameters (matching PFC2D_Vacancy defaults)
    epsilon = -0.25
    beta = 1.0
    g = 0.0
    v0 = 1.0
    Hln = 1.0
    Hng = 0.5
    a = 0.0
    phi0 = -0.35
    
    # Domain setup (256x256 grid)
    N = 256
    PPU = 32  # pixels per unit cell
    mx = N / PPU
    my = 2 * N / (np.sqrt(3) * PPU)
    mx = int(np.round(mx))
    my = int(np.round(my / 2) * 2)
    
    Lx = 4 * np.pi * mx / np.sqrt(3)
    Ly = 2 * np.pi * my
    
    domain = Domain(
        shape=(N, N),
        box_size=(Lx, Ly),
        dtype=np.float64,
        bc='periodic'
    )
    
    # Create model
    model = LogPFCModel2D(
        epsilon=epsilon,
        beta=beta,
        g=g,
        v0=v0,
        Hln=Hln,
        Hng=Hng,
        a=a,
        phi0=phi0
    )
    
    # Create operators
    operators = SpectralOperators2D()
    operators.configure({'domain': domain})
    
    # Create dynamics
    dynamics = FirstOrderDynamics(noise_amplitude=0.0)
    
    # Create backend
    backend = GPUBackend(max_phi=5.0)
    
    # Create initial conditions (triangular lattice)
    ic = TriangularLattice(
        phi0=phi0,
        amplitude=None,  # Will use phi0/3
        noise_amplitude=0.01,
        seed=42
    )
    
    # Create simulation
    sim = Simulation(
        domain=domain,
        model=model,
        dynamics=dynamics,
        backend=backend,
        operators=operators,
        initial_conditions=ic
    )
    
    # Progress callback
    def progress(step, time, energy):
        if step % 100 == 0:
            print(f"Step {step:5d}, Time {time:8.3f}, Energy {energy:12.6e}")
    
    # Run simulation
    print("Starting simulation...")
    dt = 0.1
    num_steps = 1000
    
    sim.run(
        num_steps=num_steps,
        dt=dt,
        checkpoint_interval=500,
        progress_fn=progress
    )
    
    print("\nSimulation complete!")
    print(f"Final time: {sim.current_time:.3f}")
    
    # Get final field (on CPU)
    final_fields = backend.to_numpy(sim.fields)
    phi_final = final_fields['phi']
    
    print(f"Final phi range: [{phi_final.min():.4f}, {phi_final.max():.4f}]")
    print(f"Final phi mean: {phi_final.mean():.4f}")
    
    return sim, phi_final


if __name__ == "__main__":
    sim, phi_final = run_pfc2d_vacancy_equivalent()
