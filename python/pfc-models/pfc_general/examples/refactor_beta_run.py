#!/usr/bin/env python
"""Beta script for refactored PFC model (no legacy syntax).

Runs a 2D LogPFCModel2D simulation with noisy crystal initialization and
relaxes toward a ground state using a simple energy-convergence criterion.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure local package resolution without legacy shims
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pfc_general import Domain
from pfc_general.simulation.runner import Simulation
from pfc_general.simulation.ics import TriangularLattice
from pfc_general.models.free_energy.log import LogPFCModel2D
from pfc_general.dynamics.first_order import FirstOrderDynamics
from pfc_general.operators.spectral_ops import SpectralOperators2D
from pfc_general.backends.gpu_kernels.backend import GPUBackend


def build_domain(n: int, ppu: float) -> Domain:
    mx = n / ppu
    my = 2 * n / (np.sqrt(3) * ppu)
    mx = int(np.round(mx))
    my = int(np.round(my / 2) * 2)

    lx = 4 * np.pi * mx / np.sqrt(3)
    ly = 2 * np.pi * my

    return Domain(shape=(n, n), box_size=(lx, ly), dtype=np.float64, bc="periodic")


def main() -> None:
    # Model and run parameters
    n = 180
    dt = 0.01
    ppu = 30

    epsilon = -0.25
    beta = 1.0
    g = 2.418
    v0 = 1.0
    Hln = 0.1
    Hng = 0.00
    a = 0.0
    phi0 = -0.1

    max_steps = 3000
    energy_tol = 1e-8
    patience = 50
    report_every = 50

    # Build components
    domain = build_domain(n, ppu)

    model = LogPFCModel2D(
        epsilon=epsilon,
        beta=beta,
        g=g,
        v0=v0,
        Hln=Hln,
        Hng=Hng,
        a=a,
        phi0=phi0,
        min_log=-12.0,
    )

    operators = SpectralOperators2D()
    operators.configure({"domain": domain})

    dynamics = FirstOrderDynamics(noise_amplitude=0.0)
    backend = GPUBackend(max_phi=5.0)

    ic = TriangularLattice(
        phi0=phi0,
        amplitude=None,
        noise_amplitude=0.02,
        seed=123,
    )

    sim = Simulation(
        domain=domain,
        model=model,
        dynamics=dynamics,
        backend=backend,
        operators=operators,
        initial_conditions=ic,
    )

    # Initial energy
    fields_np = backend.to_numpy(sim.fields)
    energy = model.free_energy(fields_np)
    energy_history = [energy]

    print("Starting beta run...")
    print(f"N={n}, dt={dt}, PPU={ppu}, max_steps={max_steps}")
    print(f"Initial energy: {energy:.10e}")

    stable_count = 0
    for step in range(1, max_steps + 1):
        sim.fields = backend.timestep(sim.fields, dt, model, dynamics, operators)
        sim.current_step += 1
        sim.current_time += dt

        if step % report_every == 0 or step == 1:
            fields_np = backend.to_numpy(sim.fields)
            energy = model.free_energy(fields_np)
            energy_history.append(energy)

            dE = abs(energy_history[-1] - energy_history[-2]) if len(energy_history) > 1 else 0.0
            rel_dE = dE / (abs(energy_history[-2]) + 1e-12) if len(energy_history) > 1 else 0.0

            print(
                f"step={step:5d} t={sim.current_time:8.3f} "
                f"E={energy:.10e} dE={dE:.3e} rel={rel_dE:.3e}"
            )

            if rel_dE < energy_tol:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= patience:
                print("Converged: energy change below tolerance.")
                break

    # Save final field and energy trace
    final_fields = backend.to_numpy(sim.fields)
    phi_final = final_fields["phi"]

    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "refactor_beta_phi_final.npy", phi_final)
    np.save(out_dir / "refactor_beta_energy.npy", np.array(energy_history))

    print("Beta run complete.")
    print(f"Final step: {sim.current_step}")
    print(f"Final time: {sim.current_time:.3f}")
    print(f"Final phi range: [{phi_final.min():.6f}, {phi_final.max():.6f}]")


if __name__ == "__main__":
    main()
