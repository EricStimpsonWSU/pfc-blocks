#!/usr/bin/env python
"""
Numerical validation: Legacy PFC2D_Vacancy vs Refactored compatibility layer.

Runs 2 simulation steps and compares:
- Density field (phi)
- Energy calculations

Baseline: pfc-models/ln-vacancy/PFC2D_Vacancy.py
Refactor: pfc_general.compatibility.PFC2D_Vacancy
"""

from __future__ import annotations

import sys
import numpy as np
import cupy as cp
from pathlib import Path
import importlib.util

ROOT = Path(__file__).parent
LEGACY_PATH = ROOT / "ln-vacancy" / "PFC2D_Vacancy.py"

# Ensure legacy module dependencies resolve
sys.path.insert(0, str(ROOT / "ln-vacancy"))
sys.path.insert(0, str(ROOT))

# Load legacy module directly from file to avoid shim
spec = importlib.util.spec_from_file_location("legacy_pfc2d_vacancy", LEGACY_PATH)
legacy_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(legacy_mod)

from pfc_general.compatibility import PFC2D_Vacancy as RefactoredPFC2D

print("=" * 80)
print("LEGACY VS REFACTORED: 2-STEP NUMERICAL VALIDATION")
print("=" * 80)

# Shared parameters (stable, noisy dynamics)
beta_val = 1.0
epsilon_val = -0.25
g_val = 2.418
phi_0 = -0.1

N = 64
PPU = 10
DT = 0.001
ETA = 0.0
SEED = 42

# Tolerances
TOL_ABS = 1e-10
TOL_REL = 1e-6


def init_sim(sim):
    sim.parms.epsilon = epsilon_val
    sim.parms.beta = beta_val
    sim.parms.g = g_val
    sim.parms.v0 = 1.0
    sim.parms.phi0 = phi_0
    sim.parms.Hng = 0.05
    sim.parms.Hln = 0.1
    sim.parms.a = 0.0
    sim.parms.N = N
    sim.parms.PPU = PPU
    sim.parms.eta = ETA
    sim.parms.dt = DT
    sim.parms.seed = SEED
    sim.parms.NoiseDynamicsFlag = False

    sim.InitParms()
    sim.SetGeometry(N, N, PPU, scalefactor=1.0, forceUnitCellBoundary=False)
    sim.SetDT(DT)

    sim.InitFieldFlat(noisy=False)


def set_shared_field(legacy_sim, ref_sim, amplitude: float = 0.01):
    """Set a deterministic sinusoidal field for both simulations."""
    # Use legacy coordinate grid for deterministic field construction
    Lx = legacy_sim.nx * legacy_sim.dx
    Ly = legacy_sim.ny * legacy_sim.dy
    phi_shared = (legacy_sim.parms.phi0 +
                  amplitude * (cp.sin(2 * cp.pi * legacy_sim.x / Lx) +
                               cp.cos(2 * cp.pi * legacy_sim.y / Ly)))

    # Apply to legacy
    legacy_sim.phi = phi_shared.copy()
    legacy_sim.phi_hat = cp.fft.fft2(legacy_sim.phi)
    legacy_sim.phi0 = cp.fft.ifft2(legacy_sim.phi_hat).real
    legacy_sim.t = 0

    # Apply to refactored
    ref_sim.phi = phi_shared.copy()
    ref_sim.phi_hat = cp.fft.fft2(ref_sim.phi)
    ref_sim.phi0 = cp.fft.ifft2(ref_sim.phi_hat).real
    ref_sim.t = 0


def extract_state(sim):
    phi = sim.phi.get().copy()
    E = sim.CalcEnergyDensity()
    # Legacy returns None and stores energy in sim.f
    if E is None:
        E = sim.f
    return phi, float(E)


def legacy_energy(sim) -> float:
    """Compute energy density using legacy formula for any sim with compatible fields."""
    phi = sim.phi
    phi_hat = cp.fft.fft2(phi)

    # Legacy uses linenergycoeff = beta * (-2*k2 + k4) / 2
    linenergycoeff = sim.parms.beta * (-2 * sim.k2 + sim.k4) / 2
    energy_lin_phi = cp.fft.ifft2(linenergycoeff * phi_hat).real
    energy_lin = phi * energy_lin_phi

    energy_ln = cp.zeros_like(phi)
    pos_mask = phi > -sim.parms.a
    energy_ln[pos_mask] = sim.parms.Hln * (phi[pos_mask] + sim.parms.a) * cp.log(phi[pos_mask] + sim.parms.a)

    energy_poly = (0.5 * (sim.parms.epsilon + sim.parms.beta) * cp.power(phi, 2) +
                   (1.0 / 3.0) * sim.parms.g * cp.power(phi, 3) +
                   0.25 * sim.parms.v0 * cp.power(phi, 4))

    energy = energy_lin + energy_ln + energy_poly
    f = energy.sum() / (sim.nx * sim.ny)
    return float(f)


# Initialize legacy
print("\n[1/2] Initializing legacy model...")
legacy_sim = legacy_mod.PFC2D_Vacancy()
init_sim(legacy_sim)
ref_sim = RefactoredPFC2D()
init_sim(ref_sim)

set_shared_field(legacy_sim, ref_sim)

phi_legacy_t0, _ = extract_state(legacy_sim)
print("[2/2] Initializing refactored model...")
phi_ref_t0, _ = extract_state(ref_sim)

E_legacy_t0 = legacy_energy(legacy_sim)
E_ref_t0 = legacy_energy(ref_sim)


def compare_fields(label: str, a: np.ndarray, b: np.ndarray) -> bool:
    diff = np.abs(a - b)
    max_abs = float(np.max(diff))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    rel = diff / (np.abs(a) + 1e-12)
    max_rel = float(np.max(rel))

    ok = np.allclose(a, b, rtol=TOL_REL, atol=TOL_ABS)
    status = "PASS" if ok else "FAIL"
    print(f"\n{label} - {status}")
    print(f"  max_abs: {max_abs:.6e}")
    print(f"  max_rel: {max_rel:.6e}")
    print(f"  rms:     {rms:.6e}")
    return ok


def compare_scalar(label: str, a: float, b: float) -> bool:
    diff = abs(a - b)
    rel = diff / (abs(a) + 1e-12)
    ok = (diff < TOL_ABS) or (rel < TOL_REL)
    status = "PASS" if ok else "FAIL"
    print(f"\n{label} - {status}")
    print(f"  legacy: {a:.10e}")
    print(f"  ref:    {b:.10e}")
    print(f"  abs:    {diff:.6e}")
    print(f"  rel:    {rel:.6e}")
    return ok


print("\n--- t=0 comparison ---")
res = []
res.append(compare_fields("phi(t=0)", phi_legacy_t0, phi_ref_t0))
res.append(compare_scalar("E(t=0)", E_legacy_t0, E_ref_t0))

# Step 1
legacy_sim.TimeStepCross()
ref_sim.TimeStepCross()
phi_legacy_t1, _ = extract_state(legacy_sim)
phi_ref_t1, _ = extract_state(ref_sim)

E_legacy_t1 = legacy_energy(legacy_sim)
E_ref_t1 = legacy_energy(ref_sim)

print("\n--- t=DT comparison ---")
res.append(compare_fields("phi(t=DT)", phi_legacy_t1, phi_ref_t1))
res.append(compare_scalar("E(t=DT)", E_legacy_t1, E_ref_t1))

# Step 2
legacy_sim.TimeStepCross()
ref_sim.TimeStepCross()
phi_legacy_t2, _ = extract_state(legacy_sim)
phi_ref_t2, _ = extract_state(ref_sim)

E_legacy_t2 = legacy_energy(legacy_sim)
E_ref_t2 = legacy_energy(ref_sim)

print("\n--- t=2*DT comparison ---")
res.append(compare_fields("phi(t=2*DT)", phi_legacy_t2, phi_ref_t2))
res.append(compare_scalar("E(t=2*DT)", E_legacy_t2, E_ref_t2))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

passed = sum(1 for r in res if r)
failed = len(res) - passed
print(f"\nTotal checks: {len(res)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed == 0:
    print("\n✅ Legacy and refactored implementations match within tolerance.")
    sys.exit(0)

print("\n❌ Differences exceed tolerance; inspect results above.")
sys.exit(1)
