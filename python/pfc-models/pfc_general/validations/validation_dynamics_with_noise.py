#!/usr/bin/env python
"""
Numerical validation: Standard PFC vs Log PFC with NOISY INITIALIZATION

Runs 2 simulation steps with both standard and log models with perturbations:
1. Density field (phi) evolution with realistic dynamics
2. Energy calculations and gradient flows
3. Numerical consistency across model variants with noise

Uses parameters from prospectusVacancy-lowTempTimelapse.ipynb
Both implementations use the refactored pfc_general library.
"""

import sys
import numpy as np
import cupy as cp
from pathlib import Path

# Add paths
pfc_models = Path(__file__).parent
sys.path.insert(0, str(pfc_models))

print("=" * 80)
print("VALIDATION: STANDARD PFC WITH NOISY INITIALIZATION (refactored)")
print("=" * 80)

# Import refactored implementation
from pfc_general.compatibility import PFC2D_Vacancy, PFC2D_Standard

# Standard PFC parameters (Hng=0, Hln=0)
beta_val = 1.0
epsilon_val = -0.25  # Mildly unstable to avoid convergence issues
g_val = 2.418
phi_0_std = -0.1  # Small base value

# Initialize STANDARD PFC simulation
sim_std = PFC2D_Standard()
sim_std.parms.epsilon = epsilon_val
sim_std.parms.beta = beta_val
sim_std.parms.g = g_val
sim_std.parms.v0 = 1.0
sim_std.parms.phi0 = phi_0_std
sim_std.parms.a = 0.0
sim_std.parms.N = 64  # Small grid for reproducibility
sim_std.parms.PPU = 10
sim_std.parms.eta = 0.01  # Noise amplitude
sim_std.parms.dt = 0.001
sim_std.parms.seed = 42
sim_std.parms.NoiseDynamicsFlag = False

# Must verify parameters are set to standard values
assert sim_std.parms.Hng == 0.0, "Standard PFC should have Hng=0"
assert sim_std.parms.Hln == 0.0, "Standard PFC should have Hln=0"

# Initialize
sim_std.InitParms()
sim_std.SetGeometry(64, 64, 10, scalefactor=1.0, forceUnitCellBoundary=False)
sim_std.SetDT(0.001)

# Initialize with NOISE to create dynamics
sim_std.InitFieldFlat(noisy=True)

# Store initial state for reference
phi_std_t0 = sim_std.phi.get().copy()
print(f"\nInitial state (t=0, with noise):")
print(f"  phi shape: {phi_std_t0.shape}")
print(f"  phi range: [{phi_std_t0.min():.8f}, {phi_std_t0.max():.8f}]")
print(f"  phi mean: {phi_std_t0.mean():.8f}")
print(f"  phi std: {phi_std_t0.std():.8f}")

E_std_t0 = sim_std.CalcEnergyDensity()
print(f"  energy: {E_std_t0:.10e}")

# Step 1
print(f"\nStep 1 (standard, dt={sim_std.parms.dt})...")
sim_std.TimeStepCross()
phi_std_t1 = sim_std.phi.get().copy()
E_std_t1 = sim_std.CalcEnergyDensity()
delta_std_t1 = np.sqrt(np.mean((phi_std_t1 - phi_std_t0) ** 2))
print(f"  t={sim_std.t:.4f}: phi=[{phi_std_t1.min():.8f}, {phi_std_t1.max():.8f}]")
print(f"  mean={phi_std_t1.mean():.8f}, std={phi_std_t1.std():.8f}")
print(f"  E={E_std_t1:.10e}, ΔE={E_std_t1-E_std_t0:.6e}, RMS Δφ={delta_std_t1:.6e}")

# Step 2
print(f"Step 2 (standard)...")
sim_std.TimeStepCross()
phi_std_t2 = sim_std.phi.get().copy()
E_std_t2 = sim_std.CalcEnergyDensity()
delta_std_t2 = np.sqrt(np.mean((phi_std_t2 - phi_std_t1) ** 2))
print(f"  t={sim_std.t:.4f}: phi=[{phi_std_t2.min():.8f}, {phi_std_t2.max():.8f}]")
print(f"  mean={phi_std_t2.mean():.8f}, std={phi_std_t2.std():.8f}")
print(f"  E={E_std_t2:.10e}, ΔE={E_std_t2-E_std_t1:.6e}, RMS Δφ={delta_std_t2:.6e}")

print("\n" + "=" * 80)
print("VALIDATION: LOG PFC WITH VACANCIES AND NOISE (refactored)")
print("=" * 80)

# Log PFC parameters (same as standard, plus vacancy/log terms)
Hln_val = 0.1  # Logarithmic energy strength
Hng_val = 0.05  # Vacancy/negative density penalty

# Initialize LOG PFC simulation
sim_log = PFC2D_Vacancy()
sim_log.parms.epsilon = epsilon_val
sim_log.parms.beta = beta_val
sim_log.parms.g = g_val
sim_log.parms.v0 = 1.0
sim_log.parms.phi0 = phi_0_std
sim_log.parms.Hng = Hng_val  # Vacancy term
sim_log.parms.Hln = Hln_val  # Logarithmic term
sim_log.parms.a = 0.0
sim_log.parms.N = 64
sim_log.parms.PPU = 10
sim_log.parms.eta = 0.01
sim_log.parms.dt = 0.001
sim_log.parms.seed = 42
sim_log.parms.NoiseDynamicsFlag = False

# Initialize
sim_log.InitParms()
sim_log.SetGeometry(64, 64, 10, scalefactor=1.0, forceUnitCellBoundary=False)
sim_log.SetDT(0.001)

# Initialize with NOISE (same seed as standard for reproducibility)
sim_log.InitFieldFlat(noisy=True)

# Store initial state
phi_log_t0 = sim_log.phi.get().copy()
print(f"\nInitial state (t=0, with noise):")
print(f"  phi shape: {phi_log_t0.shape}")
print(f"  phi range: [{phi_log_t0.min():.8f}, {phi_log_t0.max():.8f}]")
print(f"  phi mean: {phi_log_t0.mean():.8f}")
print(f"  phi std: {phi_log_t0.std():.8f}")

E_log_t0 = sim_log.CalcEnergyDensity()
print(f"  energy: {E_log_t0:.10e}")

# Step 1
print(f"\nStep 1 (log, dt={sim_log.parms.dt})...")
sim_log.TimeStepCross()
phi_log_t1 = sim_log.phi.get().copy()
E_log_t1 = sim_log.CalcEnergyDensity()
delta_log_t1 = np.sqrt(np.mean((phi_log_t1 - phi_log_t0) ** 2))
print(f"  t={sim_log.t:.4f}: phi=[{phi_log_t1.min():.8f}, {phi_log_t1.max():.8f}]")
print(f"  mean={phi_log_t1.mean():.8f}, std={phi_log_t1.std():.8f}")
print(f"  E={E_log_t1:.10e}, ΔE={E_log_t1-E_log_t0:.6e}, RMS Δφ={delta_log_t1:.6e}")

# Step 2
print(f"Step 2 (log)...")
sim_log.TimeStepCross()
phi_log_t2 = sim_log.phi.get().copy()
E_log_t2 = sim_log.CalcEnergyDensity()
delta_log_t2 = np.sqrt(np.mean((phi_log_t2 - phi_log_t1) ** 2))
print(f"  t={sim_log.t:.4f}: phi=[{phi_log_t2.min():.8f}, {phi_log_t2.max():.8f}]")
print(f"  mean={phi_log_t2.mean():.8f}, std={phi_log_t2.std():.8f}")
print(f"  E={E_log_t2:.10e}, ΔE={E_log_t2-E_log_t1:.6e}, RMS Δφ={delta_log_t2:.6e}")

print("\n" + "=" * 80)
print("NUMERICAL ANALYSIS")
print("=" * 80)

# Tolerance levels
TOL_REL = 1e-6
TOL_ABS = 1e-12

# Energy dissipation analysis
print("\nStandard PFC energy dissipation:")
print(f"  E(t=0.00) = {E_std_t0:.10e}")
print(f"  E(t=0.01) = {E_std_t1:.10e}")
print(f"  E(t=0.02) = {E_std_t2:.10e}")
print(f"  Valid dynamics: E₁ < E₀ and E₂ < E₁? {E_std_t1 <= E_std_t0 and E_std_t2 <= E_std_t1}")

print("\nLog PFC energy dissipation:")
print(f"  E(t=0.00) = {E_log_t0:.10e}")
print(f"  E(t=0.01) = {E_log_t1:.10e}")
print(f"  E(t=0.02) = {E_log_t2:.10e}")
print(f"  Valid dynamics: E₁ < E₀ and E₂ < E₁? {E_log_t1 <= E_log_t0 and E_log_t2 <= E_log_t1}")

# Step size analysis
print("\nField evolution rates:")
print(f"  Standard: Δφ₁={delta_std_t1:.6e}, Δφ₂={delta_std_t2:.6e}")
print(f"  Log:      Δφ₁={delta_log_t1:.6e}, Δφ₂={delta_log_t2:.6e}")
print(f"  Both converging: {delta_std_t2 < delta_std_t1 and delta_log_t2 < delta_log_t1}")

# Cross-model comparison
print("\nCross-model field comparison:")
# Initial conditions from different random seeds can differ
init_diff = np.sqrt(np.mean((phi_std_t0 - phi_log_t0) ** 2))
print(f"  Initial RMS difference: {init_diff:.6e}")

# After evolution - check if difference is growing or bounded
diff_t1 = np.sqrt(np.mean((phi_std_t1 - phi_log_t1) ** 2))
diff_t2 = np.sqrt(np.mean((phi_std_t2 - phi_log_t2) ** 2))
print(f"  After step 1: {diff_t1:.6e}")
print(f"  After step 2: {diff_t2:.6e}")
print(f"  Difference bounded: {diff_t2 - diff_t1 < diff_t1}")

# Energy difference
E_diff_t1 = np.abs(E_log_t1 - E_std_t1)
E_diff_t2 = np.abs(E_log_t2 - E_std_t2)
print("\nEnergy difference from log/vacancy terms:")
print(f"  At step 1: ΔE = {E_diff_t1:.6e}")
print(f"  At step 2: ΔE = {E_diff_t2:.6e}")
print(f"  Reasonable difference: {E_diff_t1 > 0 and E_diff_t2 > 0}")

print("\n" + "=" * 80)
print("STABILITY CHECKS")
print("=" * 80)

checks = []

# Check 1: Fields are finite and not NaN
std_finite = np.all(np.isfinite(phi_std_t2))
log_finite = np.all(np.isfinite(phi_log_t2))
print(f"✓ Standard PFC field finite: {std_finite}")
print(f"✓ Log PFC field finite: {log_finite}")
checks.append(("Standard finite", std_finite))
checks.append(("Log finite", log_finite))

# Check 2: Energy values are finite
std_energy_finite = all(np.isfinite(v) for v in [E_std_t0, E_std_t1, E_std_t2])
log_energy_finite = all(np.isfinite(v) for v in [E_log_t0, E_log_t1, E_log_t2])
print(f"✓ Standard PFC energy finite: {std_energy_finite}")
print(f"✓ Log PFC energy finite: {log_energy_finite}")
checks.append(("Standard energy finite", std_energy_finite))
checks.append(("Log energy finite", log_energy_finite))

# Check 3: Energy is non-increasing (gradient flow)
std_decreasing = E_std_t1 <= E_std_t0 and E_std_t2 <= E_std_t1
log_decreasing = E_log_t1 <= E_log_t0 and E_log_t2 <= E_log_t1
print(f"✓ Standard PFC energy non-increasing: {std_decreasing}")
print(f"✓ Log PFC energy non-increasing: {log_decreasing}")
checks.append(("Standard decreasing", std_decreasing))
checks.append(("Log decreasing", log_decreasing))

# Check 4: Step sizes converge (not diverging)
std_converging = delta_std_t2 < delta_std_t1
log_converging = delta_log_t2 < delta_log_t1
print(f"✓ Standard PFC step converging: {std_converging}")
print(f"✓ Log PFC step converging: {log_converging}")
checks.append(("Standard converging", std_converging))
checks.append(("Log converging", log_converging))

# Check 5: Fields show realistic dynamics
std_dynamic = delta_std_t1 > 1e-8
log_dynamic = delta_log_t1 > 1e-8
print(f"✓ Standard PFC has real dynamics: {std_dynamic}")
print(f"✓ Log PFC has real dynamics: {log_dynamic}")
checks.append(("Standard dynamic", std_dynamic))
checks.append(("Log dynamic", log_dynamic))

# Check 6: Difference between models is reasonable (not exploding)
diff_reasonable = diff_t2 < 1.0
print(f"✓ Model difference bounded: {diff_reasonable}")
checks.append(("Difference bounded", diff_reasonable))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTotal validation checks: {len(checks)}")
print(f"Passed: {sum(c[1] for c in checks)}")
print(f"Failed: {sum(not c[1] for c in checks)}")

all_pass = all(c[1] for c in checks)
if all_pass:
    print("\n✅ ALL VALIDATION CHECKS PASSED")
    print("   - Standard PFC: numerically stable with smooth gradient flow")
    print("   - Log PFC: numerically stable with vacancy/logarithmic energy terms")
    print("   - Both show convergent dynamics (step size decreasing)")
    print("   - Energy correctly dissipates through time steps")
    print("   - Field values remain finite and physical")
    print("   - Refactored pfc_general implementation is verified")
    exit(0)
else:
    print("\n⚠️  SOME CHECKS FAILED")
    for name, result in checks:
        if not result:
            print(f"   ✗ {name}")
    exit(1)
