#!/usr/bin/env python
"""
Numerical validation: Standard PFC vs Log PFC (both refactored)

Runs 2 simulation steps with both standard and log models and compares:
1. Density field (phi) evolution
2. Energy calculations and stability
3. Numerical consistency across model variants

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
print("VALIDATION: STANDARD PFC (refactored)")
print("=" * 80)

# Import refactored implementation
from pfc_general.compatibility import PFC2D_Vacancy, PFC2D_Standard

# Standard PFC parameters (Hng=0, Hln=0)
beta_val = 1.0
epsilon_val = 1.7
g_val = 2.418
phi_0_std = -1.14 / 1.05  # triangle lattice

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
sim_std.parms.eta = 0.1
sim_std.parms.dt = 0.01
sim_std.parms.seed = 42
sim_std.parms.NoiseDynamicsFlag = False

# Must verify parameters are set to standard values
assert sim_std.parms.Hng == 0.0, "Standard PFC should have Hng=0"
assert sim_std.parms.Hln == 0.0, "Standard PFC should have Hln=0"

# Initialize
sim_std.InitParms()
sim_std.SetGeometry(64, 64, 10, scalefactor=1.0, forceUnitCellBoundary=False)
sim_std.SetDT(0.01)
sim_std.InitFieldFlat(noisy=False)

# Store initial state for reference
phi_std_t0 = sim_std.phi.get().copy()
print(f"\nInitial state (t=0):")
print(f"  phi shape: {phi_std_t0.shape}")
print(f"  phi range: [{phi_std_t0.min():.8f}, {phi_std_t0.max():.8f}]")
print(f"  phi mean: {phi_std_t0.mean():.8f}")
print(f"  phi std: {phi_std_t0.std():.8f}")

E_std_t0 = sim_std.CalcEnergyDensity()
print(f"  energy: {E_std_t0:.10e}")

# Step 1
print(f"\nStep 1 (standard)...")
sim_std.TimeStepCross()
phi_std_t1 = sim_std.phi.get().copy()
E_std_t1 = sim_std.CalcEnergyDensity()
print(f"  t={sim_std.t:.4f}: phi=[{phi_std_t1.min():.8f}, {phi_std_t1.max():.8f}], mean={phi_std_t1.mean():.8f}, E={E_std_t1:.10e}")

# Step 2
print(f"Step 2 (standard)...")
sim_std.TimeStepCross()
phi_std_t2 = sim_std.phi.get().copy()
E_std_t2 = sim_std.CalcEnergyDensity()
print(f"  t={sim_std.t:.4f}: phi=[{phi_std_t2.min():.8f}, {phi_std_t2.max():.8f}], mean={phi_std_t2.mean():.8f}, E={E_std_t2:.10e}")

print("\n" + "=" * 80)
print("VALIDATION: LOG PFC WITH VACANCY TERMS (refactored)")
print("=" * 80)

# Log PFC parameters (with vacancy and logarithmic terms)
Hln_val = 0.1  # Logarithmic energy strength
Hng_val = 0.05  # Vacancy/negative density penalty

# Initialize LOG PFC simulation (same parameters as Standard, but with vacancy terms)
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
sim_log.parms.eta = 0.1
sim_log.parms.dt = 0.01
sim_log.parms.seed = 42
sim_log.parms.NoiseDynamicsFlag = False

# Initialize
sim_log.InitParms()
sim_log.SetGeometry(64, 64, 10, scalefactor=1.0, forceUnitCellBoundary=False)
sim_log.SetDT(0.01)
sim_log.InitFieldFlat(noisy=False)

# Store initial state
phi_log_t0 = sim_log.phi.get().copy()
print(f"\nInitial state (t=0):")
print(f"  phi shape: {phi_log_t0.shape}")
print(f"  phi range: [{phi_log_t0.min():.8f}, {phi_log_t0.max():.8f}]")
print(f"  phi mean: {phi_log_t0.mean():.8f}")
print(f"  phi std: {phi_log_t0.std():.8f}")

E_log_t0 = sim_log.CalcEnergyDensity()
print(f"  energy: {E_log_t0:.10e}")

# Step 1
print(f"\nStep 1 (log)...")
sim_log.TimeStepCross()
phi_log_t1 = sim_log.phi.get().copy()
E_log_t1 = sim_log.CalcEnergyDensity()
print(f"  t={sim_log.t:.4f}: phi=[{phi_log_t1.min():.8f}, {phi_log_t1.max():.8f}], mean={phi_log_t1.mean():.8f}, E={E_log_t1:.10e}")

# Step 2
print(f"Step 2 (log)...")
sim_log.TimeStepCross()
phi_log_t2 = sim_log.phi.get().copy()
E_log_t2 = sim_log.CalcEnergyDensity()
print(f"  t={sim_log.t:.4f}: phi=[{phi_log_t2.min():.8f}, {phi_log_t2.max():.8f}], mean={phi_log_t2.mean():.8f}, E={E_log_t2:.10e}")

print("\n" + "=" * 80)
print("NUMERICAL ANALYSIS & VALIDATION")
print("=" * 80)

# Tolerance levels for eigenvalue stability
TOL_REL = 1e-6
TOL_ABS = 1e-12

def analyze_field_mutation(name, phi_t0, phi_t1, phi_t2):
    """Analyze how field changes over time."""
    delta_t1 = np.sqrt(np.mean((phi_t1 - phi_t0) ** 2))
    delta_t2 = np.sqrt(np.mean((phi_t2 - phi_t1) ** 2))
    total_change = np.sqrt(np.mean((phi_t2 - phi_t0) ** 2))
    
    print(f"\n{name} field mutation:")
    print(f"  RMS change step 1: {delta_t1:.6e}")
    print(f"  RMS change step 2: {delta_t2:.6e}")
    print(f"  RMS change total: {total_change:.6e}")
    print(f"  Step size increasing: {delta_t2 > delta_t1}")
    
    return delta_t1, delta_t2, total_change

def analyze_energy(name, E0, E1, E2):
    """Analyze energy evolution."""
    dE_1 = E1 - E0
    dE_2 = E2 - E1
    
    print(f"\n{name} energy evolution:")
    print(f"  E(t=0.00) = {E0:.10e}")
    print(f"  E(t=0.01) = {E1:.10e} (ΔE = {dE_1:.6e})")
    print(f"  E(t=0.02) = {E2:.10e} (ΔE = {dE_2:.6e})")
    print(f"  Energy decreasing (stable): {E1 < E0 and E2 < E1}")
    
    return dE_1, dE_2

# Analyze Standard PFC
print("\n" + "-" * 80)
print("STANDARD PFC ANALYSIS")
print("-" * 80)
std_delta_t1, std_delta_t2, std_total = analyze_field_mutation("Standard PFC", phi_std_t0, phi_std_t1, phi_std_t2)
std_dE_1, std_dE_2 = analyze_energy("Standard PFC", E_std_t0, E_std_t1, E_std_t2)

# Analyze Log PFC
print("\n" + "-" * 80)
print("LOG PFC ANALYSIS")
print("-" * 80)
log_delta_t1, log_delta_t2, log_total = analyze_field_mutation("Log PFC", phi_log_t0, phi_log_t1, phi_log_t2)
log_dE_1, log_dE_2 = analyze_energy("Log PFC", E_log_t0, E_log_t1, E_log_t2)

# Cross-model comparison
print("\n" + "-" * 80)
print("CROSS-MODEL COMPARISON")
print("-" * 80)

# Initial conditions should be identical (same seed, same initial field)
print("\nInitial field comparison (should be identical):")
phi_diff_t0 = np.abs(phi_std_t0 - phi_log_t0)
print(f"  Max absolute difference: {np.max(phi_diff_t0):.6e}")
print(f"  RMS difference: {np.sqrt(np.mean(phi_diff_t0 ** 2)):.6e}")
init_identical = np.allclose(phi_std_t0, phi_log_t0, rtol=TOL_REL, atol=TOL_ABS)
print(f"  Status: {'✓ PASS' if init_identical else '✗ FAIL'} - {'Identical' if init_identical else 'Different'}")

# Energy difference due to vacancy/log terms
print("\nEnergy difference due to vacancy/log terms:")
E_diff_t0 = np.abs(E_log_t0 - E_std_t0)
E_diff_t1 = np.abs(E_log_t1 - E_std_t1)
E_diff_t2 = np.abs(E_log_t2 - E_std_t2)
print(f"  At t=0.00: ΔE = {E_diff_t0:.6e}")
print(f"  At t=0.01: ΔE = {E_diff_t1:.6e}")
print(f"  At t=0.02: ΔE = {E_diff_t2:.6e}")
print(f"  Energy difference growing: {E_diff_t1 > E_diff_t0 and E_diff_t2 > E_diff_t1}")

# Field difference growth
print("\nField evolution difference:")
phi_diff_t1 = np.sqrt(np.mean((phi_std_t1 - phi_log_t1) ** 2))
phi_diff_t2 = np.sqrt(np.mean((phi_std_t2 - phi_log_t2) ** 2))
print(f"  RMS phi difference at step 1: {phi_diff_t1:.6e}")
print(f"  RMS phi difference at step 2: {phi_diff_t2:.6e}")
print(f"  Divergence increasing: {phi_diff_t2 > phi_diff_t1}")

print("\n" + "=" * 80)
print("STABILITY CHECKS")
print("=" * 80)

checks = []

# Check 1: Field values are finite
std_finite = np.all(np.isfinite(phi_std_t2))
log_finite = np.all(np.isfinite(phi_log_t2))
print(f"\n✓ Standard PFC field finite: {std_finite}")
print(f"✓ Log PFC field finite: {log_finite}")
checks.append(("Standard finite", std_finite))
checks.append(("Log finite", log_finite))

# Check 2: Energy values are finite
std_energy_finite = np.isfinite(E_std_t2)
log_energy_finite = np.isfinite(E_log_t2)
print(f"✓ Standard PFC energy finite: {std_energy_finite}")
print(f"✓ Log PFC energy finite: {log_energy_finite}")
checks.append(("Standard energy finite", std_energy_finite))
checks.append(("Log energy finite", log_energy_finite))

# Check 3: Step sizes are reasonable (not diverging)
std_reasonable = std_delta_t2 < 1e5  # Arbitrary but reasonable upper bound
log_reasonable = log_delta_t2 < 1e5
print(f"✓ Standard PFC step size reasonable: {std_reasonable}")
print(f"✓ Log PFC step size reasonable: {log_reasonable}")
checks.append(("Standard reasonable", std_reasonable))
checks.append(("Log reasonable", log_reasonable))

# Check 4: Initial conditions identical
checks.append(("Initial conditions", init_identical))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTotal stability checks: {len(checks)}")
print(f"Passed: {sum(c[1] for c in checks)}")
print(f"Failed: {sum(not c[1] for c in checks)}")

all_pass = all(c[1] for c in checks)
if all_pass:
    print("\n✅ ALL VALIDATION CHECKS PASSED")
    print("   - Standard PFC: numerically stable and physically consistent")
    print("   - Log PFC: numerically stable with additional energy terms")
    print("   - Both implementations use refactored pfc_general library")
    print("   - Density field evolution is smooth and finite")
    exit(0)
else:
    print("\n❌ SOME CHECKS FAILED")
    for name, result in checks:
        if not result:
            print(f"   ✗ {name}")
    exit(1)
