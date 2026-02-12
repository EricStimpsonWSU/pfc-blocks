#!/usr/bin/env python
"""
Numerical validation: Original PFC2D_Vacancy vs Refactored Implementation

Runs 2 simulation steps with both implementations and compares:
1. Density field (phi) values
2. Energy calculations

Uses parameters from prospectusVacancy-lowTempTimelapse.ipynb
"""

import sys
import numpy as np
import cupy as cp
from pathlib import Path

# Add paths
pfc_models = Path(__file__).parent / "pfc-models"
sys.path.insert(0, str(pfc_models))
sys.path.insert(0, str(pfc_models / "ln-vacancy"))

# Import ORIGINAL implementation (legacy)
import importlib
import PFC2D_Vacancy as PFC2D_Vacancy_Legacy
importlib.reload(PFC2D_Vacancy_Legacy)

print("=" * 80)
print("ORIGINAL IMPLEMENTATION (from ln-vacancy)")
print("=" * 80)

# Test parameters from prospectus notebook (standard variant: Hng=0, Hln=0)
beta_val = 1.0
epsilon_val = 1.7
g_val = 2.418
phi_0_std = -1.14 / 1.05  # triangle lattice
q_eff = 1.0  # Effective wavenumber

# Initialize ORIGINAL simulation
sim_orig = PFC2D_Vacancy_Legacy.PFC2D_Vacancy()
sim_orig.parms.epsilon = epsilon_val
sim_orig.parms.beta = beta_val
sim_orig.parms.g = g_val
sim_orig.parms.v0 = 1.0
sim_orig.parms.phi0 = phi_0_std
sim_orig.parms.Hng = 0.0  # Standard (non-log)
sim_orig.parms.Hln = 0.0  # Standard (non-log)
sim_orig.parms.a = 0.0
sim_orig.parms.N = 64  # Small grid for quick test
sim_orig.parms.PPU = 10
sim_orig.parms.eta = 0.1
sim_orig.parms.dt = 0.01
sim_orig.parms.seed = 42
sim_orig.parms.NoiseDynamicsFlag = False

# Initialize parameters
sim_orig.InitParms()
sim_orig.SetGeometry(64, 64, 10, scalefactor=1.0, forceUnitCellBoundary=False)
sim_orig.SetDT(0.01)

# Flat initialization for reproducibility
sim_orig.InitFieldFlat(noisy=False)

# Store initial state
phi_orig_t0 = sim_orig.phi.get().copy()
print(f"Initial phi shape: {phi_orig_t0.shape}")
print(f"Initial phi range: [{phi_orig_t0.min():.6f}, {phi_orig_t0.max():.6f}]")
print(f"Initial phi mean: {phi_orig_t0.mean():.6f}")

# Compute initial energy
E_orig_t0 = sim_orig.CalcEnergyDensity()
print(f"Initial energy (density): {E_orig_t0:.10e}")

# Step 1
print("\nStep 1...")
sim_orig.TimeStepCross()
time_orig_t1 = sim_orig.t
phi_orig_t1 = sim_orig.phi.get().copy()
E_orig_t1 = sim_orig.CalcEnergyDensity()
print(f"After step 1: t={time_orig_t1:.4f}")
print(f"  phi range: [{phi_orig_t1.min():.6f}, {phi_orig_t1.max():.6f}]")
print(f"  phi mean: {phi_orig_t1.mean():.6f}")
print(f"  energy: {E_orig_t1:.10e}")

# Step 2
print("\nStep 2...")
sim_orig.TimeStepCross()
time_orig_t2 = sim_orig.t
phi_orig_t2 = sim_orig.phi.get().copy()
E_orig_t2 = sim_orig.CalcEnergyDensity()
print(f"After step 2: t={time_orig_t2:.4f}")
print(f"  phi range: [{phi_orig_t2.min():.6f}, {phi_orig_t2.max():.6f}]")
print(f"  phi mean: {phi_orig_t2.mean():.6f}")
print(f"  energy: {E_orig_t2:.10e}")

print("\n" + "=" * 80)
print("REFACTORED IMPLEMENTATION (from pfc_general)")
print("=" * 80)

# Import REFACTORED implementation
from pfc_general.compatibility import PFC2D_Vacancy as PFC2D_Vacancy_Refactored

# Initialize REFACTORED simulation
sim_ref = PFC2D_Vacancy_Refactored()
sim_ref.parms.epsilon = epsilon_val
sim_ref.parms.beta = beta_val
sim_ref.parms.g = g_val
sim_ref.parms.v0 = 1.0
sim_ref.parms.phi0 = phi_0_std
sim_ref.parms.Hng = 0.0  # Standard (non-log)
sim_ref.parms.Hln = 0.0  # Standard (non-log)
sim_ref.parms.a = 0.0
sim_ref.parms.N = 64
sim_ref.parms.PPU = 10
sim_ref.parms.eta = 0.1
sim_ref.parms.dt = 0.01
sim_ref.parms.seed = 42
sim_ref.parms.NoiseDynamicsFlag = False

# Initialize parameters
sim_ref.InitParms()
sim_ref.SetGeometry(64, 64, 10, scalefactor=1.0, forceUnitCellBoundary=False)
sim_ref.SetDT(0.01)

# Flat initialization (identical seed to original)
sim_ref.InitFieldFlat(noisy=False)

# Store initial state
phi_ref_t0 = sim_ref.phi.get().copy()
print(f"Initial phi shape: {phi_ref_t0.shape}")
print(f"Initial phi range: [{phi_ref_t0.min():.6f}, {phi_ref_t0.max():.6f}]")
print(f"Initial phi mean: {phi_ref_t0.mean():.6f}")

# Compute initial energy
E_ref_t0 = sim_ref.CalcEnergyDensity()
print(f"Initial energy (density): {E_ref_t0:.10e}")

# Step 1
print("\nStep 1...")
sim_ref.TimeStepCross()
time_ref_t1 = sim_ref.t
phi_ref_t1 = sim_ref.phi.get().copy()
E_ref_t1 = sim_ref.CalcEnergyDensity()
print(f"After step 1: t={time_ref_t1:.4f}")
print(f"  phi range: [{phi_ref_t1.min():.6f}, {phi_ref_t1.max():.6f}]")
print(f"  phi mean: {phi_ref_t1.mean():.6f}")
print(f"  energy: {E_ref_t1:.10e}")

# Step 2
print("\nStep 2...")
sim_ref.TimeStepCross()
time_ref_t2 = sim_ref.t
phi_ref_t2 = sim_ref.phi.get().copy()
E_ref_t2 = sim_ref.CalcEnergyDensity()
print(f"After step 2: t={time_ref_t2:.4f}")
print(f"  phi range: [{phi_ref_t2.min():.6f}, {phi_ref_t2.max():.6f}]")
print(f"  phi mean: {phi_ref_t2.mean():.6f}")
print(f"  energy: {E_ref_t2:.10e}")

print("\n" + "=" * 80)
print("NUMERICAL COMPARISON")
print("=" * 80)

# Define tolerance levels
TOL_RELATIVE = 1e-6  # Relative tolerance for floating-point differences
TOL_ABSOLUTE = 1e-12  # Absolute tolerance

def compare_fields(name, field_orig, field_ref, tol_rel=TOL_RELATIVE, tol_abs=TOL_ABSOLUTE):
    """Compare two field arrays with tolerance reporting."""
    diff = np.abs(field_orig - field_ref)
    rel_diff = diff / (np.abs(field_orig) + 1e-10)
    
    max_abs_diff = np.max(diff)
    max_rel_diff = np.max(rel_diff)
    rms_diff = np.sqrt(np.mean(diff ** 2))
    rms_rel_diff = np.sqrt(np.mean(rel_diff ** 2))
    
    # Check tolerance
    within_tol = np.all(diff < tol_abs) or np.all(rel_diff < tol_rel)
    status = "✓ PASS" if within_tol else "✗ FAIL"
    
    print(f"\n{name}:")
    print(f"  Status: {status}")
    print(f"  Max absolute difference: {max_abs_diff:.6e}")
    print(f"  Max relative difference: {max_rel_diff:.6e}")
    print(f"  RMS difference: {rms_diff:.6e}")
    print(f"  RMS relative difference: {rms_rel_diff:.6e}")
    print(f"  Tolerance (absolute): {tol_abs:.6e}")
    print(f"  Tolerance (relative): {tol_rel:.6e}")
    
    return within_tol

def compare_scalars(name, val_orig, val_ref, tol_rel=TOL_RELATIVE, tol_abs=TOL_ABSOLUTE):
    """Compare two scalar values with tolerance reporting."""
    diff = np.abs(val_orig - val_ref)
    rel_diff = diff / (np.abs(val_orig) + 1e-10) if val_orig != 0 else diff
    
    # Check tolerance
    within_tol = (diff < tol_abs) or (rel_diff < tol_rel)
    status = "✓ PASS" if within_tol else "✗ FAIL"
    
    print(f"\n{name}:")
    print(f"  Original: {val_orig:.10e}")
    print(f"  Refactored: {val_ref:.10e}")
    print(f"  Status: {status}")
    print(f"  Absolute difference: {diff:.6e}")
    print(f"  Relative difference: {rel_diff:.6e}")
    print(f"  Tolerance (absolute): {tol_abs:.6e}")
    print(f"  Tolerance (relative): {tol_rel:.6e}")
    
    return within_tol

# Initial state comparison
print("\n--- Initial State (t=0) ---")
results = []
results.append(("phi initial", compare_fields("phi at t=0", phi_orig_t0, phi_ref_t0)))
results.append(("E initial", compare_scalars("Energy at t=0", E_orig_t0, E_ref_t0)))

# After step 1
print("\n--- After Step 1 (t=0.01) ---")
results.append(("phi step 1", compare_fields("phi at step 1", phi_orig_t1, phi_ref_t1)))
results.append(("E step 1", compare_scalars("Energy at step 1", E_orig_t1, E_ref_t1)))

# After step 2
print("\n--- After Step 2 (t=0.02) ---")
results.append(("phi step 2", compare_fields("phi at step 2", phi_orig_t2, phi_ref_t2)))
results.append(("E step 2", compare_scalars("Energy at step 2", E_orig_t2, E_ref_t2)))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
all_pass = all(r[1] for r in results)
print(f"\nTotal tests: {len(results)}")
print(f"Passed: {sum(r[1] for r in results)}")
print(f"Failed: {sum(not r[1] for r in results)}")

if all_pass:
    print("\n✅ ALL TESTS PASSED - Refactored implementation is numerically equivalent!")
    exit(0)
else:
    print("\n❌ SOME TESTS FAILED - Check differences above")
    for name, passed in results:
        if not passed:
            print(f"  ✗ {name}")
    exit(1)
