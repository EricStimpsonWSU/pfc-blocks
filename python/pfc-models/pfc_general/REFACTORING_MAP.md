# Refactoring Map: PFC2D_Vacancy → PFC General

This document maps components from the original `PFC2D_Vacancy` to the new modular structure.

## Original Structure (PFC2D_Vacancy.py)

```
PFC2D_Vacancy class (445 lines):
  - Parameters (PFC2D_Vacancy_Parms)
  - Geometry setup (SetGeometry)
  - Linear/nonlinear calculations (Get_N_hat, _getN)
  - Time stepping (TimeStepCross)
  - Energy calculation (CalcEnergyDensity)
  - Initial conditions (InitFieldFlat, InitFieldCrystal)
  - Noise generation (GetEtaNoise)
  - I/O (Save)
```

## New Modular Structure

### 1. Model Layer → `models/free_energy/log.py`
**Class: `LogPFCModel2D`**

Maps from PFC2D_Vacancy:
- Parameters: `epsilon, beta, g, v0, Hln, Hng, a, phi0`
- `Get_N_hat()` → `functional_derivative()`
- `_getN()` (kernel) → integrated into `functional_derivative()`
- `CalcEnergyDensity()` → `free_energy()`

Key changes:
- Pure model: no FFT, no time, no domain
- Returns dict of functional derivatives
- GPU-agnostic interface (implementation uses CuPy internally)

### 2. Operators Layer → `operators/spectral_ops.py`
**Class: `SpectralOperators2D`**

Maps from PFC2D_Vacancy:
- `SetGeometry()` k-space setup → `_setup_kernels()`
- `kx, ky, k2, k4, k6` → properties accessible via operators
- Laplacian/gradient operations (implicit in FFT)

Key changes:
- Separated from model and domain
- Provides clean interface for derivatives
- Can be swapped for FD operators later

### 3. Dynamics Layer → `dynamics/first_order.py`
**Class: `FirstOrderDynamics`**

Maps from PFC2D_Vacancy:
- `TimeStepCross()` → `compute_fields_next()`
- ETD scheme (expcoeff, expcoeff_nonlin, expcoeff_nonlin2)
- Predictor-corrector logic

Key changes:
- Pure dynamics: assembles L + N, no FFT details
- Receives operators and model as dependencies
- Separated noise handling

### 4. Backend Layer → `backends/gpu_kernels/backend.py`
**Class: `GPUBackend`**

Maps from PFC2D_Vacancy:
- GPU array management (`cp.zeros`, etc.)
- `GetEtaNoise()` → `_generate_noise()`
- Data transfer (implicit in original)

Key changes:
- Owns GPU memory and data layout
- Orchestrates model/dynamics/operators on device
- Clean CPU ↔ GPU interface

### 5. Domain Layer → `simulation/domain.py`
**Class: `Domain`**

Maps from PFC2D_Vacancy:
- `SetGeometry()` spatial grid → `__init__(shape, box_size)`
- `nx, ny, mx, my, dx, dy` → properties
- `x, y` coordinates → `get_coordinates()`
- `kx, ky` wavenumbers → `get_wavenumbers()`

Key changes:
- Pure metadata: no field storage
- Separated from operators
- Dimension-agnostic (1D/2D/3D ready)

### 6. Initial Conditions → `simulation/ics.py`
**Classes: `FlatNoisy`, `TriangularLattice`**

Maps from PFC2D_Vacancy:
- `InitFieldFlat()` → `FlatNoisy()`
- `InitFieldCrystal()` → `TriangularLattice()`

Key changes:
- Callable objects, not methods
- Domain-aware, but no time/state
- Reusable across simulations

### 7. Simulation Runner → `simulation/runner.py`
**Class: `Simulation`**

Maps from PFC2D_Vacancy:
- Time loop orchestration
- Checkpoint save/load
- Progress reporting

Key changes:
- No numerical computation
- Wires all components together
- Consistent interface for all PFC models

## Parameter Mapping

| PFC2D_Vacancy | PFC General | Location |
|---------------|-------------|----------|
| `parms.epsilon` | `LogPFCModel2D.epsilon` | Model |
| `parms.beta` | `LogPFCModel2D.beta` | Model |
| `parms.g` | `LogPFCModel2D.g` | Model |
| `parms.v0` | `LogPFCModel2D.v0` | Model |
| `parms.Hln` | `LogPFCModel2D.Hln` | Model |
| `parms.Hng` | `LogPFCModel2D.Hng` | Model |
| `parms.a` | `LogPFCModel2D.a` | Model |
| `parms.phi0` | `LogPFCModel2D.phi0` | Model |
| `parms.eta` | `dynamics.noise_amplitude` | Dynamics |
| `parms.dt` | `sim.run(dt=...)` | Simulation |
| `parms.N` | `domain.shape[0]` | Domain |
| `parms.PPU` | (computed) | Domain |
| `parms.seed` | `ic.seed` | InitialConditions |
| `parms.NoiseDynamicsFlag` | `noise_amplitude > 0` | Backend |

## Code Comparison

### Original (PFC2D_Vacancy)
```python
pfc = PFC2D_Vacancy()
pfc.parms.epsilon = -0.25
pfc.parms.beta = 1.0
# ... set all parameters
pfc.InitParms()
pfc.InitFieldCrystal()
for i in range(1000):
    pfc.TimeStepCross()
```

### Refactored (PFC General)
```python
domain = Domain(shape=(256, 256), box_size=(Lx, Ly))
model = LogPFCModel2D(epsilon=-0.25, beta=1.0, ...)
operators = SpectralOperators2D()
dynamics = FirstOrderDynamics()
backend = GPUBackend()
ic = TriangularLattice(phi0=-0.35)

operators.configure({'domain': domain})
sim = Simulation(domain, model, dynamics, backend, operators, ic)
sim.run(num_steps=1000, dt=0.1)
```

## Benefits of Refactoring

1. **Separation of Concerns**: Model, dynamics, operators, simulation are independent
2. **Testability**: Each component can be unit tested in isolation
3. **Extensibility**: 
   - Swap spectral → FD operators without changing model
   - Add new models without changing dynamics
   - Switch CPU ↔ GPU backends transparently
4. **Reusability**: Same operators/dynamics work for different models
5. **Clarity**: Each class has single responsibility
6. **Dimension Flexibility**: 1D/2D/3D supported via domain

## What's Not Yet Implemented

From PFC2D_Vacancy features:
- [ ] Time-smooth noise (`NoiseTimeSmoothFlag`)
- [ ] Boundary potential (`BoundaryPotential, BoundaryDensity`)
- [ ] Custom CuPy kernel (`calc_phiN` RawKernel) - using fused operations instead
- [ ] SPT peak tracking methods (`SPT_LocateCM`, `SPT_AppendData`)
- [ ] Detailed energy breakdown (`CalcEnergyDensityDetails`)

These can be added as:
- Advanced noise → subclass of `FirstOrderDynamics`
- Boundary potential → extension of `LogPFCModel2D`
- Custom kernels → custom `Backend.timestep()` implementation
- SPT tracking → analysis utilities in `io/` or separate module

## Next Steps (Milestone 1)

1. ✅ Refactor PFC2D_Vacancy into new structure
2. [ ] Build unit tests for core components
3. [ ] Create beta testing notebook
4. [ ] Verify numerical equivalence with original code
