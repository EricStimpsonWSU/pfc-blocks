# PFC General Library

A refactored phase field crystal (PFC) simulation library with clean separation of model definition, dynamics, and simulation concerns.

## Structure

- **models/** - PFC model definitions (free energy and functional derivatives)
  - `base_model.py` - Abstract PFCModel interface
  - `free_energy/` - Standard, log, and other energy formulations
  - `variants/` - Full-field, amplitude, single/multi-field, single/multi-mode

- **dynamics/** - Time evolution equations
  - `base_dynamics.py` - Abstract Dynamics interface
  - `first_order.py` - First-order dynamics (d/dt)
  - `second_order.py` - Second-order dynamics (d²/dt² + d/dt) [future]

- **operators/** - Spatial differential operators
  - `base_ops.py` - Abstract Operators interface
  - `spectral_ops.py` - FFT-based (periodic BC)
  - `fd_ops.py` - Finite difference (configurable BC)

- **backends/** - Computational backends
  - `base_backend.py` - Abstract Backend interface
  - `cpu/` - CPU backend (NumPy)
  - `cpu_numba/` - CPU implementation with Numba JIT (experimental)
  - `gpu_kernels/` - GPU implementation with custom kernels (CuPy)

- **simulation/** - Simulation orchestration
  - `domain.py` - Spatial grid and boundary conditions
  - `initial_conditions.py` - Field initialization
  - `runner.py` - Main simulation loop

- **io/** - Configuration and results I/O
- **tests/** - Unit tests

## Documentation

- [PFC2D_General_plan.txt](PFC2D_General_plan.txt) - Overall refactor plan and milestones
- [BASE_INTERFACES.txt](BASE_INTERFACES.txt) - Detailed interface specifications

## Development Status

**Milestone 1** (in progress):
- [x] Define interfaces and folder structure
- [ ] Refactor PFC2D_Vacancy into new structure
- [ ] Build unit tests for core components
- [ ] Create beta testing notebook

**Milestone 2** (planned):
- Advective dynamics
- Finite boundaries (non-periodic)
- FD operators with configurable BC

**Milestone 3+** (TBD)

## Installation

```bash
# From pfc-blocks root
cd python/pfc-models/pfc-general
pip install -e .
```

## Quick Start

```python
from pfc_general import Domain, Simulation
from pfc_general.backends import CPUBackend

# ... (example to be added after Milestone 1)
```

## Backend Selection

CPU-only usage does not require CuPy. GPU backends are imported lazily and only
load CuPy when explicitly referenced.

```python
from pfc_general.backends import CPUBackend, GPUBackendPBC2DKernel

cpu_backend = CPUBackend()
# gpu_backend = GPUBackendPBC2DKernel()  # Requires CuPy/CUDA
```
