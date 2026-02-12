# PFC General Test Suite

This directory contains pytest tests for the PFC General library.

## Test Files

- **test_syntax.py**: Import and instantiation tests (no GPU required)
- **test_core.py**: Full functionality tests (requires CuPy/GPU)
- **test_backends.py**: Backend API and CPU numerical accuracy tests
- **test_gpu_kernel_backend.py**: GPU kernel backend tests (requires CuPy/GPU)
- **conftest.py**: Shared fixtures for all tests

## Running Tests

### All Tests
```bash
pytest -v
```

### Syntax Tests Only (no GPU needed)
```bash
pytest test_syntax.py -v
```

### Core Tests Only (requires GPU)
```bash
pytest test_core.py -v
```

### GPU Kernel Tests (requires GPU)
```bash
pytest test_gpu_kernel_backend.py -v
```

### Backend Tests (CPU only)
```bash
pytest test_backends.py -v
```

### With Coverage
```bash
pytest --cov=../ --cov-report=html
```

## Test Coverage

The test suite covers:

1. **Domain**: Grid setup, coordinates, wavenumbers
2. **LogPFCModel2D**: Free energy, functional derivative, parameter validation
3. **SpectralOperators2D**: Laplacian, gradient, bilaplacian
4. **FirstOrderDynamics**: ETD timestep, predictor-corrector
5. **CPUBackend**: CPU field initialization, dtype conversion, accuracy checks
6. **GPUBackend**: Array allocation, CPU↔GPU transfer, noise generation
7. **Initial Conditions**: FlatNoisy, TriangularLattice
8. **Simulation**: Full workflow, checkpoints
9. **Integration**: End-to-end simulation runs

## GPU Availability

GPU test files are skipped automatically when CuPy is unavailable. When CuPy
is installed and compatible with the active NumPy version, all GPU tests run
normally.

## Fixtures

Common fixtures available in conftest.py:

- `small_domain`: 32×32 grid for fast testing
- `medium_domain`: 64×64 grid
- `log_model`: Standard log PFC model
- `spectral_ops`: Configured spectral operators
- `first_order_dynamics`: First-order dynamics
- `gpu_backend`: GPU backend
- `flat_ic`: Flat noisy IC
- `triangular_ic`: Triangular lattice IC

## Adding New Tests

1. Create test file: `test_<component>.py`
2. Use fixtures from conftest.py
3. Follow naming convention: `test_<feature>`
4. Add docstrings explaining what is tested
