from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable
import shutil

import numpy as np

try:
  import numba as nb
  HAS_NUMBA = True
except ImportError:
  nb = None
  HAS_NUMBA = False

DEFAULT_DTYPE = np.float32

# Simulation overview (data structures and algorithm)
# - Coordinates: `coords` is an (N, dim) array of sphere centers in a cubic box
#   centered at the origin with side length `Lbox`. Hard-sphere exclusion uses
#   radius `r`, so valid centers satisfy |x| <= Lbox/2 - r in each dimension.
# - Linked-cell grid: space is divided into `n_cells` per dimension, with cell
#   edge length `Lcell = Lbox / n_cells` chosen so each cell is about one
#   diameter (2r). The grid is flattened into a 1D array using row-major
#   `cellStrides` so a cell index maps to `dot(cell_idx, cellStrides)`.
# - Cell storage: `cellGrid` is a (n_cells**dim, max_cell_occupancy) int array
#   storing particle indices in each cell; empty slots are -1. The occupancy
#   cap is a small fixed number (2**dim + 1) for speed and memory locality.
# - Reverse mapping: `cellAssignments[i]` stores the cell index (dim entries)
#   plus the slot index for particle i. This allows O(1) removal when a particle
#   crosses into a new cell.
# - Neighbor offsets: `neighbor_offsets` enumerates the 3**dim local neighbor
#   cells (including the current cell) used for overlap checks.
#
# Algorithm (one Monte Carlo sweep in `TimeStep`):
# 1) Randomize particle order and draw an isotropic unit direction per particle.
# 2) Propose a displacement of length `dl` along that direction.
# 3) Reject if the proposed center exits the hard-wall box.
# 4) Reject if any sphere in neighboring cells is within 2r.
# 5) If accepted, update `coords` and, if needed, move the particle between
#    cells in `cellGrid` while updating `cellAssignments`.
#
# The same logic is implemented in pure NumPy and in a Numba-accelerated kernel
# to improve performance without changing behavior.
#
# ASCII schematic (2D example, cells flattened row-major):
#   cell indices: (ix, iy)
#   flatten: linear = ix * n_cells + iy
#
#   y ^
#     |  (0,2)  (1,2)  (2,2)
#     |  (0,1)  (1,1)  (2,1)
#     |  (0,0)  (1,0)  (2,0)  -> x
#
# Each cell stores a small fixed list of particle indices. When a particle
# moves across a cell boundary, we clear its old slot and insert it into the
# new cell's first empty slot.


def _as_coords_array(coords: Iterable[Iterable[float]], dim: int) -> np.ndarray:
  """Normalize coordinates to a finite (N, dim) float array.

  Raises a ValueError when the input is not a 2D array, is empty, or contains
  non-finite values.

  Args:
    coords: Iterable of coordinate iterables (shape (N, dim)).
    dim: Spatial dimension to validate against.

  Returns:
    A float array of shape (N, dim) with dtype DEFAULT_DTYPE.
  """
  arr = np.asarray(coords, dtype=DEFAULT_DTYPE)
  if arr.ndim != 2 or arr.shape[1] != dim:
    raise ValueError(f"coords must be a 2D array with shape (N, {dim})")
  if arr.shape[0] == 0:
    raise ValueError("coords must contain at least one point")
  if not np.isfinite(arr).all():
    raise ValueError("coords must be finite numbers")
  return arr


def validate_initial_conditions(
  coords: Iterable[Iterable[float]],
  dim: int,
  box_length: float,
  radius: float = 0.5,
) -> np.ndarray:
  """Validate initial sphere centers and reject overlaps or out-of-bounds.

  Uses a linked-cell (cell list) neighbor search to detect overlaps in
  near-linear time by only checking nearby cells instead of all pairs.

  Args:
    coords: Iterable of coordinate iterables (shape (N, dim)).
    dim: Spatial dimension (1, 2, or 3).
    box_length: Box side length; centers must lie within +/- box_length/2.
    radius: Sphere radius used for overlap and boundary checks.

  Returns:
    A validated float array of shape (N, dim).
  """
  if dim not in (1, 2, 3):
    raise ValueError("dim must be 1, 2, or 3")
  if box_length <= 0:
    raise ValueError("box_length must be positive")
  if radius <= 0:
    raise ValueError("radius must be positive")

  arr = _as_coords_array(coords, dim)
  half = box_length * 0.5
  limit = half - radius
  if limit < 0:
    raise ValueError("box_length is too small for the given radius")

  if np.any(np.abs(arr) > limit + 1e-12):
    raise ValueError("all sphere centers must be inside the box (including radius)")

  cell_size = 2.0 * radius
  inv_cell = 1.0 / cell_size
  grid: dict[tuple[int, ...], list[int]] = {}
  offsets = list(product([-1, 0, 1], repeat=dim))
  min_dist2 = (2.0 * radius) ** 2

  for idx, pos in enumerate(arr):
    cell = tuple(int(np.floor(p * inv_cell)) for p in pos)
    for delta in offsets:
      neighbor = tuple(c + d for c, d in zip(cell, delta))
      for j in grid.get(neighbor, []):
        diff = pos - arr[j]
        if float(np.dot(diff, diff)) < min_dist2 - 1e-12:
          raise ValueError("overlapping spheres detected")
    grid.setdefault(cell, []).append(idx)

  return arr


def has_overlap_at_array(
  pos: np.ndarray,
  coords: np.ndarray,
  cell_grid: np.ndarray,
  r: float,
  Lbox: float,
  Lcell: float,
  n_cells: int,
  max_cell_occupancy: int,
  neighbor_offsets: np.ndarray,
  cell_strides: np.ndarray,
  ignore_index: int,
) -> bool:
  """Check for overlaps at a proposed position using linked-cell data.

  Args:
    pos: Proposed center position (shape (dim,)).
    coords: All current sphere centers (shape (N, dim)).
    cell_grid: Flattened linked-cell grid with fixed occupancy slots.
    r: Sphere radius.
    Lbox: Box side length.
    Lcell: Cell edge length.
    n_cells: Number of cells per dimension.
    max_cell_occupancy: Max number of indices stored per cell.
    neighbor_offsets: Offsets for neighbor cells (shape (3**dim, dim)).
    cell_strides: Strides for flattening a cell index.
    ignore_index: Particle index to ignore (e.g., self when proposing move).

  Returns:
    True if any neighbor overlaps within 2r, otherwise False.
  """
  # Convert the position into a cell index in the linked-cell grid.
  half = Lbox * 0.5
  inv_Lcell = 1.0 / Lcell
  cell_idx = np.floor((pos + half) * inv_Lcell).astype(int)
  min_dist2 = (2.0 * r) ** 2

  # Scan neighbor cells and test squared distances against the hard-core radius.
  for delta in neighbor_offsets:
    neighbor = cell_idx + delta
    if np.any(neighbor < 0) or np.any(neighbor >= n_cells):
      continue
    neighbor_linear = int(np.dot(neighbor, cell_strides))
    for m in range(max_cell_occupancy):
      j = int(cell_grid[neighbor_linear, m])
      if j < 0 or j == ignore_index:
        continue
      diff = pos - coords[j]
      if float(np.dot(diff, diff)) < min_dist2 - 1e-12:
        return True
  return False


def cell_index_from_pos_array(pos: np.ndarray, Lbox: float, Lcell: float) -> np.ndarray:
  """Map a position to its cell index in the linked-cell grid.

  Args:
    pos: Center position (shape (dim,)).
    Lbox: Box side length.
    Lcell: Cell edge length.

  Returns:
    Integer cell index (shape (dim,)).
  """
  # Use box bounds to shift to [0, Lbox] and scale to cell coordinates.
  half = Lbox * 0.5
  inv_Lcell = 1.0 / Lcell
  return np.floor((pos + half) * inv_Lcell).astype(int)


def cell_strides_for_dim(n_cells: int, dim: int) -> np.ndarray:
  """Return row-major strides for flattening cell indices.

  Args:
    n_cells: Number of cells per dimension.
    dim: Spatial dimension.

  Returns:
    Integer strides (shape (dim,)) for row-major flattening.
  """
  # Strides convert a multi-index into a flat index: dot(cell_idx, strides).
  return np.array([n_cells ** (dim - 1 - i) for i in range(dim)], dtype=int)


def cell_linear_index(cell_idx: np.ndarray, cell_strides: np.ndarray) -> int:
  """Flatten a cell index using precomputed strides.

  Args:
    cell_idx: Integer cell index (shape (dim,)).
    cell_strides: Strides from cell_strides_for_dim (shape (dim,)).

  Returns:
    Flat integer index into the cell grid.
  """
  # Keep as a tiny helper so JIT and Python paths share the same indexing rule.
  return int(np.dot(cell_idx, cell_strides))


def clear_cell_slot_array(
  index: int,
  cell_grid: np.ndarray,
  cell_assignments: np.ndarray,
  cell_strides: np.ndarray,
  dim: int,
) -> None:
  """Clear a sphere slot in the linked-cell grid.

  Args:
    index: Particle index to clear.
    cell_grid: Flattened linked-cell grid with fixed occupancy slots.
    cell_assignments: Per-particle cell index + slot mapping.
    cell_strides: Strides for flattening a cell index.
    dim: Spatial dimension.
  """
  # Read the current cell index + slot and mark that slot empty.
  cell_idx = cell_assignments[index, :dim]
  slot = int(cell_assignments[index, dim])
  if slot < 0:
    return
  cell_linear = cell_linear_index(cell_idx, cell_strides)
  cell_grid[cell_linear, slot] = -1


def assign_cell_slot_array(
  index: int,
  cell_idx: np.ndarray,
  cell_grid: np.ndarray,
  cell_assignments: np.ndarray,
  cell_strides: np.ndarray,
  max_cell_occupancy: int,
) -> None:
  """Insert a sphere index into the first open slot of a cell.

  Args:
    index: Particle index to insert.
    cell_idx: Target cell index (shape (dim,)).
    cell_grid: Flattened linked-cell grid with fixed occupancy slots.
    cell_assignments: Per-particle cell index + slot mapping.
    cell_strides: Strides for flattening a cell index.
    max_cell_occupancy: Max number of indices stored per cell.
  """
  # Find the first empty slot in the cell and record the assignment.
  cell_linear = cell_linear_index(cell_idx, cell_strides)
  for m in range(max_cell_occupancy):
    if cell_grid[cell_linear, m] == -1:
      cell_grid[cell_linear, m] = index
      cell_assignments[index, : cell_idx.shape[0]] = cell_idx
      cell_assignments[index, cell_idx.shape[0]] = m
      return
  raise ValueError("cell grid is full for a sphere placement")


def time_step_core(
  coords: np.ndarray,
  cell_grid: np.ndarray,
  cell_assignments: np.ndarray,
  order: np.ndarray,
  directions: np.ndarray,
  dl: float,
  r: float,
  Lbox: float,
  Lcell: float,
  n_cells: int,
  max_cell_occupancy: int,
  neighbor_offsets: np.ndarray,
  cell_strides: np.ndarray,
) -> int:
  """Advance one Monte Carlo sweep using array-only inputs.

  Args:
    coords: Sphere centers (shape (N, dim)).
    cell_grid: Flattened linked-cell grid with fixed occupancy slots.
    cell_assignments: Per-particle cell index + slot mapping.
    order: Permutation of particle indices for the sweep.
    directions: Random unit directions (shape (N, dim)).
    dl: Step length for trial displacements.
    r: Sphere radius.
    Lbox: Box side length.
    Lcell: Cell edge length.
    n_cells: Number of cells per dimension.
    max_cell_occupancy: Max number of indices stored per cell.
    neighbor_offsets: Offsets for neighbor cells (shape (3**dim, dim)).
    cell_strides: Strides for flattening a cell index.

  Returns:
    Number of accepted moves in the sweep.
  """
  # Sweep particles in the provided order and attempt random displacements.
  dim = coords.shape[1]
  limit = Lbox * 0.5 - r
  moved = 0

  for idx, i in enumerate(order):
    # Build the trial displacement for this particle.
    proposed = coords[i] + dl * directions[idx]
    # Reject moves that push the center beyond the box (including radius).
    if np.any(np.abs(proposed) > limit + 1e-12):
      continue

    # Reject moves that overlap any neighbor in the linked-cell list.
    if has_overlap_at_array(
      pos=proposed,
      coords=coords,
      cell_grid=cell_grid,
      r=r,
      Lbox=Lbox,
      Lcell=Lcell,
      n_cells=n_cells,
      max_cell_occupancy=max_cell_occupancy,
      neighbor_offsets=neighbor_offsets,
      cell_strides=cell_strides,
      ignore_index=int(i),
    ):
      continue

    old_cell = cell_assignments[i, :dim]
    new_cell = cell_index_from_pos_array(proposed, Lbox, Lcell)
    # Update the linked-cell grid if the particle moved across cells.
    if not np.array_equal(new_cell, old_cell):
      clear_cell_slot_array(i, cell_grid, cell_assignments, cell_strides, dim)
      assign_cell_slot_array(
        i,
        new_cell,
        cell_grid,
        cell_assignments,
        cell_strides,
        max_cell_occupancy,
      )

    # Commit the accepted move into the coordinate array.
    coords[i] = proposed
    moved += 1

  return moved


if HAS_NUMBA:

  @nb.njit(cache=True)
  def _has_overlap_at_array_numba(
    pos: np.ndarray,
    coords: np.ndarray,
    cell_grid: np.ndarray,
    r: float,
    Lbox: float,
    Lcell: float,
    n_cells: int,
    max_cell_occupancy: int,
    neighbor_offsets: np.ndarray,
    cell_strides: np.ndarray,
    ignore_index: int,
  ) -> bool:
    """Numba-friendly overlap check with explicit loops and no Python objects.

    Args:
      pos: Proposed center position (shape (dim,)).
      coords: All current sphere centers (shape (N, dim)).
      cell_grid: Flattened linked-cell grid with fixed occupancy slots.
      r: Sphere radius.
      Lbox: Box side length.
      Lcell: Cell edge length.
      n_cells: Number of cells per dimension.
      max_cell_occupancy: Max number of indices stored per cell.
      neighbor_offsets: Offsets for neighbor cells (shape (3**dim, dim)).
      cell_strides: Strides for flattening a cell index.
      ignore_index: Particle index to ignore (e.g., self when proposing move).

    Returns:
      True if any neighbor overlaps within 2r, otherwise False.
    """
    half = Lbox * 0.5
    inv_Lcell = 1.0 / Lcell
    dim = pos.shape[0]
    cell_idx = np.empty(dim, dtype=np.int64)
    for d in range(dim):
      cell_idx[d] = int(np.floor((pos[d] + half) * inv_Lcell))
    min_dist2 = (2.0 * r) ** 2

    # Iterate neighbor cells in a flattened loop to avoid allocations.
    for k in range(neighbor_offsets.shape[0]):
      in_bounds = True
      neighbor_linear = 0
      for d in range(dim):
        neighbor = cell_idx[d] + neighbor_offsets[k, d]
        if neighbor < 0 or neighbor >= n_cells:
          in_bounds = False
          break
        neighbor_linear += neighbor * cell_strides[d]
      if not in_bounds:
        continue
      # Scan each slot in the neighbor cell for a conflicting sphere.
      for m in range(max_cell_occupancy):
        j = int(cell_grid[neighbor_linear, m])
        if j < 0 or j == ignore_index:
          continue
        dist2 = 0.0
        for d in range(dim):
          diff = pos[d] - coords[j, d]
          dist2 += diff * diff
        if dist2 < min_dist2 - 1e-12:
          return True
    return False


  @nb.njit(cache=True)
  def _time_step_core_numba(
    coords: np.ndarray,
    cell_grid: np.ndarray,
    cell_assignments: np.ndarray,
    order: np.ndarray,
    directions: np.ndarray,
    dl: float,
    r: float,
    Lbox: float,
    Lcell: float,
    n_cells: int,
    max_cell_occupancy: int,
    neighbor_offsets: np.ndarray,
    cell_strides: np.ndarray,
  ) -> int:
    """Numba-friendly sweep that mirrors the Python version but avoids allocations.

    Args:
      coords: Sphere centers (shape (N, dim)).
      cell_grid: Flattened linked-cell grid with fixed occupancy slots.
      cell_assignments: Per-particle cell index + slot mapping.
      order: Permutation of particle indices for the sweep.
      directions: Random unit directions (shape (N, dim)).
      dl: Step length for trial displacements.
      r: Sphere radius.
      Lbox: Box side length.
      Lcell: Cell edge length.
      n_cells: Number of cells per dimension.
      max_cell_occupancy: Max number of indices stored per cell.
      neighbor_offsets: Offsets for neighbor cells (shape (3**dim, dim)).
      cell_strides: Strides for flattening a cell index.

    Returns:
      Number of accepted moves in the sweep.
    """
    dim = coords.shape[1]
    limit = Lbox * 0.5 - r
    moved = 0

    for idx in range(order.shape[0]):
      i = order[idx]
      proposed = np.empty(dim, dtype=coords.dtype)
      out_of_bounds = False
      # Build the proposed position and reject if it exits the box.
      for d in range(dim):
        proposed_d = coords[i, d] + dl * directions[idx, d]
        proposed[d] = proposed_d
        if abs(proposed_d) > limit + 1e-12:
          out_of_bounds = True
          break
      if out_of_bounds:
        continue

      # Overlap test against neighbor cells.
      if _has_overlap_at_array_numba(
        proposed,
        coords,
        cell_grid,
        r,
        Lbox,
        Lcell,
        n_cells,
        max_cell_occupancy,
        neighbor_offsets,
        cell_strides,
        int(i),
      ):
        continue

      old_linear = 0
      new_linear = 0
      cell_changed = False
      # Compute old and new cell indices and check if a cell transfer is needed.
      for d in range(dim):
        old_cell = int(cell_assignments[i, d])
        new_cell = int(np.floor((proposed[d] + Lbox * 0.5) / Lcell))
        if new_cell != old_cell:
          cell_changed = True
        old_linear += old_cell * cell_strides[d]
        new_linear += new_cell * cell_strides[d]

      # Update the linked-cell grid when the particle crosses into a new cell.
      if cell_changed:
        slot = int(cell_assignments[i, dim])
        if slot >= 0:
          cell_grid[old_linear, slot] = -1
        # Insert into the first free slot in the new cell and record assignment.
        for m in range(max_cell_occupancy):
          if cell_grid[new_linear, m] == -1:
            cell_grid[new_linear, m] = int(i)
            for d in range(dim):
              cell_assignments[i, d] = int(np.floor((proposed[d] + Lbox * 0.5) / Lcell))
            cell_assignments[i, dim] = m
            break

      # Commit the move.
      for d in range(dim):
        coords[i, d] = proposed[d]
      moved += 1

    return moved


@dataclass
class TimeStepHistory:
  time: int
  dl: float
  Nm: int


@dataclass
class HardSpheresSimulation:
  """Hard-sphere Monte Carlo simulation with a linked-cell neighbor grid.

  The simulation stores a uniform spatial grid to accelerate overlap checks
  and performs random displacement moves for each time step.

  History caching:
    - history_enabled: when False, no history stats are stored or cached.
    - cache_history: when True, keep a fixed-size in-memory cache that is
      written to disk in cache_stride-sized chunks inside cache_dir (default .temp).
    - cache_stride: number of time steps to buffer before writing.
    - cache_mode: "sequence" writes separate files per chunk; "append" keeps
      a single file that is re-written with appended data.
    - cache_compress: use np.savez_compressed when True.
  """
  dim: int
  Lbox: float
  r: float = 0.5
  t: int = 0
  N: int = 0
  dl: float = 1.0
  #: Toggle storing TimeStepHistory in memory.
  history_enabled: bool = True
  #: Toggle writing TimeStepHistory to disk in cache_dir.
  cache_history: bool = False
  #: Number of steps to buffer before writing to disk.
  cache_stride: int = 1_000
  #: Cache file strategy: "sequence" or "append".
  cache_mode: str = "sequence"
  #: Use np.savez_compressed when True.
  cache_compress: bool = True
  #: Folder for cached data (cleared on creation when cache_history=True).
  cache_dir: str | Path = ".temp"
  #: Enable Numba-accelerated time stepping when available.
  use_numba: bool = False

  def __post_init__(self) -> None:
    if self.dim not in (1, 2, 3):
      raise ValueError("dim must be 1, 2, or 3")
    if self.Lbox <= 0:
      raise ValueError("Lbox must be positive")
    if self.r <= 0:
      raise ValueError("r must be positive")
    if not isinstance(self.t, int) or self.t < 0:
      raise ValueError("t must be a non-negative integer")
    if self.N < 0:
      raise ValueError("N must be non-negative")
    if self.dl <= 0:
      raise ValueError("dl must be positive")
    if self.cache_stride <= 0:
      raise ValueError("cache_stride must be positive")
    if self.cache_mode not in ("sequence", "append"):
      raise ValueError("cache_mode must be 'sequence' or 'append'")
    if self.use_numba and not HAS_NUMBA:
      raise ValueError("use_numba=True but numba is not installed")
    if not self.history_enabled:
      self.cache_history = False
    self.coords: np.ndarray | None = None
    self._time_step_history: list[TimeStepHistory] = []
    self._cache_times: np.ndarray | None = None
    self._cache_dls: np.ndarray | None = None
    self._cache_nms: np.ndarray | None = None
    self._cache_fill = 0
    self._cache_index = 0
    self.cellGrid: np.ndarray | None = None
    self.n_cells: int | None = None
    self.Lcell: float | None = None
    self.max_cell_occupancy: int | None = None
    self.cellStrides: np.ndarray | None = None
    self.cellAssignments: np.ndarray | None = None
    self._neighbor_offsets = list(product([-1, 0, 1], repeat=self.dim))
    self._neighbor_offsets_array = np.array(self._neighbor_offsets, dtype=int)
    self._dtype = DEFAULT_DTYPE

    self._cache_path = Path(self.cache_dir)
    if self.cache_history:
      self._reset_cache_dir()
      self._cache_times = np.empty(self.cache_stride, dtype=int)
      self._cache_dls = np.empty(self.cache_stride, dtype=self._dtype)
      self._cache_nms = np.empty(self.cache_stride, dtype=int)
    elif not self._cache_path.exists():
      self._cache_path.mkdir(parents=True, exist_ok=True)

  def _reset_cache_dir(self) -> None:
    if self._cache_path.exists():
      shutil.rmtree(self._cache_path)
    self._cache_path.mkdir(parents=True, exist_ok=True)

  @property
  def n_spheres(self) -> int:
    if self.coords is None:
      return int(self.N)
    return int(self.coords.shape[0])

  def _cell_index_from_pos(self, pos: np.ndarray) -> np.ndarray:
    """Map a position to its cell index in the linked-cell grid.

    Args:
      pos: Center position (shape (dim,)).

    Returns:
      Integer cell index (shape (dim,)).
    """
    if self.Lcell is None:
      raise ValueError("cell grid metadata is missing")
    # Shift to [0, Lbox] before converting to integer cell indices.
    half = self.Lbox * 0.5
    inv_Lcell = 1.0 / self.Lcell
    return np.floor((pos + half) * inv_Lcell).astype(int)

  def _iter_neighbor_cells(self, cell_idx: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    """Yield valid neighbor cells (including the cell itself).

    Args:
      cell_idx: Integer cell index (shape (dim,)).

    Yields:
      Neighbor cell indices within the box bounds.
    """
    if self.n_cells is None:
      raise ValueError("cell grid metadata is missing")
    # Iterate local neighbor offsets and skip any cells outside the domain.
    for delta in self._neighbor_offsets:
      neighbor = tuple(c + d for c, d in zip(cell_idx, delta))
      if any(c < 0 or c >= self.n_cells for c in neighbor):
        continue
      yield neighbor

  def _assign_cell_slot(self, index: int, cell_idx: np.ndarray) -> int:
    """Insert a sphere index into the first open slot of a cell.

    Args:
      index: Particle index to insert.
      cell_idx: Target cell index (shape (dim,)).

    Returns:
      The slot index where the particle was inserted.
    """
    if self.cellGrid is None or self.cellAssignments is None:
      raise ValueError("cell grid has not been initialized")
    if self.cellStrides is None:
      raise ValueError("cell grid metadata is missing")
    # Flatten the cell index and find the first empty slot for insertion.
    cell_linear = cell_linear_index(np.asarray(cell_idx, dtype=int), self.cellStrides)
    for m in range(self.max_cell_occupancy or 0):
      if self.cellGrid[cell_linear, m] == -1:
        self.cellGrid[cell_linear, m] = index
        self.cellAssignments[index, : self.dim] = np.array(cell_idx, dtype=int)
        self.cellAssignments[index, self.dim] = m
        return m
    raise ValueError("cell grid is full for a sphere placement")

  def _clear_cell_slot(self, index: int) -> None:
    """Clear a sphere's slot in the linked-cell grid.

    Args:
      index: Particle index to clear.
    """
    if self.cellGrid is None or self.cellAssignments is None:
      raise ValueError("cell grid has not been initialized")
    if self.cellStrides is None:
      raise ValueError("cell grid metadata is missing")
    # Look up the recorded cell + slot and mark it empty.
    cell_idx = np.asarray(self.cellAssignments[index, : self.dim], dtype=int)
    slot = int(self.cellAssignments[index, self.dim])
    if slot < 0:
      return
    cell_linear = cell_linear_index(cell_idx, self.cellStrides)
    self.cellGrid[cell_linear, slot] = -1

  def _has_overlap_at(self, pos: np.ndarray, ignore_index: int) -> bool:
    """Check for overlaps at a proposed position using neighbor cells.

    Args:
      pos: Proposed center position (shape (dim,)).
      ignore_index: Particle index to ignore (e.g., self when proposing move).

    Returns:
      True if any neighbor overlaps within 2r, otherwise False.
    """
    if self.coords is None:
      raise ValueError("initial conditions have not been set")
    if self.cellGrid is None:
      raise ValueError("cell grid has not been initialized")
    if self.cellAssignments is None or self.n_cells is None or self.Lcell is None:
      raise ValueError("cell grid metadata is missing")
    if self.cellStrides is None:
      raise ValueError("cell grid metadata is missing")
    # Delegate to the array-based helper so both Python/Numba share logic.
    max_occ = self.max_cell_occupancy or 0
    return has_overlap_at_array(
      pos=pos,
      coords=self.coords,
      cell_grid=self.cellGrid,
      r=self.r,
      Lbox=self.Lbox,
      Lcell=self.Lcell,
      n_cells=self.n_cells,
      max_cell_occupancy=max_occ,
      neighbor_offsets=self._neighbor_offsets_array,
      cell_strides=self.cellStrides,
      ignore_index=ignore_index,
    )

  def CheckCellGridIntegrity(self) -> None:
    """Validate that the linked-cell grid references each sphere exactly once.

    Raises:
      ValueError if indices are invalid or if any particle is missing/duplicated.
    """
    if self.coords is None:
      raise ValueError("initial conditions have not been set")
    if self.cellGrid is None or self.cellAssignments is None:
      raise ValueError("cell grid has not been initialized")
    if self.cellStrides is None:
      raise ValueError("cell grid metadata is missing")

    counts = np.zeros(self.N, dtype=int)
    for value in np.nditer(self.cellGrid):
      idx = int(value)
      if idx < 0:
        continue
      if idx >= self.N:
        raise ValueError("cell grid contains invalid indices")
      counts[idx] += 1

    if not np.all(counts == 1):
      raise ValueError("cell grid does not reference each sphere exactly once")

    for i in range(self.N):
      cell_idx = self.cellAssignments[i, : self.dim]
      slot = int(self.cellAssignments[i, self.dim])
      if slot < 0:
        raise ValueError("cell assignment slot is invalid")
      cell_linear = cell_linear_index(cell_idx, self.cellStrides)
      if int(self.cellGrid[cell_linear, slot]) != i:
        raise ValueError("cell assignment does not match the cell grid")

  def SetInitialConditions(
    self,
    coords: Iterable[Iterable[float]] | None = None,
    *,
    N: int | None = None,
    randomInit: bool = False,
    max_rand_init_iterations: int = 100_000,
    seed: int | None = None,
  ) -> None:
    """Initialize the simulation from explicit coordinates or random placement.

    When randomInit is True, spheres are placed sequentially with rejection
    sampling against the linked-cell grid. Each sphere gets up to
    max_rand_init_iterations random trials before giving up.

    Args:
      coords: Explicit coordinates for all spheres (shape (N, dim)).
      N: Number of spheres when randomInit is True.
      randomInit: If True, randomly place spheres with rejection sampling.
      max_rand_init_iterations: Max trials per sphere in random placement.
      seed: RNG seed for reproducible random placement.
    """
    if randomInit:
      if N is None or N <= 0:
        raise ValueError("N must be a positive integer when randomInit is True")
      if max_rand_init_iterations <= 0:
        raise ValueError("max_rand_init_iterations must be positive")
      if self.Lbox * 0.5 - self.r < 0:
        raise ValueError("Lbox is too small for the given radius")
    elif coords is None:
      raise ValueError("coords must be provided when randomInit is False")

    if randomInit:
      self.N = int(N)
      self.coords = np.zeros((self.N, self.dim), dtype=self._dtype)
    else:
      self.coords = validate_initial_conditions(
        coords, self.dim, self.Lbox, self.r
      )
      self.N = int(self.coords.shape[0])

    n_cells = int(np.floor(self.Lbox / (2.0 * self.r)))
    Lcell = self.Lbox / n_cells
    max_occupancy = 2**self.dim + 1
    self.cellStrides = cell_strides_for_dim(n_cells, self.dim)
    self.cellGrid = np.full((n_cells ** self.dim, max_occupancy), -1, dtype=int)
    self.n_cells = n_cells
    self.Lcell = Lcell
    self.max_cell_occupancy = max_occupancy
    self.cellAssignments = np.full((self.N, self.dim + 1), -1, dtype=int)

    if randomInit:
      rng = np.random.default_rng(seed)
      limit = self._dtype(self.Lbox * 0.5 - self.r)
      for i in range(self.N):
        placed = False
        for _ in range(max_rand_init_iterations):
          pos = rng.uniform(-limit, limit, size=self.dim).astype(self._dtype, copy=False)
          if not self._has_overlap_at(pos, ignore_index=-1):
            self.coords[i] = pos
            cell_idx = self._cell_index_from_pos(pos)
            self._assign_cell_slot(i, cell_idx)
            placed = True
            break
        if not placed:
          raise ValueError(
            "unable to place sphere without overlap; "
            "increase max_rand_init_iterations or reduce density"
          )
    else:
      for i, pos in enumerate(self.coords):
        cell_idx = self._cell_index_from_pos(pos)
        self._assign_cell_slot(i, cell_idx)

  def _random_direction(self, rng: np.random.Generator) -> np.ndarray:
    """Generate a random unit direction in 1D/2D/3D.

    Args:
      rng: NumPy random generator.

    Returns:
      Unit direction vector (shape (dim,)).
    """
    # 1D uses a random sign; higher dimensions draw from isotropic normals.
    if self.dim == 1:
      return np.array([1.0 if rng.random() < 0.5 else -1.0], dtype=self._dtype)

    vec = rng.normal(size=self.dim).astype(self._dtype, copy=False)
    norm = float(np.linalg.norm(vec))
    while norm == 0.0:
      vec = rng.normal(size=self.dim).astype(self._dtype, copy=False)
      norm = float(np.linalg.norm(vec))
    return vec / self._dtype(norm)

  def _random_directions(self, rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate a batch of random unit directions in 1D/2D/3D.

    Args:
      rng: NumPy random generator.
      n: Number of directions to generate.

    Returns:
      Array of unit directions (shape (n, dim)).
    """
    if n <= 0:
      return np.empty((0, self.dim), dtype=self._dtype)
    # Use closed-form sampling in 2D/3D to avoid slow rejection loops.
    if self.dim == 1:
      signs = np.where(rng.random(n) < 0.5, -1.0, 1.0)
      return signs.astype(self._dtype, copy=False).reshape(-1, 1)
    if self.dim == 2:
      theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
      return np.stack((np.cos(theta), np.sin(theta)), axis=1).astype(
        self._dtype, copy=False
      )
    u = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - u * u)
    return np.stack((s * np.cos(phi), s * np.sin(phi), u), axis=1).astype(
      self._dtype, copy=False
    )

  def MeanPseudoFreePath(
    self, n_directions: int, step: float, seed: int | None = None
  ) -> float:
    """Estimate the mean pseudo free path from random trial displacements.

    Args:
      n_directions: Number of random directions sampled per particle.
      step: Trial step length for displacement.
      seed: RNG seed for reproducibility.

    Returns:
      Mean accepted trial step length across all samples.
    """
    if self.coords is None:
      raise ValueError("initial conditions have not been set")
    if n_directions <= 0:
      raise ValueError("n_directions must be positive")
    if step <= 0:
      raise ValueError("step must be positive")

    rng = np.random.default_rng(seed)
    coords = self.coords
    limit = self._dtype(self.Lbox * 0.5 - self.r)
    min_dist2 = self._dtype((2.0 * self.r) ** 2)

    total = 0.0
    n_particles = coords.shape[0]

    for i in range(n_particles):
      origin = coords[i]
      for _ in range(n_directions):
        # Draw a random direction and test a single step from the origin.
        direction = self._random_direction(rng)

        new_pos = origin + self._dtype(step) * direction
        if np.any(np.abs(new_pos) > limit + 1e-12):
          continue

        diff = coords - new_pos
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dist2[i] = np.inf
        if np.any(dist2 < min_dist2 - 1e-12):
          continue

        total += step

    return total / (n_particles * n_directions)

  def TimeStep(self, seed: int | None = None) -> None:
    """Advance the simulation by one Monte Carlo sweep.

    Uses a Metropolis-style displacement move for hard spheres: trial moves
    are accepted when no overlaps occur.

    References:
      Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H.,
      & Teller, E. (1953). Equation of state calculations by fast computing
      machines. Journal of Chemical Physics, 21(6), 1087-1092.

    Args:
      seed: RNG seed for reproducible sweep order and directions.
    """
    if self.coords is None:
      raise ValueError("initial conditions have not been set")
    if self.cellAssignments is None or self.cellGrid is None:
      raise ValueError("cell grid has not been initialized")
    if self.n_cells is None or self.Lcell is None:
      raise ValueError("cell grid metadata is missing")
    if self.cellStrides is None:
      raise ValueError("cell grid metadata is missing")
    if self.cellStrides is None:
      raise ValueError("cell grid metadata is missing")

    # Randomize sweep order and generate an isotropic direction per particle.
    rng = np.random.default_rng(seed)
    coords = self.coords
    n_particles = coords.shape[0]
    order = rng.permutation(n_particles)
    directions = self._random_directions(rng, n_particles)
    # Run either the Numba-accelerated kernel or the pure NumPy version.
    if self.use_numba:
      moved = _time_step_core_numba(
        coords,
        self.cellGrid,
        self.cellAssignments,
        order,
        directions,
        float(self.dl),
        float(self.r),
        float(self.Lbox),
        float(self.Lcell),
        int(self.n_cells),
        int(self.max_cell_occupancy or 0),
        self._neighbor_offsets_array,
        self.cellStrides,
      )
    else:
      moved = time_step_core(
        coords=coords,
        cell_grid=self.cellGrid,
        cell_assignments=self.cellAssignments,
        order=order,
        directions=directions,
        dl=float(self.dl),
        r=float(self.r),
        Lbox=float(self.Lbox),
        Lcell=float(self.Lcell),
        n_cells=int(self.n_cells),
        max_cell_occupancy=int(self.max_cell_occupancy or 0),
        neighbor_offsets=self._neighbor_offsets_array,
        cell_strides=self.cellStrides,
      )

    # Bookkeeping for accepted moves and time index.
    self.coords = coords
    self.N = int(coords.shape[0])
    self.t += 1

    if self.history_enabled:
      # Store the sweep statistics either in memory or in a rolling cache.
      history = TimeStepHistory(time=self.t, dl=float(self.dl), Nm=moved)
      if self.cache_history:
        if self._cache_times is None or self._cache_dls is None or self._cache_nms is None:
          raise ValueError("history cache arrays are not initialized")
        self._cache_times[self._cache_fill] = history.time
        self._cache_dls[self._cache_fill] = history.dl
        self._cache_nms[self._cache_fill] = history.Nm
        self._cache_fill += 1
        if self._cache_fill >= self.cache_stride:
          self._flush_history_cache()
      else:
        self._time_step_history.append(history)
    return None

  def _flush_history_cache(self) -> None:
    """Write buffered history data to disk based on the cache mode."""
    if not self.cache_history or self._cache_fill == 0:
      return
    if self._cache_times is None or self._cache_dls is None or self._cache_nms is None:
      raise ValueError("history cache arrays are not initialized")

    times = self._cache_times[: self._cache_fill].copy()
    dls = self._cache_dls[: self._cache_fill].copy()
    moves = self._cache_nms[: self._cache_fill].copy()

    if self.cache_mode == "append":
      cache_file = self._cache_path / "history.npz"
      if cache_file.exists():
        existing = np.load(cache_file)
        times = np.concatenate([existing["time"], times])
        dls = np.concatenate([existing["dl"], dls])
        moves = np.concatenate([existing["Nm"], moves])
    else:
      cache_file = self._cache_path / f"history_{self._cache_index:06d}.npz"
      self._cache_index += 1

    if self.cache_compress:
      np.savez_compressed(cache_file, time=times, dl=dls, Nm=moves)
    else:
      np.savez(cache_file, time=times, dl=dls, Nm=moves)

    self._cache_fill = 0

  def FlushHistoryCache(self) -> None:
    """Persist any cached history entries to disk immediately."""
    self._flush_history_cache()

  def GetTimeStepHistory(self) -> list[TimeStepHistory]:
    """Return full history from cache plus in-memory entries.

    Returns:
      List of history records sorted by time.
    """
    if not self.history_enabled and not self.cache_history:
      return []
    cached = self._load_cached_history()
    pending: list[TimeStepHistory] = []
    if self.cache_history and self._cache_fill > 0:
      if self._cache_times is None or self._cache_dls is None or self._cache_nms is None:
        raise ValueError("history cache arrays are not initialized")
      for idx in range(self._cache_fill):
        pending.append(
          TimeStepHistory(
            time=int(self._cache_times[idx]),
            dl=float(self._cache_dls[idx]),
            Nm=int(self._cache_nms[idx]),
          )
        )
    in_mem = self._time_step_history if self.history_enabled and not self.cache_history else []

    merged: dict[int, TimeStepHistory] = {}
    for history in cached + pending + in_mem:
      merged[history.time] = history

    return [merged[key] for key in sorted(merged)]

  @property
  def history(self) -> list[TimeStepHistory]:
    """Public history accessor for cached + in-memory entries.

    Returns:
      List of history records sorted by time.
    """
    return self.GetTimeStepHistory()

  def _load_cached_history(self) -> list[TimeStepHistory]:
    """Load cached history records from disk.

    Returns:
      List of history records found in the cache files.
    """
    if not self.cache_history or not self._cache_path.exists():
      return []

    histories: list[TimeStepHistory] = []
    if self.cache_mode == "append":
      cache_file = self._cache_path / "history.npz"
      if not cache_file.exists():
        return []
      data = np.load(cache_file)
      for time, dl, nm in zip(data["time"], data["dl"], data["Nm"]):
        histories.append(TimeStepHistory(time=int(time), dl=float(dl), Nm=int(nm)))
      return histories

    for cache_file in sorted(self._cache_path.glob("history_*.npz")):
      data = np.load(cache_file)
      for time, dl, nm in zip(data["time"], data["dl"], data["Nm"]):
        histories.append(TimeStepHistory(time=int(time), dl=float(dl), Nm=int(nm)))
    return histories

  def DetectOverlaps(self) -> bool:
    """Return True if any overlaps exist in the current configuration.

    Returns:
      True if any pair of spheres overlaps, otherwise False.
    """
    if self.coords is None:
      raise ValueError("initial conditions have not been set")
    if self.cellAssignments is None or self.cellGrid is None:
      raise ValueError("cell grid has not been initialized")
    if self.n_cells is None or self.Lcell is None:
      raise ValueError("cell grid metadata is missing")

    coords = self.coords
    min_dist2 = (2.0 * self.r) ** 2

    for i in range(self.N):
      cell_idx = self.cellAssignments[i, : self.dim]
      for delta in self._neighbor_offsets_array:
        neighbor = cell_idx + delta
        if np.any(neighbor < 0) or np.any(neighbor >= self.n_cells):
          continue
        neighbor_linear = cell_linear_index(neighbor, self.cellStrides)
        for m in range(self.max_cell_occupancy or 0):
          j = int(self.cellGrid[neighbor_linear, m])
          if j < 0 or j == i or j < i:
            continue
          diff = coords[i] - coords[j]
          if float(np.dot(diff, diff)) < min_dist2 - 1e-12:
            return True

    return False
