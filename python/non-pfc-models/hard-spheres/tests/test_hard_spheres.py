from pathlib import Path

import numpy as np
import pytest

import hard_spheres
from hard_spheres import HardSpheresSimulation, validate_initial_conditions


class _StubRng:
    def __init__(self, order: list[int]) -> None:
        self._order = np.array(order, dtype=int)

    def permutation(self, n: int) -> np.ndarray:
        return self._order.copy()


def _direction_sequence(directions: list[list[float]]):
    iterator = iter(directions)

    def _next(_rng: np.random.Generator) -> np.ndarray:
        return np.array(next(iterator), dtype=float)

    return _next


def _direction_batch_sequence(directions: list[list[float]]):
    iterator = iter(directions)

    def _next(_rng: np.random.Generator, n: int) -> np.ndarray:
        batch = [np.array(next(iterator), dtype=float) for _ in range(n)]
        if not batch:
            return np.empty((0, 0), dtype=float)
        return np.stack(batch, axis=0)

    return _next


def _make_sim(*args, **kwargs) -> HardSpheresSimulation:
    kwargs.setdefault("history_enabled", False)
    kwargs.setdefault("cache_history", False)
    return HardSpheresSimulation(*args, **kwargs)


def test_valid_1d_setup() -> None:
    sim = _make_sim(dim=1, Lbox=10.0)
    sim.SetInitialConditions([[-4.0], [0.0], [4.0]])
    assert sim.n_spheres == 3


def test_valid_2d_setup() -> None:
    sim = _make_sim(dim=2, Lbox=6.0)
    sim.SetInitialConditions([[-2.0, -2.0], [2.0, 2.0]])
    assert sim.coords is not None


def test_valid_3d_setup() -> None:
    sim = _make_sim(dim=3, Lbox=8.0)
    sim.SetInitialConditions([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert sim.n_spheres == 2


@pytest.mark.parametrize(
    "dim,Lbox,expected_n_cells,expected_Lcell,expected_max_occupancy,expected_shape",
    [
        (1, 10.0, 10, 1.0, 3, (10, 3)),
        (1, 10.5, 10, 1.05, 3, (10, 3)),
        (2, 10.0, 10, 1.0, 5, (100, 5)),
        (2, 10.5, 10, 1.05, 5, (100, 5)),
        (3, 10.0, 10, 1.0, 9, (1000, 9)),
        (3, 10.5, 10, 1.05, 9, (1000, 9)),
    ],
)
def test_cell_grid_dimensions_and_spacing(
    dim: int,
    Lbox: float,
    expected_n_cells: int,
    expected_Lcell: float,
    expected_max_occupancy: int,
    expected_shape: tuple[int, ...],
) -> None:
    sim = _make_sim(dim=dim, Lbox=Lbox, r=0.5)
    sim.SetInitialConditions([[0.0] * dim])

    assert sim.n_cells == expected_n_cells
    assert sim.Lcell == expected_Lcell
    assert sim.max_cell_occupancy == expected_max_occupancy
    assert sim.cellGrid is not None
    assert sim.cellGrid.shape == expected_shape

    assert sim.Lcell >= 1.0
    assert sim.Lcell <= 2.0


def test_reject_overlaps() -> None:
    with pytest.raises(ValueError):
        validate_initial_conditions([[0.0, 0.0], [0.4, 0.0]], 2, 5.0)


def test_reject_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        validate_initial_conditions([[4.8, 0.0]], 2, 10.0)


def test_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        validate_initial_conditions([1.0, 2.0], 1, 5.0)
    with pytest.raises(ValueError):
        HardSpheresSimulation(dim=4, Lbox=5.0)


@pytest.mark.parametrize("bad_Lbox", [-1.0, 0.0])
def test_reject_invalid_box_length(bad_Lbox: float) -> None:
    with pytest.raises(ValueError):
        HardSpheresSimulation(dim=2, Lbox=bad_Lbox)
    with pytest.raises(ValueError):
        validate_initial_conditions([[0.0, 0.0]], 2, bad_Lbox)


@pytest.mark.parametrize("bad_radius", [-1.0, 0.0])
def test_reject_invalid_radius(bad_radius: float) -> None:
    with pytest.raises(ValueError):
        HardSpheresSimulation(dim=2, Lbox=5.0, r=bad_radius)
    with pytest.raises(ValueError):
        validate_initial_conditions([[0.0, 0.0]], 2, 5.0, bad_radius)


def test_reject_box_too_small_for_radius() -> None:
    with pytest.raises(ValueError):
        validate_initial_conditions([[0.0, 0.0]], 2, 0.5, 1.0)


def test_cell_assignments_match_grid_1d() -> None:
    sim = _make_sim(dim=1, Lbox=10.0, r=0.5)
    sim.SetInitialConditions([[-4.0], [0.0], [4.0]])

    assert sim.cellAssignments is not None
    assert sim.cellGrid is not None
    assert sim.cellAssignments.tolist() == [[1, 0], [5, 0], [9, 0]]
    assert sim.cellGrid[1, 0] == 0
    assert sim.cellGrid[5, 0] == 1
    assert sim.cellGrid[9, 0] == 2


def test_cell_assignments_match_grid_2d() -> None:
    sim = _make_sim(dim=2, Lbox=10.0, r=0.5)
    sim.SetInitialConditions([[-4.0, -4.0], [0.0, 0.0], [4.0, 4.0]])

    assert sim.cellAssignments is not None
    assert sim.cellGrid is not None
    assert sim.cellAssignments.tolist() == [[1, 1, 0], [5, 5, 0], [9, 9, 0]]
    assert sim.cellGrid[11, 0] == 0
    assert sim.cellGrid[55, 0] == 1
    assert sim.cellGrid[99, 0] == 2


def test_cell_assignments_match_grid_3d() -> None:
    sim = _make_sim(dim=3, Lbox=10.0, r=0.5)
    sim.SetInitialConditions(
        [[-4.0, -4.0, -4.0], [0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]
    )

    assert sim.cellAssignments is not None
    assert sim.cellGrid is not None
    assert sim.cellAssignments.tolist() == [
        [1, 1, 1, 0],
        [5, 5, 5, 0],
        [9, 9, 9, 0],
    ]
    assert sim.cellGrid[111, 0] == 0
    assert sim.cellGrid[555, 0] == 1
    assert sim.cellGrid[999, 0] == 2


@pytest.mark.parametrize(
    "dim,Lbox,expected_max",
    [
        (1, 1.5, 3),
        (2, 1.5, 5),
        (3, 1.5, 9),
    ],
)
def test_cell_grid_overflow_raises(
    dim: int, Lbox: float, expected_max: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    coords = np.zeros((expected_max + 1, dim), dtype=float)

    def _passthrough_validate(
        coords_in: np.ndarray,
        dim_in: int,
        box_length: float,
        radius: float = 0.5,
    ) -> np.ndarray:
        return np.asarray(coords_in, dtype=float)

    monkeypatch.setattr(hard_spheres, "validate_initial_conditions", _passthrough_validate)

    sim = _make_sim(dim=dim, Lbox=Lbox, r=0.5)
    with pytest.raises(ValueError, match="cell grid is full"):
        sim.SetInitialConditions(coords)


def test_detect_overlaps_returns_false_for_valid_setup() -> None:
    sim = _make_sim(dim=2, Lbox=10.0, r=0.5)
    sim.SetInitialConditions([[-4.0, -4.0], [0.0, 0.0], [4.0, 4.0]])
    assert sim.DetectOverlaps() is False


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_detect_overlaps_returns_true_for_overlap(
    dim: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    coords = np.zeros((2, dim), dtype=float)

    def _passthrough_validate(
        coords_in: np.ndarray,
        dim_in: int,
        box_length: float,
        radius: float = 0.5,
    ) -> np.ndarray:
        return np.asarray(coords_in, dtype=float)

    monkeypatch.setattr(hard_spheres, "validate_initial_conditions", _passthrough_validate)

    sim = _make_sim(dim=dim, Lbox=2.0, r=0.5)
    sim.SetInitialConditions(coords)
    assert sim.DetectOverlaps() is True


def test_detect_overlaps_neighbor_corner_2d(monkeypatch: pytest.MonkeyPatch) -> None:
    coords = np.array([[-4.5, -4.5], [-4.0, -4.0]], dtype=float)

    def _passthrough_validate(
        coords_in: np.ndarray,
        dim_in: int,
        box_length: float,
        radius: float = 0.5,
    ) -> np.ndarray:
        return np.asarray(coords_in, dtype=float)

    monkeypatch.setattr(hard_spheres, "validate_initial_conditions", _passthrough_validate)

    sim = _make_sim(dim=2, Lbox=10.0, r=0.5)
    sim.SetInitialConditions(coords)
    assert sim.DetectOverlaps() is True


def test_detect_overlaps_neighbor_edge_2d(monkeypatch: pytest.MonkeyPatch) -> None:
    coords = np.array([[-4.5, -4.5], [-4.0, -4.5]], dtype=float)

    def _passthrough_validate(
        coords_in: np.ndarray,
        dim_in: int,
        box_length: float,
        radius: float = 0.5,
    ) -> np.ndarray:
        return np.asarray(coords_in, dtype=float)

    monkeypatch.setattr(hard_spheres, "validate_initial_conditions", _passthrough_validate)

    sim = _make_sim(dim=2, Lbox=10.0, r=0.5)
    sim.SetInitialConditions(coords)
    assert sim.DetectOverlaps() is True


def test_detect_overlaps_neighbor_corner_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    coords = np.array([[-4.5, -4.5, -4.5], [-4.0, -4.0, -4.0]], dtype=float)

    def _passthrough_validate(
        coords_in: np.ndarray,
        dim_in: int,
        box_length: float,
        radius: float = 0.5,
    ) -> np.ndarray:
        return np.asarray(coords_in, dtype=float)

    monkeypatch.setattr(hard_spheres, "validate_initial_conditions", _passthrough_validate)

    sim = _make_sim(dim=3, Lbox=10.0, r=0.5)
    sim.SetInitialConditions(coords)
    assert sim.DetectOverlaps() is True


def test_detect_overlaps_neighbor_face_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    coords = np.array([[-4.5, -4.5, -4.5], [-4.0, -4.5, -4.5]], dtype=float)

    def _passthrough_validate(
        coords_in: np.ndarray,
        dim_in: int,
        box_length: float,
        radius: float = 0.5,
    ) -> np.ndarray:
        return np.asarray(coords_in, dtype=float)

    monkeypatch.setattr(hard_spheres, "validate_initial_conditions", _passthrough_validate)

    sim = _make_sim(dim=3, Lbox=10.0, r=0.5)
    sim.SetInitialConditions(coords)
    assert sim.DetectOverlaps() is True


def test_detect_overlaps_neighbor_edge_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    coords = np.array([[-4.5, -4.5, -4.5], [-4.0, -4.0, -4.5]], dtype=float)

    def _passthrough_validate(
        coords_in: np.ndarray,
        dim_in: int,
        box_length: float,
        radius: float = 0.5,
    ) -> np.ndarray:
        return np.asarray(coords_in, dtype=float)

    monkeypatch.setattr(hard_spheres, "validate_initial_conditions", _passthrough_validate)

    sim = _make_sim(dim=3, Lbox=10.0, r=0.5)
    sim.SetInitialConditions(coords)
    assert sim.DetectOverlaps() is True


def test_time_step_rejects_overlapping_move_and_preserves_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = np.array([[-4.0, 0.0], [-2.0, 0.0]], dtype=float)
    sim = _make_sim(dim=2, Lbox=10.0, r=0.5, dl=1.5, history_enabled=True)
    sim.SetInitialConditions(coords)
    sim.CheckCellGridIntegrity()

    before_coords = sim.coords.copy()
    before_grid = sim.cellGrid.copy()
    before_assignments = sim.cellAssignments.copy()

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _StubRng([0, 1]))
    monkeypatch.setattr(
        sim,
        "_random_directions",
        _direction_batch_sequence([[1.0, 0.0], [-1.0, 0.0]]),
    )

    sim.TimeStep(seed=1)
    history = sim.history[-1]

    assert history.Nm == 0
    assert np.allclose(sim.coords, before_coords)
    assert np.array_equal(sim.cellGrid, before_grid)
    assert np.array_equal(sim.cellAssignments, before_assignments)
    sim.CheckCellGridIntegrity()


def test_time_step_accepts_move_and_updates_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = np.array([[-4.5, -4.5], [4.5, 4.5]], dtype=float)
    sim = _make_sim(dim=2, Lbox=10.0, r=0.5, dl=1.2, history_enabled=True)
    sim.SetInitialConditions(coords)
    sim.CheckCellGridIntegrity()

    old_cell = tuple(int(v) for v in sim.cellAssignments[0, :2])
    old_slot = int(sim.cellAssignments[0, 2])

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _StubRng([0, 1]))
    monkeypatch.setattr(
        sim,
        "_random_directions",
        _direction_batch_sequence([[1.0, 0.0], [1.0, 0.0]]),
    )

    sim.TimeStep(seed=1)
    history = sim.history[-1]

    assert history.Nm == 1
    assert np.allclose(sim.coords[0], np.array([-3.3, -4.5]))
    assert np.allclose(sim.coords[1], coords[1])

    new_cell = tuple(int(v) for v in sim.cellAssignments[0, :2])
    new_slot = int(sim.cellAssignments[0, 2])
    assert new_cell != old_cell
    old_linear = int(np.dot(np.array(old_cell, dtype=int), sim.cellStrides))
    new_linear = int(np.dot(np.array(new_cell, dtype=int), sim.cellStrides))
    assert sim.cellGrid[old_linear, old_slot] == -1
    assert sim.cellGrid[new_linear, new_slot] == 0
    sim.CheckCellGridIntegrity()


def test_time_step_moves_six_spheres_out_and_back_3d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    sim = _make_sim(dim=3, Lbox=10.1, r=0.5, dl=1.2, history_enabled=True)
    sim.SetInitialConditions(coords)
    sim.CheckCellGridIntegrity()

    original_cells = [
        tuple(int(v) for v in sim.cellAssignments[i, :3])
        for i in range(sim.N)
    ]
    assert len(set(original_cells)) == 1

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _StubRng([4, 5, 2, 3, 0, 1]))
    monkeypatch.setattr(
        sim,
        "_random_directions",
        _direction_batch_sequence([[1.0, 0.0, 0.0]] * 6),
    )

    sim.TimeStep(seed=1)
    history_out = sim.history[-1]
    assert history_out.Nm == 6
    sim.CheckCellGridIntegrity()

    for i in range(sim.N):
        cell_idx = tuple(int(v) for v in sim.cellAssignments[i, :3])
        assert cell_idx != original_cells[i]

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _StubRng([0, 1, 2, 3, 4, 5]))
    monkeypatch.setattr(
        sim,
        "_random_directions",
        _direction_batch_sequence([[-1.0, 0.0, 0.0]] * 6),
    )

    sim.TimeStep(seed=2)
    history_back = sim.history[-1]
    assert history_back.Nm == 6
    sim.CheckCellGridIntegrity()
    assert np.allclose(sim.coords, coords, atol=1e-6)

    for i in range(sim.N):
        cell_idx = tuple(int(v) for v in sim.cellAssignments[i, :3])
        assert cell_idx == original_cells[i]


def test_time_step_history_records_multiple_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = _make_sim(dim=1, Lbox=10.0, r=0.5, dl=1.0, history_enabled=True)
    sim.SetInitialConditions([[0.0]])

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _StubRng([0]))
    monkeypatch.setattr(
        sim,
        "_random_directions",
        _direction_batch_sequence([[1.0], [-1.0], [1.0]]),
    )

    sim.TimeStep(seed=1)
    sim.TimeStep(seed=2)
    sim.TimeStep(seed=3)
    histories = sim.history

    assert sim.t == 3
    assert len(sim.history) == 3
    assert [entry.time for entry in sim.history] == [1, 2, 3]
    assert [entry.dl for entry in histories] == [1.0, 1.0, 1.0]
    assert [entry.Nm for entry in histories] == [1, 1, 1]


def _setup_cache_sim(
    tmp_path: Path,
    *,
    cache_mode: str,
    cache_stride: int,
    history_enabled: bool,
) -> HardSpheresSimulation:
    cache_dir = tmp_path / "cache"
    sim = HardSpheresSimulation(
        dim=1,
        Lbox=10.0,
        r=0.5,
        dl=1.0,
        history_enabled=history_enabled,
        cache_history=True,
        cache_stride=cache_stride,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
    )
    sim.SetInitialConditions([[0.0]])
    return sim


def _drive_steps(
    sim: HardSpheresSimulation,
    monkeypatch: pytest.MonkeyPatch,
    n_steps: int,
) -> None:
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _StubRng([0]))
    monkeypatch.setattr(
        sim,
        "_random_directions",
        _direction_batch_sequence([[1.0]] * n_steps),
    )
    for step in range(n_steps):
        sim.TimeStep(seed=step)


def test_cache_dir_created_and_cleared(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "stale.txt").write_text("old", encoding="utf-8")

    sim = HardSpheresSimulation(
        dim=1,
        Lbox=10.0,
        cache_history=True,
        cache_dir=cache_dir,
    )

    assert cache_dir.exists()
    assert not (cache_dir / "stale.txt").exists()
    assert sim.cache_history is True


def test_cache_sequence_mode_writes_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sim = _setup_cache_sim(
        tmp_path,
        cache_mode="sequence",
        cache_stride=2,
        history_enabled=True,
    )

    _drive_steps(sim, monkeypatch, n_steps=5)
    sim.FlushHistoryCache()

    cache_files = sorted((tmp_path / "cache").glob("history_*.npz"))
    assert len(cache_files) == 3

    history = sim.GetTimeStepHistory()
    assert [entry.time for entry in history] == [1, 2, 3, 4, 5]
    assert [entry.Nm for entry in history] == [1, 1, 1, 1, 0]


def test_cache_append_mode_uses_single_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sim = _setup_cache_sim(
        tmp_path,
        cache_mode="append",
        cache_stride=2,
        history_enabled=True,
    )

    _drive_steps(sim, monkeypatch, n_steps=3)
    sim.FlushHistoryCache()

    cache_file = tmp_path / "cache" / "history.npz"
    assert cache_file.exists()

    history = sim.GetTimeStepHistory()
    assert [entry.time for entry in history] == [1, 2, 3]


def test_cache_merges_memory_and_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sim = _setup_cache_sim(
        tmp_path,
        cache_mode="sequence",
        cache_stride=2,
        history_enabled=True,
    )

    _drive_steps(sim, monkeypatch, n_steps=3)

    history = sim.GetTimeStepHistory()
    assert [entry.time for entry in history] == [1, 2, 3]
