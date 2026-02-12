"""Domain configuration for spatial grid and geometry."""

from typing import Tuple, Optional
import numpy as np


class Domain:
    """
    Spatial domain configuration.
    
    Encapsulates spatial grid, geometry, and boundary conditions (BCs).
    Does NOT hold field data; just metadata.
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        box_size: Tuple[float, ...],
        dtype=np.float64,
        bc: str = 'periodic'
    ):
        """
        Args:
            shape: tuple (nx,) or (nx, ny) or (nx, ny, nz) for 1D/2D/3D
            box_size: tuple (Lx,) or (Lx, Ly) or (Lx, Ly, Lz) physical size
            dtype: numpy dtype for field arrays
            bc: 'periodic' or 'finite_difference' or custom variant
        """
        self._shape = shape
        self._box_size = box_size
        self._dtype = dtype
        self._bc = bc
        
        # Validate
        if len(shape) != len(box_size):
            raise ValueError("shape and box_size must have same dimension")
        
        # Pre-compute spacing
        self._spacing = tuple(L / n for L, n in zip(box_size, shape))
        
        # Cache for wavenumbers and coordinates
        self._wavenumbers: Optional[Tuple] = None
        self._coordinates: Optional[Tuple] = None
    
    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (1, 2, or 3)."""
        return len(self._shape)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Grid shape (nx,) or (nx, ny) or (nx, ny, nz)."""
        return self._shape
    
    @property
    def box_size(self) -> Tuple[float, ...]:
        """Physical domain size (Lx,) or (Lx, Ly) or (Lx, Ly, Lz)."""
        return self._box_size
    
    @property
    def spacing(self) -> Tuple[float, ...]:
        """Grid spacing (dx,) or (dx, dy) or (dx, dy, dz)."""
        return self._spacing
    
    @property
    def dtype(self):
        """Data type for fields (numpy dtype)."""
        return self._dtype
    
    @property
    def bc(self) -> str:
        """Boundary condition type."""
        return self._bc
    
    def get_wavenumbers(self) -> Tuple[np.ndarray, ...]:
        """
        Return k-space wavenumbers (for spectral methods).
        For 1D: (kx,); for 2D: (kx, ky); for 3D: (kx, ky, kz).
        """
        if self._wavenumbers is None:
            k_arrays = []
            for i, (n, L) in enumerate(zip(self._shape, self._box_size)):
                k = 2.0 * np.pi * np.fft.fftfreq(n, L / n)
                k_arrays.append(k)
            self._wavenumbers = tuple(k_arrays)
        return self._wavenumbers
    
    def get_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        Return real-space coordinates.
        For 1D: (x,); for 2D: (x, y); for 3D: (x, y, z).
        """
        if self._coordinates is None:
            coord_arrays = []
            for i, (n, L) in enumerate(zip(self._shape, self._box_size)):
                x = np.linspace(0, L, n, endpoint=False, dtype=self._dtype)
                coord_arrays.append(x)
            self._coordinates = tuple(coord_arrays)
        return self._coordinates
