"""Main simulation orchestrator."""

from typing import Optional, Callable, Dict
import numpy as np
from .domain import Domain
from .initial_conditions import InitialConditions
from ..models.base_model import PFCModel
from ..dynamics.base_dynamics import Dynamics
from ..backends.base_backend import Backend
from ..operators.base_ops import Operators


class Simulation:
    """
    Main simulation orchestrator.
    
    Orchestrates time loop. Creates/manages Domain, InitialConditions, Backend,
    Operators, Model, Dynamics, and outputs. No numerical computations.
    """
    
    def __init__(
        self,
        domain: Domain,
        model: PFCModel,
        dynamics: Dynamics,
        backend: Backend,
        operators: Operators,
        initial_conditions: InitialConditions
    ):
        """Wire up all components."""
        self.domain = domain
        self.model = model
        self.dynamics = dynamics
        self.backend = backend
        self.operators = operators
        self.initial_conditions = initial_conditions
        
        # Initialize fields
        self.fields = backend.initialize_fields(domain, initial_conditions)
        
        # Set field shape on model
        model.set_field_shape(domain.shape)
        
        # State tracking
        self.current_step = 0
        self.current_time = 0.0
    
    def run(
        self,
        num_steps: int,
        dt: float,
        checkpoint_interval: int = 100,
        output_fn: Optional[Callable] = None,
        progress_fn: Optional[Callable] = None
    ) -> None:
        """
        Run simulation for num_steps timesteps.
        
        Args:
            num_steps: number of timesteps
            dt: timestep size (may be adaptive)
            checkpoint_interval: save checkpoint every N steps (0 = no checkpoints)
            output_fn: optional callable(step, fields, time) for custom output
            progress_fn: optional callable(step, time, energy, ...) for diagnostics
        
        The loop is:
            for step in range(num_steps):
              fields = backend.timestep(fields, dt, ...)
              compute diagnostics (energy, etc.)
              if step % checkpoint_interval == 0: save checkpoint
              if output_fn: output_fn(step, fields, step*dt)
              if progress_fn: progress_fn(step, step*dt, energy, ...)
        """
        for step in range(num_steps):
            # Timestep
            self.fields = self.backend.timestep(
                self.fields,
                dt,
                self.model,
                self.dynamics,
                self.operators
            )
            
            self.current_step += 1
            self.current_time += dt
            
            # Diagnostics
            if progress_fn is not None:
                fields_np = self.backend.to_numpy(self.fields)
                energy = self.model.free_energy(fields_np)
                progress_fn(self.current_step, self.current_time, energy)
            
            # Checkpoint
            if checkpoint_interval > 0 and self.current_step % checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.current_step}.npz")
            
            # Custom output
            if output_fn is not None:
                fields_np = self.backend.to_numpy(self.fields)
                output_fn(self.current_step, fields_np, self.current_time)
    
    def load_checkpoint(self, filename: str) -> None:
        """Restore fields and metadata from checkpoint."""
        data = np.load(filename, allow_pickle=True)
        fields_np = {key: data[key] for key in data.files if key.startswith('field_')}
        self.fields = self.backend.from_numpy(fields_np)
        self.current_step = int(data.get('step', 0))
        self.current_time = float(data.get('time', 0.0))
    
    def save_checkpoint(self, filename: str) -> None:
        """Save current fields and metadata to checkpoint."""
        fields_np = self.backend.to_numpy(self.fields)
        save_dict = {f'field_{name}': arr for name, arr in fields_np.items()}
        save_dict['step'] = self.current_step
        save_dict['time'] = self.current_time
        np.savez_compressed(filename, **save_dict)
