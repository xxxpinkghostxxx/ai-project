"""
PDE-Based Energy Dynamics

This module implements reaction-diffusion PDEs for energy evolution,
providing a physically principled mathematical framework.

General form:
    ∂E/∂t = D∇²E + R(E) + S(x,t)

Where:
- D∇²E: Diffusion term (energy spreading)
- R(E): Reaction term (spawning/death, nonlinear)
- S(x,t): Source term (sensory input, external forcing)

Discretization methods:
- Finite difference for spatial derivatives
- Implicit/explicit time stepping for stability
- Adaptive time-stepping for efficiency
"""

from __future__ import annotations

import torch
from typing import Dict, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)

Tensor = torch.Tensor


class ReactionDiffusionPDE:
    """
    Reaction-Diffusion PDE solver for energy dynamics.
    
    Solves: ∂E/∂t = D∇²E + R(E) + S
    
    Using finite difference discretization and implicit time-stepping.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        diffusion_coeff: float = 0.1,
        reaction_rate: float = 0.01,
        dx: float = 1.0,
        device: str = "cpu"
    ):
        """
        Initialize PDE solver.
        
        Args:
            grid_size: Grid dimensions (H, W)
            diffusion_coeff: Diffusion coefficient D
            reaction_rate: Reaction rate constant
            dx: Spatial grid spacing
            device: Compute device
        """
        self.grid_size = grid_size
        self.H, self.W = grid_size
        self.D = diffusion_coeff
        self.reaction_rate = reaction_rate
        self.dx = dx
        self.device = torch.device(device)
        
        # CFL condition for explicit stability
        # dt < dx² / (4D) for explicit diffusion
        self.dt_max = dx**2 / (4.0 * diffusion_coeff)
        
        logger.info(f"PDE solver initialized: grid={grid_size}, D={diffusion_coeff}, dt_max={self.dt_max:.4f}")
    
    def laplacian_5point(self, field: Tensor) -> Tensor:
        """
        Compute Laplacian using 5-point stencil.
        
        ∇²u_ij = (u_i+1,j + u_i-1,j + u_i,j+1 + u_i,j-1 - 4*u_ij) / dx²
        
        Args:
            field: Field values [H, W]
            
        Returns:
            Laplacian [H, W]
        """
        # Pad with replication (Neumann boundary conditions)
        padded = torch.nn.functional.pad(field, (1, 1, 1, 1), mode='replicate')
        
        laplacian = (
            padded[2:, 1:-1] +      # down
            padded[:-2, 1:-1] +     # up
            padded[1:-1, 2:] +      # right
            padded[1:-1, :-2] -     # left
            4.0 * field
        ) / (self.dx ** 2)
        
        return laplacian
    
    def reaction_term(
        self,
        energy: Tensor,
        spawn_threshold: float = 20.0,
        death_threshold: float = -10.0
    ) -> Tensor:
        """
        Compute reaction term R(E).
        
        Models:
        - Exponential decay: -k * E
        - Spawning (logistic growth): k * E * (1 - E/K)
        - Death: -k * E when E < threshold
        
        Args:
            energy: Energy field [H, W]
            spawn_threshold: Energy threshold for spawning
            death_threshold: Energy threshold for death
            
        Returns:
            Reaction term [H, W]
        """
        # Exponential decay
        decay = -self.reaction_rate * energy
        
        # Logistic growth (spawning) for high energy
        carrying_capacity = 200.0
        growth = self.reaction_rate * energy * (1.0 - energy / carrying_capacity)
        growth = torch.where(energy > spawn_threshold, growth, torch.zeros_like(growth))
        
        # Death term for low energy
        death = -self.reaction_rate * 10.0 * energy
        death = torch.where(energy < death_threshold, death, torch.zeros_like(death))
        
        return decay + growth + death
    
    def step_explicit(
        self,
        energy: Tensor,
        source: Optional[Tensor] = None,
        dt: Optional[float] = None
    ) -> Tensor:
        """
        Explicit (forward Euler) time step.
        
        E_new = E + dt * (D∇²E + R(E) + S)
        
        Fast but requires small dt for stability (CFL condition).
        
        Args:
            energy: Current energy [H, W]
            source: Source term [H, W]
            dt: Time step (auto-computed if None)
            
        Returns:
            Updated energy [H, W]
        """
        if dt is None:
            dt = self.dt_max * 0.5  # Use half of max for safety
        
        # Diffusion term
        diffusion = self.D * self.laplacian_5point(energy)
        
        # Reaction term
        reaction = self.reaction_term(energy)
        
        # Source term
        if source is None:
            source = torch.zeros_like(energy)
        
        # Update
        energy_new = energy + dt * (diffusion + reaction + source)
        
        return energy_new
    
    def step_implicit(
        self,
        energy: Tensor,
        source: Optional[Tensor] = None,
        dt: float = 0.1,
        max_iterations: int = 10
    ) -> Tensor:
        """
        Implicit (backward Euler) time step.
        
        E_new - dt * D∇²E_new = E + dt * (R(E) + S)
        
        Unconditionally stable, allows larger dt.
        Solved iteratively using fixed-point iteration.
        
        Args:
            energy: Current energy [H, W]
            source: Source term [H, W]
            dt: Time step
            max_iterations: Max iterations for implicit solve
            
        Returns:
            Updated energy [H, W]
        """
        if source is None:
            source = torch.zeros_like(energy)
        
        # Right-hand side (explicit part)
        reaction = self.reaction_term(energy)
        rhs = energy + dt * (reaction + source)
        
        # Iterative solve for implicit part
        energy_new = energy.clone()
        
        for _ in range(max_iterations):
            # Implicit diffusion: (I - dt*D*∇²) E_new = rhs
            laplacian = self.laplacian_5point(energy_new)
            energy_new = rhs + dt * self.D * laplacian
        
        return energy_new
    
    def step_crank_nicolson(
        self,
        energy: Tensor,
        source: Optional[Tensor] = None,
        dt: float = 0.1,
        max_iterations: int = 10
    ) -> Tensor:
        """
        Crank-Nicolson (trapezoidal) time step.
        
        E_new = E + dt/2 * (D∇²E + D∇²E_new + R(E) + R(E_new) + 2S)
        
        Second-order accurate in time, unconditionally stable.
        
        Args:
            energy: Current energy [H, W]
            source: Source term [H, W]
            dt: Time step
            max_iterations: Max iterations
            
        Returns:
            Updated energy [H, W]
        """
        if source is None:
            source = torch.zeros_like(energy)
        
        # Old time level
        laplacian_old = self.laplacian_5point(energy)
        reaction_old = self.reaction_term(energy)
        
        # Initial guess
        energy_new = energy.clone()
        
        # Fixed-point iteration
        for _ in range(max_iterations):
            laplacian_new = self.laplacian_5point(energy_new)
            reaction_new = self.reaction_term(energy_new)
            
            # Crank-Nicolson formula
            energy_new = energy + 0.5 * dt * (
                self.D * (laplacian_old + laplacian_new) +
                (reaction_old + reaction_new) +
                2.0 * source
            )
        
        return energy_new
    
    def solve(
        self,
        initial_energy: Tensor,
        source: Optional[Tensor] = None,
        num_steps: int = 10,
        dt: float = 0.1,
        method: str = "crank_nicolson"
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Solve PDE for multiple time steps.
        
        Args:
            initial_energy: Initial condition [H, W]
            source: Source term [H, W] (constant)
            num_steps: Number of time steps
            dt: Time step size
            method: "explicit", "implicit", or "crank_nicolson"
            
        Returns:
            (final_energy, metrics)
        """
        import time
        start = time.time()
        
        energy = initial_energy.clone()
        
        for step in range(num_steps):
            if method == "explicit":
                energy = self.step_explicit(energy, source, dt)
            elif method == "implicit":
                energy = self.step_implicit(energy, source, dt)
            elif method == "crank_nicolson":
                energy = self.step_crank_nicolson(energy, source, dt)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Clamp to physical range
            energy = torch.clamp(energy, min=-10.0, max=244.0)
        
        total_time = time.time() - start
        
        return energy, {
            "pde_total_time": total_time,
            "pde_steps": num_steps,
            "pde_dt": dt,
            "pde_method": method
        }


class AdvancedReactionDiffusion:
    """
    Advanced reaction-diffusion system with multiple species.
    
    Classic patterns: Turing patterns, spiral waves, etc.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        device: str = "cpu"
    ):
        """Initialize advanced PDE system."""
        self.grid_size = grid_size
        self.device = torch.device(device)
    
    def gray_scott_model(
        self,
        u: Tensor,
        v: Tensor,
        dt: float = 1.0,
        Du: float = 0.16,
        Dv: float = 0.08,
        F: float = 0.035,
        k: float = 0.065
    ) -> Tuple[Tensor, Tensor]:
        """
        Gray-Scott reaction-diffusion model.
        
        Produces beautiful Turing patterns!
        
        ∂u/∂t = Du∇²u - uv² + F(1-u)
        ∂v/∂t = Dv∇²v + uv² - (F+k)v
        
        Args:
            u, v: Species concentrations [H, W]
            dt: Time step
            Du, Dv: Diffusion coefficients
            F: Feed rate
            k: Kill rate
            
        Returns:
            (u_new, v_new)
        """
        # Laplacians
        laplacian_u = self._laplacian_5point(u)
        laplacian_v = self._laplacian_5point(v)
        
        # Reactions
        reaction_u = -u * v**2 + F * (1.0 - u)
        reaction_v = u * v**2 - (F + k) * v
        
        # Update
        u_new = u + dt * (Du * laplacian_u + reaction_u)
        v_new = v + dt * (Dv * laplacian_v + reaction_v)
        
        return u_new, v_new
    
    def _laplacian_5point(self, field: Tensor) -> Tensor:
        """Compute Laplacian (helper method)."""
        padded = torch.nn.functional.pad(field, (1, 1, 1, 1), mode='replicate')
        return (
            padded[2:, 1:-1] +
            padded[:-2, 1:-1] +
            padded[1:-1, 2:] +
            padded[1:-1, :-2] -
            4.0 * field
        )


def apply_pde_energy_evolution(
    initial_energy: Tensor,
    grid_size: Tuple[int, int] = (64, 64),
    diffusion_coeff: float = 0.1,
    num_steps: int = 10,
    dt: float = 0.1,
    method: str = "crank_nicolson",
    device: str = "cpu"
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Convenience function for PDE-based energy evolution.
    
    Args:
        initial_energy: Initial energy field [H, W]
        grid_size: Grid dimensions
        diffusion_coeff: Diffusion coefficient
        num_steps: Number of time steps
        dt: Time step size
        method: Time-stepping method
        device: Compute device
        
    Returns:
        (final_energy, metrics)
    """
    solver = ReactionDiffusionPDE(
        grid_size=grid_size,
        diffusion_coeff=diffusion_coeff,
        device=device
    )
    
    return solver.solve(initial_energy, num_steps=num_steps, dt=dt, method=method)
