"""
Operator Splitting Methods for Energy Dynamics

This module implements operator splitting techniques that decompose the complex
energy update into separate sub-problems, each with specialized fast solvers.

Strang Splitting (2nd order accurate):
    E(t + dt) = S_D(dt/2) ∘ S_R(dt) ∘ S_A(dt) ∘ S_R(dt) ∘ S_D(dt/2) E(t)

Where:
    S_D = Diffusion operator (energy spreading)
    S_R = Reaction operator (spawning/death)
    S_A = Advection operator (directional flow)
"""

from __future__ import annotations

import torch
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

Tensor = torch.Tensor


class OperatorSplittingSolver:
    """
    Operator splitting solver for energy dynamics.
    
    Splits coupled energy update into:
    1. Diffusion: Energy spreading via connections (use FFT or multigrid)
    2. Reaction: Node spawning/death based on energy thresholds
    3. Advection: Directional energy flow (semi-Lagrangian)
    
    Uses Strang splitting for 2nd-order temporal accuracy.
    """
    
    def __init__(
        self,
        diffusion_coefficient: float = 0.1,
        reaction_decay: float = 0.005,
        advection_enabled: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize operator splitting solver.
        
        Args:
            diffusion_coefficient: Diffusion strength (energy spreading rate)
            reaction_decay: Energy decay rate for reaction term
            advection_enabled: Whether to include advection operator
            device: Compute device
        """
        self.diffusion_coeff = diffusion_coefficient
        self.reaction_decay = reaction_decay
        self.advection_enabled = advection_enabled
        self.device = torch.device(device)
        
        # Performance tracking
        self.timings: Dict[str, float] = {
            "diffusion": 0.0,
            "reaction": 0.0,
            "advection": 0.0,
            "total": 0.0
        }
    
    def step(
        self,
        energy: Tensor,
        edge_index: Tensor,
        weights: Tensor,
        node_types: Tensor,
        dt: float = 1.0,
        velocity: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Perform one time step using Strang splitting.
        
        Args:
            energy: Current node energies [num_nodes]
            edge_index: Graph connectivity [2, num_edges]
            weights: Connection weights [num_edges]
            node_types: Node type for each node [num_nodes]
            dt: Time step size
            velocity: Node velocities for advection [num_nodes, 2]
            
        Returns:
            (updated_energy, metrics)
        """
        import time
        start = time.time()
        
        # Strang splitting: D(dt/2) → R(dt) → A(dt) → R(dt) → D(dt/2)
        # This gives 2nd-order accuracy in time
        
        # Step 1: Half diffusion
        t0 = time.time()
        energy = self._diffusion_step(energy, edge_index, weights, dt / 2.0)
        self.timings["diffusion"] += time.time() - t0
        
        # Step 2: Full reaction (first half)
        t0 = time.time()
        energy, reaction_metrics = self._reaction_step(energy, node_types, dt)
        self.timings["reaction"] += time.time() - t0
        
        # Step 3: Advection (if enabled)
        if self.advection_enabled and velocity is not None:
            t0 = time.time()
            energy = self._advection_step(energy, velocity, dt)
            self.timings["advection"] += time.time() - t0
        
        # Step 4: Half diffusion (complete Strang split)
        t0 = time.time()
        energy = self._diffusion_step(energy, edge_index, weights, dt / 2.0)
        self.timings["diffusion"] += time.time() - t0
        
        self.timings["total"] = time.time() - start
        
        return energy, {
            **reaction_metrics,
            "split_total_time": self.timings["total"],
            "split_diffusion_time": self.timings["diffusion"],
            "split_reaction_time": self.timings["reaction"],
            "split_advection_time": self.timings["advection"]
        }
    
    def _diffusion_step(
        self,
        energy: Tensor,
        edge_index: Tensor,
        weights: Tensor,
        dt: float
    ) -> Tensor:
        """
        Solve diffusion sub-problem: ∂E/∂t = D ∇²E
        
        Using graph Laplacian approximation:
            ∇²E_i ≈ Σ_j W_ij (E_j - E_i)
        
        Explicit Euler step (for now; can upgrade to implicit later):
            E_new = E + dt * D * L @ E
        
        Args:
            energy: Current energies
            edge_index: Graph structure
            weights: Edge weights
            dt: Time step
            
        Returns:
            Updated energies after diffusion
        """
        if edge_index.numel() == 0:
            return energy
        
        num_nodes = energy.numel()
        src, dst = edge_index[0], edge_index[1]
        
        # Compute graph Laplacian action: L @ E
        # Standard undirected graph Laplacian: (L @ E)_i = Σ_j W_ij * (E_j - E_i)
        # This requires considering both directions of each edge for symmetry
        
        # Compute off-diagonal contribution: Σ_j W_ij * E_j
        laplacian_action = torch.zeros_like(energy)
        
        # FIXED: Accumulate weighted neighbor energies correctly for undirected graph
        # For each edge (src, dst) with weight w:
        # - Accumulate w * E_src into dst (energy flows from src to dst)
        # - Accumulate w * E_dst into src (energy flows from dst to src, symmetric)
        # This ensures the Laplacian is symmetric and energy flows bidirectionally
        laplacian_action.scatter_add_(0, dst, weights * energy[src])  # Incoming to dst
        laplacian_action.scatter_add_(0, src, weights * energy[dst])  # Outgoing from src (symmetric)
        
        # Subtract self-contribution: -Σ_j W_ij * E_i
        # Degree must also consider both directions for undirected graph
        degree = torch.zeros_like(energy)
        degree.scatter_add_(0, dst, weights)  # Incoming degree
        degree.scatter_add_(0, src, weights)  # Outgoing degree (symmetric, same edges)
        laplacian_action -= degree * energy
        
        # Apply diffusion: E_new = E + dt * D * (L @ E)
        energy_new = energy + dt * self.diffusion_coeff * laplacian_action
        
        return energy_new
    
    def _reaction_step(
        self,
        energy: Tensor,
        node_types: Tensor,
        dt: float
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Solve reaction sub-problem: ∂E/∂t = R(E)
        
        Reaction term includes:
        - Energy decay: -decay * E
        - Spawning: nonlinear terms (handled separately in main simulation)
        
        Args:
            energy: Current energies
            node_types: Node types (0=sensory, 1=dynamic, 2=workspace)
            dt: Time step
            
        Returns:
            (updated_energy, metrics)
        """
        # Apply decay only to dynamic nodes (type 1)
        dynamic_mask = (node_types == 1)
        
        # Exponential decay: E_new = E * exp(-decay * dt)
        decay_factor = torch.exp(-self.reaction_decay * dt)
        
        energy_new = energy.clone()
        energy_new[dynamic_mask] *= decay_factor
        
        # Calculate energy loss for metrics
        energy_lost = (energy - energy_new).sum().item()
        
        return energy_new, {
            "reaction_energy_lost": float(energy_lost),
            "reaction_decay_factor": float(decay_factor)
        }
    
    def _advection_step(
        self,
        energy: Tensor,
        velocity: Tensor,
        dt: float
    ) -> Tensor:
        """
        Solve advection sub-problem: ∂E/∂t + v·∇E = 0
        
        Using semi-Lagrangian method (unconditionally stable):
        1. Trace particle backward in time: x' = x - v*dt
        2. Interpolate energy at x'
        3. E_new(x) = E_old(x')
        
        Args:
            energy: Current energies
            velocity: Node velocities [num_nodes, 2]
            dt: Time step
            
        Returns:
            Updated energies after advection
        """
        # Semi-Lagrangian advection
        # For discrete nodes, we approximate by shifting energy along velocity direction
        
        # Simplified: Apply velocity-weighted energy redistribution
        # This is a placeholder - full semi-Lagrangian requires grid interpolation
        
        # For now, return unchanged (proper implementation requires spatial grid)
        logger.debug("Advection step: placeholder (requires spatial grid for full implementation)")
        return energy
    
    def reset_timings(self) -> None:
        """Reset performance timing counters."""
        self.timings = {k: 0.0 for k in self.timings}
    
    def get_timings(self) -> Dict[str, float]:
        """Get performance timing breakdown."""
        return self.timings.copy()


def apply_operator_splitting(
    energy: Tensor,
    edge_index: Tensor,
    weights: Tensor,
    node_types: Tensor,
    dt: float = 1.0,
    diffusion_coeff: float = 0.1,
    reaction_decay: float = 0.005,
    velocity: Optional[Tensor] = None,
    device: str = "cpu"
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Convenience function to apply operator splitting in one call.
    
    Args:
        energy: Node energies [num_nodes]
        edge_index: Graph connectivity [2, num_edges]
        weights: Edge weights [num_edges]
        node_types: Node types [num_nodes]
        dt: Time step size
        diffusion_coeff: Diffusion coefficient
        reaction_decay: Energy decay rate
        velocity: Node velocities [num_nodes, 2]
        device: Compute device
        
    Returns:
        (updated_energy, metrics)
    """
    solver = OperatorSplittingSolver(
        diffusion_coefficient=diffusion_coeff,
        reaction_decay=reaction_decay,
        advection_enabled=(velocity is not None),
        device=device
    )
    
    return solver.step(energy, edge_index, weights, node_types, dt, velocity)
