"""
Random Seed Manager for the Neural Simulation System.
Provides reproducible random seed management for Python, NumPy, and PyTorch.
"""


import os
import random
from typing import Optional

import numpy as np
import torch


class RandomSeedManager:
    """Manages random seeds for reproducible simulations."""

    def __init__(self, base_seed: Optional[int] = None):
        """Initialize the seed manager with a base seed."""
        self.base_seed = base_seed or int(os.environ.get('NEURAL_SIMULATION_SEED', 42))
        self.current_seed = self.base_seed
        self._initialized = False

    def initialize(self, seed: Optional[int] = None):
        """Initialize random seeds for Python, NumPy, and PyTorch."""

        if seed is not None:
            self.current_seed = seed
        else:
            self.current_seed = self.base_seed
        random.seed(self.current_seed)
        np.random.seed(self.current_seed)
        torch.manual_seed(self.current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.current_seed)
            torch.cuda.manual_seed_all(self.current_seed)
        self._initialized = True
    def get_seed(self) -> int:
        """Get the current seed value."""
        return self.current_seed
    def set_seed(self, seed: int):
        """Set the seed to a specific value."""
        self.initialize(seed)
    def increment_seed(self, increment: int = 1):
        """Increment the current seed by a given amount."""
        self.set_seed(self.current_seed + increment)
    def reset_to_base(self):
        """Reset the seed to the base seed."""
        self.initialize(self.base_seed)
    def is_initialized(self) -> bool:
        """Check if the seed manager is initialized."""
        return self._initialized
    def get_random_state(self) -> dict:
        """Get the current random state of all libraries."""
        return {
            'python_random': random.getstate(),
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
            'cuda_random': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'current_seed': self.current_seed
        }
    def set_random_state(self, state: dict):
        """Set the random state from a saved state dictionary."""
        if 'python_random' in state:
            random.setstate(state['python_random'])
        if 'numpy_random' in state:
            np.random.set_state(state['numpy_random'])
        if 'torch_random' in state:
            torch.set_rng_state(state['torch_random'])
        if 'cuda_random' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state['cuda_random'])
        if 'current_seed' in state:
            self.current_seed = state['current_seed']
class RandomSeedManagerSingleton:
    """Singleton wrapper for RandomSeedManager."""

    _instance: RandomSeedManager = None

    @classmethod
    def get_instance(cls) -> RandomSeedManager:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = RandomSeedManager()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance."""
        cls._instance = None


def get_seed_manager() -> RandomSeedManager:
    """Get the seed manager instance."""
    return RandomSeedManagerSingleton.get_instance()


def initialize_random_seeds(seed: Optional[int] = None):
    """Initialize random seeds using the manager."""
    manager = get_seed_manager()
    manager.initialize(seed)


def set_random_seed(seed: int):
    """Set the random seed globally."""
    manager = get_seed_manager()
    manager.set_seed(seed)


def get_current_seed() -> int:
    """Get the current seed value globally."""
    manager = get_seed_manager()
    return manager.get_seed()


def increment_seed(increment: int = 1):
    """Increment the seed globally."""
    manager = get_seed_manager()
    manager.increment_seed(increment)


def reset_to_base_seed():
    """Reset to base seed globally."""
    manager = get_seed_manager()
    manager.reset_to_base()


def ensure_seeds_initialized():
    """Ensure seeds are initialized if not already."""
    manager = get_seed_manager()
    if not manager.is_initialized():
        manager.initialize()


def random_choice(choices, size=None, replace=True):
    """Random choice from array with replacement."""
    ensure_seeds_initialized()
    return np.random.choice(choices, size=size, replace=replace)


def random_uniform(low=0.0, high=1.0, size=None):
    """Uniform random floats."""
    ensure_seeds_initialized()
    return np.random.uniform(low, high, size)


def random_normal(mean=0.0, std=1.0, size=None):
    """Normal random floats."""
    ensure_seeds_initialized()
    return np.random.normal(mean, std, size)


def random_randn(*args):
    """Standard normal random floats."""
    ensure_seeds_initialized()
    return np.random.randn(*args)


def random_rand(*args):
    """Uniform random floats in [0,1)."""
    ensure_seeds_initialized()
    return np.random.rand(*args)


def random_int(low, high=None, size=None):
    """Random integers."""
    ensure_seeds_initialized()
    return np.random.randint(low, high, size)


def random_float(low=0.0, high=1.0):
    """Random float in range."""
    ensure_seeds_initialized()
    return random.uniform(low, high)


def random_bool(probability=0.5):
    """Random boolean with probability."""
    ensure_seeds_initialized()
    return random.random() < probability
initialize_random_seeds()






