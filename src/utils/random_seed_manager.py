
import random
import numpy as np
import torch
import os
from typing import Optional


class RandomSeedManager:
    def __init__(self, base_seed: Optional[int] = None):

        self.base_seed = base_seed or int(os.environ.get('NEURAL_SIMULATION_SEED', 42))
        self.current_seed = self.base_seed
        self._initialized = False
    def initialize(self, seed: Optional[int] = None):

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
        return self.current_seed
    def set_seed(self, seed: int):
        self.initialize(seed)
    def increment_seed(self, increment: int = 1):
        self.set_seed(self.current_seed + increment)
    def reset_to_base(self):
        self.initialize(self.base_seed)
    def is_initialized(self) -> bool:
        return self._initialized
    def get_random_state(self) -> dict:
        return {
            'python_random': random.getstate(),
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
            'cuda_random': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'current_seed': self.current_seed
        }
    def set_random_state(self, state: dict):
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
_seed_manager = None


def get_seed_manager() -> RandomSeedManager:
    global _seed_manager
    if _seed_manager is None:
        _seed_manager = RandomSeedManager()
    return _seed_manager


def initialize_random_seeds(seed: Optional[int] = None):
    manager = get_seed_manager()
    manager.initialize(seed)


def set_random_seed(seed: int):
    manager = get_seed_manager()
    manager.set_seed(seed)


def get_current_seed() -> int:
    manager = get_seed_manager()
    return manager.get_seed()


def increment_seed(increment: int = 1):
    manager = get_seed_manager()
    manager.increment_seed(increment)


def reset_to_base_seed():
    manager = get_seed_manager()
    manager.reset_to_base()


def ensure_seeds_initialized():
    manager = get_seed_manager()
    if not manager.is_initialized():
        manager.initialize()


def random_choice(choices, size=None, replace=True):
    ensure_seeds_initialized()
    return np.random.choice(choices, size=size, replace=replace)


def random_uniform(low=0.0, high=1.0, size=None):
    ensure_seeds_initialized()
    return np.random.uniform(low, high, size)


def random_normal(mean=0.0, std=1.0, size=None):
    ensure_seeds_initialized()
    return np.random.normal(mean, std, size)


def random_randn(*args):
    ensure_seeds_initialized()
    return np.random.randn(*args)


def random_rand(*args):
    ensure_seeds_initialized()
    return np.random.rand(*args)


def random_int(low, high=None, size=None):
    ensure_seeds_initialized()
    return np.random.randint(low, high, size)


def random_float(low=0.0, high=1.0):
    ensure_seeds_initialized()
    return random.uniform(low, high)


def random_bool(probability=0.5):
    ensure_seeds_initialized()
    return random.random() < probability
initialize_random_seeds()






