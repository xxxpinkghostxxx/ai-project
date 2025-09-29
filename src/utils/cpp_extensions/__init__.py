"""
C++ Extensions for Neural Simulation Acceleration

This package contains optimized extensions that accelerate performance-critical
components of the neural simulation system. Currently provides Numba-optimized
implementations as a stepping stone to full Cython/C++ acceleration.

Modules:
- synaptic_calculator: Optimized synaptic input calculations using Numba JIT
"""

__version__ = "0.1.0"

# Import the available implementation
try:
    # Try Cython version first
    from .synaptic_calculator import (SynapticCalculator,
                                      create_synaptic_calculator)
except ImportError:
    try:
        # Fall back to Numba-optimized Python version
        from .synaptic_calculator import (SynapticCalculator,
                                          create_synaptic_calculator)
    except ImportError as e:
        raise ImportError(
            "No synaptic_calculator implementation available. "
            "Please ensure the module is properly installed. "
            f"Error: {e}"
        ) from e
