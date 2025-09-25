"""
Setup script for building Cython extensions for neural simulation acceleration.

This script compiles the Cython modules into optimized C extensions that can be
imported and used from Python code.
"""

from setuptools import setup, Extension
try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("Cython is required to build the extensions. Please install it with 'pip install Cython'")
import numpy as np
import os

# Compiler optimizations
compile_args = [
    '-O3',  # Maximum optimization
    '-march=native',  # Optimize for current CPU architecture
    '-ffast-math',  # Fast math operations
    '-funroll-loops',  # Unroll loops for better performance
    '-fopenmp',  # Enable OpenMP for parallel processing
]

# Linker arguments
link_args = ['-fopenmp']

# Define extensions
extensions = [
    Extension(
        "cpp_extensions.synaptic_calculator",
        ["synaptic_calculator.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++",
    ),
]

# Setup configuration
setup(
    name="neural_simulation_extensions",
    version="0.1.0",
    description="Cython extensions for accelerating neural simulation",
    author="AI Neural Simulation Team",
    packages=["cpp_extensions"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'boundscheck': False,      # Disable bounds checking for performance
            'wraparound': False,       # Disable negative indexing
            'nonecheck': False,        # Assume arguments are not None
            'cdivision': True,         # Don't check for zero division
            'language_level': 3,       # Python 3 syntax
            'infer_types': True,       # Infer variable types
            'embedsignature': True,    # Include function signatures in docstrings
        },
        nthreads=os.cpu_count(),  # Use all available CPU cores for compilation
    ),
    install_requires=[
        'numpy>=1.20.0',
        'Cython>=3.0.0',
    ],
    python_requires='>=3.8',
)






