"""
Analysis and optimization components.

This package contains analysis and optimization components including:
- Duplicate code detection and optimization
- NASA compliance analysis
- Code quality verification
"""

from .duplicate_code_detector import *
from .focused_optimizer import *
from .nasa_code_analyzer import *
from .verify_nasa_compliance import *

__all__ = [
    'DuplicateCodeDetector',
    'FocusedOptimizer',
    'NASACodeAnalyzer',
    'VerifyNASACompliance'
]
