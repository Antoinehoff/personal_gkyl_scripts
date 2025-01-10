"""
pygkyl

A Python package for handling and analyzing Gkeyll simulation data. This package provides classes and tools for 
setting up simulations, processing data, and visualizing results.

Modules:
- classes: Contains core classes such as Simulation, Species, Frame, etc.
- tools: Provides various tools for mathematical operations, interfacing with pgkyl, and physical computations.
- utils: Includes utility functions for file handling, plotting, and mathematical operations.
"""

from .classes import Species, Simulation, NumParam, PhysParam, Frame, GeomParam, GBsource, OMPsources, Source
from .tools import math_tools, pgkyl_interface, phys_tools
from .utils import fig_utils, file_utils, math_utils, plot_utils

__all__ = [
    'Species',
    'Simulation',
    'NumParam',
    'PhysParam',
    'Frame',
    'GeomParam',
    'GBsource',
    'Sources',
    'OMPsources',
    'tools',
    'utils',
    'math_tools',
    'pgkyl_interface',
    'phys_tools',
    'fig_utils',
    'file_utils',
    'math_utils',
    'plot_utils'
]