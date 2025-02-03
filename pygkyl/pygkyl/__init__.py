"""
pygkyl

A Python package for handling and analyzing Gkeyll simulation data. This package provides classes and tools for 
setting up simulations, processing data, and visualizing results.

Modules:
- classes: Contains core classes such as Simulation, Species, Frame, etc.
- tools: Provides various tools for mathematical operations, interfacing with pgkyl, and physical computations.
- utils: Includes utility functions for file handling, plotting, and mathematical operations.
"""

from .classes import Species, Simulation, NumParam, PhysParam, Frame, GeomParam, GBsource, Source
from .tools import fig_tools, math_tools, pgkyl_interface, phys_tools, DG_tools
from .utils import file_utils, math_utils, plot_utils

__all__ = [
    'Species',
    'Simulation',
    'NumParam',
    'PhysParam',
    'Frame',
    'GeomParam',
    'GBsource',
    'Sources',
    'tools',
    'utils',
    'math_tools',
    'pgkyl_interface',
    'phys_tools',
    'fig_tools',
    'file_utils',
    'math_utils',
    'plot_utils',
    'DG_tools'
]