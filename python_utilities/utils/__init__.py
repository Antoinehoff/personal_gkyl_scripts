# src/utils/__init__.py

# Import specific utility functions or classes
from .file_utils import find_prefix, find_available_frames, check_latex_installed
from .math_utils import func_time_ave, func_calc_norm_fluc, integral_xyz,\
    custom_meshgrid, integral_yz
from .plot_utils import func_data_omp,get_1xt_diagram,make_2D_movie,\
    plot_1D_time_evolution,label, plot_2D_cut, plot_domain, plot_GBsource,\
    plot_1D_time_avg, plot_volume_integral_vs_t, plot_1D, plot_GB_loss
# You can also define __all__ to control what gets imported with a wildcard import (*)
__all__ = [
    'find_prefix', 
    'func_data_omp',
    'func_time_ave',
    'func_calc_norm_fluc',
    'find_available_frames',
    'get_1xt_diagram',
    'make_2D_movie',
    'plot_1D_time_evolution',
    'label',
    'plot_2D_cut',
    'plot_domain',
    'plot_GBsource',
    'integral_xyz',
    'custom_meshgrid',
    'integral_yz',
    'plot_1D_time_avg',
    'plot_volume_integral_vs_t',
    'plot_1D',
    'check_latex_installed',
    'plot_GB_loss'
]

# You can also include some utility constants or helper functions here if needed.