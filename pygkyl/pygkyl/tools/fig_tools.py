"""
fig_tools.py -- Various utilities for handling and plotting figures.

Functions:
- label_from_simnorm: Generates a label from simulation normalization.
- label: Generates a label with units.
- multiply_by_m3_expression: Modifies an expression to include or exclude m^3.
- setup_figure: Sets up a figure with subplots based on field names.
- get_figdatadict: Extracts data from all curves in a figure.
- plot_figdatadict: Plots data from a figure data dictionary.
- save_figout: Saves figure data to a pickle file.
- load_figout: Loads figure data from a pickle file.
- plot_figout: Plots figure data from a pickle file.
- compare_figouts: Compares and plots data from two figure data dictionaries.

"""

from ..tools import math_tools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pickle
import numpy as np

default_figsz = [5,3.5]

def label_from_simnorm(simulation,name):
    return label(simulation.normalization.dict[name+'symbol'],simulation.normalization.dict[name+'units'])

def label(label,units):
    if units:
        label += ' ('+units+')'
    return label

def multiply_by_m3_expression(expression):
    
    if expression[-6:]=='/m$^3$':
        expression_new = expression[:-6]
    elif expression[-6:]=='m$^{-3}$':
        expression_new = expression[:-8]
    else:
        expression_new = expression + r'm$^3$'
    return expression_new

def setup_figure(fieldnames):
    if fieldnames == '':
        ncol = 2
        fields = ['ne','upari','Tpari','Tperpi']
    elif not isinstance(fieldnames,list):
        ncol   = 1
        fields = [fieldnames]
    else:
        ncol = 1 * (len(fieldnames) == 1) + 2 * (len(fieldnames) > 1)
        fields = fieldnames
    nrow = len(fields)//ncol + len(fields)%ncol
    fig,axs = plt.subplots(nrow,ncol,figsize=(default_figsz[0]*ncol,default_figsz[1]*nrow))
    if ncol == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    return fields,fig,axs

def plot_2D(fig,ax,x,y,z, xlim=None, ylim=None, clim=None, vmin=None,vmax=None,
            xlabel='', ylabel='', clabel='', title='',
            cmap='viridis', colorscale='linear', plot_type='pcolormesh'):
    z = np.squeeze(z)
    if colorscale == 'log':
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    if plot_type == 'pcolormesh':
        x,y = math_tools.custom_meshgrid(x,y)
        im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    elif plot_type == 'contourf':
        im = ax.contourf(x, y, z, cmap=cmap, norm=norm)
    elif plot_type == 'smoothed':
        # smooth the data
        from scipy.ndimage import gaussian_filter
        z = gaussian_filter(z, sigma=0.5)
        im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if clim: im.set_clim(clim)

    return fig

def get_figdatadict(fig):
    """
    Get data from all curves in a figure
    
    Parameters:
    fig : matplotlib.figure.Figure
        The figure containing the curves.
    """
    figdatadict = []
    # Loop through each Axes in the Figure
    for ax in fig.get_axes():
        a_ = {}
        # axis labels
        a_['xlabel'] = ax.get_xlabel()
        a_['ylabel'] = ax.get_ylabel()
        a_['curves'] = []
        # Write data for each line in the Axes
        for line in ax.get_lines():
            l_ = {}
            l_['xdata'] = line.get_xdata()  # Extract x data
            l_['ydata'] = line.get_ydata()  # Extract y data
            l_['label'] = line.get_label()  # Extract label
            a_['curves'].append(l_)
        figdatadict.append(a_)

    return figdatadict

def plot_figdatadict(figdatadict):
    naxes = len(figdatadict)
    if naxes == 1:
        ncol   = 1
    else:
        ncol = 1 * (naxes == 1) + 2 * (naxes > 1)
    nrow = naxes//ncol + naxes%ncol
    fig,axs = plt.subplots(nrow,ncol,figsize=(default_figsz[0]*ncol,default_figsz[1]*nrow))
    if ncol == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
        
    n_ = 0
    for ax in axs:
        a_ = figdatadict[n_]
        for l_ in a_['curves']:
            ax.plot(l_['xdata'], l_['ydata'], label=l_['label'])
        ax.set_xlabel(a_['xlabel'])
        ax.set_ylabel(a_['ylabel'])
        ax.legend()
        n_ = n_ + 1

def save_figout(figout,fname):
    figdatadict = get_figdatadict(figout[0])
    if not fname[-4:] == '.pkl':
        fname+'.pkl'
    # Save the dictionary to a JSON file
    with open(fname, 'wb') as f:
        pickle.dump(figdatadict, f)
    print(fname+' saved.')

def load_figout(fname):
    if not fname[-4:] == '.pkl':
         fname+'.pkl'
   # Load the dictionary from the pickle file
    with open(fname, 'rb') as f:
        figdatadict = pickle.load(f)
    return figdatadict

def plot_figout(fname):
    fdict = load_figout(fname)
    plot_figdatadict(fdict)

def compare_figouts(file1,file2,name1='',name2='',clr1='',clr2='',plot_idx=0,lnums='all'):
    if file1[-4:] == '.pkl':
        file1 = file1[:-4]
    if file2[-4:] == '.pkl':
        file2 = file2[:-4]
    fdict1 = load_figout(file1)    
    fdict2 = load_figout(file2)
    
    ax1 = fdict1[plot_idx]    
    ax2 = fdict2[plot_idx]
    
    lnums,fig,axs = setup_figure(lnums)
    for ax,lnum in zip(axs,lnums):
        for lnums_sub in lnum:
            if not isinstance(lnums_sub,list):
                lnums_sub = [lnums_sub]
            il = 0
            for l_ in ax1['curves']:
                if lnums_sub[0]=='all' or il in lnums_sub:
                    ax.plot(l_['xdata'], l_['ydata'], label=name1)
                il += 1
            il = 0
            for l_ in ax2['curves']:    
                if lnums_sub[0]=='all' or il in lnums_sub:
                    ax.plot(l_['xdata'], l_['ydata'], label=name2)
                    ylabel = l_['label']
                il += 1
        ax.set_ylabel(ylabel)
        ax.set_xlabel(ax1['xlabel'])
        ax.legend()
    plt.tight_layout()
    plt.show()