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
- finalize_plot: Finalizes a plot with labels, limits, and other settings.

"""

from ..tools import math_tools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pickle
import numpy as np
from PIL import Image
import os

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
        x,y = math_tools.custom_meshgrid(x,y)
        im = ax.contourf(x, y, z, cmap=cmap, norm=norm)
    elif plot_type in ['imshow','smooth']:
        # transpose z
        z = z.T
        im = ax.imshow(z, cmap=cmap, norm=norm, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', interpolation='quadric')
        # adapt the aspect ratio
        ax.set_aspect('auto')
    cbar = fig.colorbar(im, ax=ax)
    finalize_plot(ax,fig,xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim,
                  cbar=cbar,clabel=clabel,clim=clim,pcm=im)
    return fig

def compile_movie(frameFileList,movieName,extension='gif',rmFrames=False,
                  pilOptimize=False, pilLoop=0, pilDuration=100):
    '''
    Compiles a movie from a list of frames.

    Parameters:
    frameFileList : list
        The list of frame files.
    movieName : str
        The name of the movie.
    extension : str, optional
        The extension of the movie file.
    rmFrames : bool, optional
        Whether to remove the frame files after compiling the movie.
    '''
    movieName += '.'+extension
    # Compiling the movie images
    images = [Image.open(frameFile) for frameFile in frameFileList]
    # Save as gif
    print("Creating movie "+movieName+"...")
    images[0].save(movieName, save_all=True, append_images=images[1:], 
                   duration=pilDuration, loop=pilLoop, optimize=pilOptimize)
    print("movie "+movieName+" created.")
    # Remove the temporary files
    if rmFrames:
        for frameFile in frameFileList:
            os.remove(frameFile)
        # Remove the temporary folder
        os.rmdir(os.path.dirname(frameFileList[0]))

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
        finalize_plot(ax,fig,xlabel=a_['xlabel'],ylabel=a_['ylabel'],legend=True)
        n_ = n_ + 1

def save_figout(figout,fname):
    figdatadict = get_figdatadict(figout[0])
    if not fname[-4:] == '.pkl':
        fname+='.pkl'
    # Save the dictionary to a JSON file
    with open(fname, 'wb') as f:
        pickle.dump(figdatadict, f)
    print(fname+' saved.')

def load_figout(fname):
    if not fname[-4:] == '.pkl':
         fname+='.pkl'
   # Load the dictionary from the pickle file
    with open(fname, 'rb') as f:
        figdatadict = pickle.load(f)
    return figdatadict

def plot_figout(fname):
    fdict = load_figout(fname)
    plot_figdatadict(fdict)
    
def add_figout_plot(fname, axis, subplotidx=0, curveidx = 0, format = '', label = ''):
    '''
    Add the data in fname to the given plot axis.
    '''
    fdict = load_figout(fname)
    l_ = fdict[subplotidx]['curves'][curveidx]
    
    label = l_['label'] if not label else label
    if format:
        axis.plot(l_['xdata'], l_['ydata'], format, label=label)
    else:
        axis.plot(l_['xdata'], l_['ydata'], label=label)
        
    axis.set_xlabel(fdict[subplotidx]['xlabel'])
    axis.set_ylabel(fdict[subplotidx]['ylabel'])
    return axis

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
        finalize_plot(ax,fig,xlabel=ax1['xlabel'],ylabel=ylabel,legend=True)

def figdatadict_get_data(filename, fieldname):
    """
    Get x and y data from a figure data dictionary.

    Parameters:
    filename : str
        The file name to extract data for.
    fieldname : str
        The field name to extract data for.
    
    Returns:
    xdata : list
        The x data.
    ydata : list
        The y data.
    xlabel : str
        The x-axis label.
    ylabel : str
        The y-axis label.
    vlabel : str
        The variable label.
    """
    figdatadict = load_figout(filename) if filename else figdatadict
    xdata = []
    ydata = []
    for ax in figdatadict:
        for l_ in ax['curves']:
            label = l_['label']
            # remove all latex characters
            label = label.replace('$','')
            label = label.replace('{','')
            label = label.replace('}','')
            label = label.replace('^','')
            label = label.replace('_','')
            label = label.replace('\\','')
            label = label.replace(',','')
            label = label.replace(' ','')
            # check if the filename is a substring of label
            if fieldname in label:
                xdata = l_['xdata']
                ydata = l_['ydata']
                xlabel = ax['xlabel']
                ylabel = ax['ylabel']
                vlabel = l_['label']
    if len(xdata) == 0:
        #print availale fields
        print('Available fields:')
        available_field = []
        for ax in figdatadict:
            for l_ in ax['curves']:
                available_field.append(l_['label'])
        #print unique fields
        print(list(set(available_field)))
        raise ValueError('Field not found in figure data dictionary')
    return xdata, ydata, xlabel, ylabel, vlabel

def finalize_plot(ax,fig, xlim=None, ylim=None, clim=None, xscale='', yscale='',
                  cbar=None, xlabel='',ylabel='',clabel='', title='', pcm = None,
                  cmap=None, legend=False, figout=[], aspect='', grid=False):
    '''
    Finalize a plot with labels, limits, and other settings.

    Parameters:
    ax : matplotlib.axes.Axes
        The axes to finalize.
    fig : matplotlib.figure.Figure
        The figure containing the axes.
    xlim : list, optional
        The x-axis limits.
    ylim : list, optional
        The y-axis limits.
    clim : list, optional
        The colorbar limits.
    xscale : str, optional
        The x-axis scale.
    yscale : str, optional
        The y-axis scale.
    cbar : matplotlib.colorbar.Colorbar, optional
        The colorbar to finalize.
    xlabel : str, optional
        The x-axis label.
    ylabel : str, optional
        The y-axis label.
    clabel : str, optional
        The colorbar label.
    title : str, optional
        The plot title.
    aspect : str, optional
        The plot aspect ratio.
    grid : bool, optional
        Whether to show the grid.
    figout : list, optional
        The list of figures to append to.
    '''
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if clim and pcm : pcm.set_clim(clim)
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if cbar and clabel: cbar.set_label(clabel)
    if cmap: pcm.set_cmap(cmap)
    if legend: 
        ncol = len(ax.get_legend_handles_labels()[1])
        k = 1 if ncol <= 4 else 2 if ncol <= 10 else 3
        ax.legend(ncol = k)
        if k > 1:
            leg = ax.get_legend()
            leg.get_frame().set_alpha(0.5)
        
    if title: ax.set_title(title)
    if aspect: ax.set_aspect(aspect)
    if grid: ax.grid(True)
    fig.tight_layout()
    figout.append(fig)

def optimize_str_format(value):
    """
    Optimize the string format of a value for better readability.
    
    Parameters:
    value : float
        The value to format.
    
    Returns:
    str
        The formatted string.
    """
    if isinstance(value, float):
        if value == 0:
            return '0.0'
        elif abs(value) < 1e-3 or abs(value) > 1e3:
            return f'{value:.2e}'
        else:
            return f'{value:.2f}'
    else:
        return str(value)