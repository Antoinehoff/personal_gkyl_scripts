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
default_fig_dpi = 100

def label_from_simnorm(simulation,name):
    return label(simulation.normalization.dict[name+'symbol'],simulation.normalization.dict[name+'units'])

def label(label,units):
    if units:
        label += ' [%s]'%units
    return label

def multiply_by_m3_expression(expression):
    
    if expression[-6:]=='/m$^3$':
        expression_new = expression[:-6]
    elif expression[-6:]=='m$^{-3}$':
        expression_new = expression[:-8]
    else:
        expression_new = expression + r'm$^3$'
    return expression_new

def setup_figure(fieldnames,figsize=None,fig_dpi=None):
    if figsize is None:
        figsize = default_figsz
    if fig_dpi is None:
        fig_dpi = default_fig_dpi
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
    fig,axs = plt.subplots(nrow,ncol,figsize=(figsize[0]*ncol,figsize[1]*nrow), dpi=fig_dpi)
    if ncol == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    return fields,fig,axs

def plot_2D(fig,ax,x,y,z, xlim=None, ylim=None, clim=None, vmin=None,vmax=None,
            xlabel='', ylabel='', clabel='', title='', cmap_period=1,
            cmap='viridis', colorscale='linear', plot_type='pcolormesh', aspect='auto'):
    z = np.squeeze(z)
    
    if colorscale == 'log':
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        # set negative values to nan
        z[z <= 0] = np.nan
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
    # Handle periodic colormap
    if cmap_period > 1:
        # Create a periodic colormap by repeating the original colormap
        original_cmap = cm.get_cmap(cmap)
        colors = original_cmap(np.linspace(0, 1, 512))
        # Create repeated colors with reversed alternate periods for continuity
        repeated_colors = []
        for i in range(cmap_period):
            if i % 2 == 1:
                repeated_colors.append(colors[::-1])
            else:
                repeated_colors.append(colors)  # Reverse for continuity
        repeated_colors = np.vstack(repeated_colors)
        # Create new colormap from repeated colors
        periodic_cmap = mcolors.ListedColormap(repeated_colors)
        im.set_cmap(periodic_cmap)
        
    # adapt the aspect ratio
    ax.set_aspect(aspect)
    
    cbar = fig.colorbar(im, ax=ax)
    finalize_plot(ax,fig,xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim,
                  cbar=cbar,clabel=clabel,clim=clim,pcm=im)
    return fig

def compile_movie(frameFileList,movieName,extension='gif',rmFrames=False,
                  pilOptimize=False, pilLoop=0, pilDuration=100):
    '''
    Compiles a movie from a list of frames.

    Parameters
    ----------
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

    Parameters
    ----------
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
    '''
    Save figure data to a pickle file. 
    '''
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

def compare_figouts(files, names=None, colors=None, linestyles=None, plot_idx=None,
                    xlim=None, ylim=None, figsize=None, fig_dpi=None, figout=None,
                    close_fig=False):
    """
    Overlay curves from multiple saved figout files onto a shared set of subplots.
    Each file is assigned one color (auto-cycled when not specified), so all curves
    from the same simulation share a color and are distinguishable from other simulations.

    When ``names`` are provided, curve symbols are moved to the y-axis label so that
    the legend only shows the simulation name, keeping it compact.
    When ``names`` are omitted, labels fall back to the original ``symbol + filename``
    format so the legend remains self-contained.

    By default, linestyles are cycled by curve index within each subplot so that
    the same field (e.g. ``ne``) always gets the same linestyle across all files.
    Pass ``linestyles`` to override with one linestyle per file instead.

    Parameters
    ----------
    files : str or list of str
        One or more pickle file paths produced by save_figout.
    names : str or list of str, optional
        One label per file shown in the legend.  When provided, the y-axis
        displays the field symbol instead, keeping the legend uncluttered.
        When omitted, names default to the base filename and are appended to
        the curve symbol in the legend.
    colors : str or list of str, optional
        One color per file. Defaults to matplotlib's prop_cycle.
    linestyles : str or list of str, optional
        One linestyle per file (e.g. '-', '--', ':').  When omitted, linestyles
        are cycled by curve index within each subplot so the same field always
        gets the same linestyle across all files.
    plot_idx : int or list of int, optional
        Subplot index or indices to include. Defaults to all subplots.
    xlim : list, optional
        x-axis limits applied to all subplots.
    ylim : list, optional
        y-axis limits applied to all subplots.
    figsize : list, optional
        Figure size [width, height] per subplot cell.
    fig_dpi : int, optional
        Figure DPI.
    figout : list, optional
        List to append the resulting figure to.
    close_fig : bool, optional
        Close the figure after appending to figout. Useful in scripts to avoid
        double display in Jupyter notebooks.

    Returns
    ----------
    fig : matplotlib.figure.Figure
    """
    _default_linestyles = ['-', '--', ':', '-.']

    if figout is None:
        figout = []
    if isinstance(files, str):
        files = [files]
    files = [f[:-4] if f.endswith('.pkl') else f for f in files]
    nfiles = len(files)

    # Track whether the caller supplied names explicitly
    user_provided_names = names is not None

    # Auto-derive names from basenames when not supplied
    if names is None:
        names = [os.path.basename(f) for f in files]
    elif isinstance(names, str):
        names = [names]
    names = list(names) + [os.path.basename(files[i]) for i in range(len(names), nfiles)]

    # Auto-assign one color per file from prop_cycle
    prop_cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = [prop_cycle_colors[i % len(prop_cycle_colors)] for i in range(nfiles)]
    elif isinstance(colors, str):
        colors = [colors]
    colors = list(colors) + [prop_cycle_colors[i % len(prop_cycle_colors)]
                              for i in range(len(colors), nfiles)]

    # Per-file linestyle override: None means auto-cycle by curve index instead
    per_file_ls = None
    if linestyles is not None:
        if isinstance(linestyles, str):
            linestyles = [linestyles]
        per_file_ls = list(linestyles) + ['-'] * (nfiles - len(linestyles))

    fdicts = [[a for a in load_figout(f) if a['curves']] for f in files]

    # Select subplot indices from the first file's axes
    if plot_idx is not None:
        indices = [plot_idx] if isinstance(plot_idx, int) else list(plot_idx)
    else:
        indices = list(range(len(fdicts[0])))

    naxes = len(indices)
    fsz = figsize if figsize else default_figsz
    dpi = fig_dpi if fig_dpi else default_fig_dpi
    ncol = 1 if naxes == 1 else 2
    nrow = naxes // ncol + naxes % ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(fsz[0]*ncol, fsz[1]*nrow), dpi=dpi)
    if naxes == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for ax, idx in zip(axs, indices):
        ref_ax = fdicts[0][idx]
        for fi, (fdict, name, color) in enumerate(zip(fdicts, names, colors)):
            if idx >= len(fdict):
                continue
            curves = fdict[idx]['curves']
            for ci, l_ in enumerate(curves):
                # Linestyle: per-file override or auto-cycle by curve index
                ls = per_file_ls[fi] if per_file_ls is not None else _default_linestyles[ci % len(_default_linestyles)]
                # When names are user-supplied: legend shows only the name and
                # the symbol goes on the y-axis; otherwise keep symbol+name together.
                lbl = name if user_provided_names else (l_['label'] + ' ' + name)
                ax.plot(l_['xdata'], l_['ydata'], color=color, linestyle=ls, label=lbl)

        # Y-axis: show the field symbol when names are user-supplied, full ylabel otherwise
        if user_provided_names and ref_ax['curves']:
            ylabel = ref_ax['curves'][0]['label']
        else:
            ylabel = ref_ax['ylabel']
        ax.set_xlabel(ref_ax['xlabel'])
        ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        # Deduplicate legend entries (multiple curves share the same name per file)
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        k = 1 if len(seen) <= 4 else 2 if len(seen) <= 10 else 3
        ax.legend(seen.values(), seen.keys(), ncol=k)

    for ax in axs[naxes:]:
        ax.set_visible(False)

    fig.tight_layout()
    figout.append(fig)
    if close_fig:
        plt.close(fig)
    return fig

def figdatadict_get_data(filename, fieldname):
    """
    Get x and y data from a figure data dictionary.

    Parameters
    ----------
    filename : str
        The file name to extract data for.
    fieldname : str
        The field name to extract data for.

    Returns
    ----------
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
    def delatexify(label):
        label = label.replace('$','')
        label = label.replace('{','')
        label = label.replace('}','')
        label = label.replace('^','')
        label = label.replace('_','')
        label = label.replace('\\','')
        label = label.replace(',','')
        label = label.replace(' ','')
        return label
    
    for ax in figdatadict:
        for l_ in ax['curves']:
            label = l_['label']
            # remove all latex characters
            label = delatexify(label)
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
                available_field.append(delatexify(l_['label']))
        #print unique fields
        print(list(set(available_field)))
        raise ValueError('Field not found in figure data dictionary')
    return xdata, ydata, xlabel, ylabel, vlabel

def finalize_plot(ax,fig, xlim=None, ylim=None, clim=None, xscale='', yscale='',
                  cbar=None, xlabel='',ylabel='',clabel='', title='', pcm = None,
                  cmap=None, legend=False, figout=[], aspect='', grid=False):
    '''
    Finalize a plot with labels, limits, and other settings.

    Parameters
    ----------
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

    Parameters
    ----------
    value : float
        The value to format.

    Returns
    ----------
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
    
def check_colormap(colormap):
    """
    Check if the provided colormap is valid.
    Returns the colormap if valid, otherwise returns False.
    """
    if isinstance(colormap, str):
        try:
            return cm.get_cmap(colormap)
        except ValueError:
            return False
    elif isinstance(colormap, mcolors.Colormap):
        return colormap
    else:
        return False