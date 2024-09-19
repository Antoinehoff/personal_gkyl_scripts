import postgkyl as pg
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .math_utils import *
from .file_utils import *
from classes import Frame
import os

# Function reads gkyl data of 2D axisymmetric fields and produces 1D array
# at outer midplane (omp)
def func_data_omp(field2d, comp):
    field2dInterp = pg.data.GInterpModal(field2d, 1, 'ms')
    interpGrid, field2dValues = field2dInterp.interpolate(comp)
    # get cell center coordinates since interpolate returns edge values
    CCC = []
    for j in range(0,len(interpGrid)):
        CCC.append((interpGrid[j][1:] + interpGrid[j][:-1])/2)
    x_vals = CCC[0]
    z_vals = CCC[1]
    z_slice = len(z_vals)//2
    field1dValues = field2dValues[:,z_slice,0]
    return x_vals,field1dValues

def get_1xt_slice(simulation, fieldname, cutdirection, ccoords, tf):
    frame = Frame(simulation,fieldname,tf)
    frame.load()
    frame.slice_1D(cutdirection,ccoords)
    return frame

def get_1xt_diagram(simulation, fieldname, cutdirection, ccoords,tfs=[]):
    # Get available time frames
    dataname = simulation.data_param.data_files_dict[fieldname+'file']
    if not tfs:
        tfs = find_available_frames(simulation, dataname)
    if not isinstance(tfs, list):
        tfs = [tfs]
    # to store iteratively times and values
    t  = []
    vv = []
    # Fill ZZ with data for each time frame
    for it, tf in enumerate(tfs):
        frame = get_1xt_slice(simulation,fieldname,cutdirection,ccoords,tf)
        t.append(frame.time)
        vv.append(frame.values)
    frame.free_values() # remove values to free memory
    x = frame.new_grids[0]
    return {'x':x,'t':t,'values':vv,'name':frame.name,
            'xsymbol':frame.new_gsymbols[0], 'xunits':frame.new_gunits[0], 
            'vsymbol':frame.vsymbol, 'vunits':frame.vunits,
            'slicecoords':frame.slicecoords, 'fulltitle':frame.fulltitle}

def get_2D_slice(simulation, fieldname, cutdirection, ccoord, tf):
    frame = Frame(simulation,fieldname,tf)
    frame.load()
    frame.slice_2D(cutdirection,ccoord)
    return frame

def make_2D_movie(simulation, fieldname, cdirection, ccoord, tfs,
                  xlim=[], ylim=[], clim=[], fixed_cbar=False):
    # Get a first frame to compute meshgrids
    os.makedirs('gif_tmp', exist_ok=True)
    frame = get_2D_slice(simulation, fieldname, cdirection, ccoord, tfs[0])
    YY,XX = np.meshgrid(frame.new_grids[1],frame.new_grids[0])
    for tf in tfs:
        frame = get_2D_slice(simulation, fieldname, cdirection, ccoord, tf)
        fig,ax = plt.subplots()
        pcm = ax.pcolormesh(XX,YY,frame.values,cmap='inferno')
        xlabel = label(frame.new_gsymbols[0],frame.new_gunits[0])
        ylabel = label(frame.new_gsymbols[1],frame.new_gunits[1])
        title  = frame.fulltitle
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        cbar = fig.colorbar(pcm,label=label(frame.vsymbol,frame.vunits))
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_xlim(ylim)
        if clim:
            pcm.set_clim(clim)    
        fig.tight_layout()
        fig.savefig(f'gif_tmp/plot_{tf}.png')
        plt.close()
    moviename = 'movie_'+fieldname+'_'+cdirection[0]+cdirection[1]+'='+('%2.2f'%ccoord)
    if xlim:
        moviename+='_xlim_%2.2d_%2.2d'%(xlim[0],xlim[1])
    if ylim:
        moviename+='_ylim_%2.2d_%2.2d'%(ylim[0],ylim[1])
    if clim:
        moviename+='_clim_%2.2d_%2.2d'%(clim[0],clim[1])
    moviename += '.gif'
    # Load images
    images = [Image.open(f'gif_tmp/plot_{tf}.png') for tf in tfs]
    # Save as gif
    images[0].save(moviename, save_all=True, append_images=images[1:], duration=200, loop=1)
    print("movie "+moviename+" created.")
    
def plot_1D_time_evolution(simulation,cdirection,ccoords,fieldname='', spec='e',
                           twindow=[],space_time=False, cmap='inferno',
                           xlim=[], ylim=[], clim=[], time_avg=False, full_plot=False):
    full_plot = (full_plot or (fieldname=='')) and (not fieldname=='phi')
    if full_plot:
        fig,axs = plt.subplots(2,2,figsize=(8,6))
        axs    = axs.flatten()
        fields = ['n','upar','Tpar','Tperp']
        fields = [f_+spec for f_ in fields]
    else:
        fig,axs = plt.subplots(1,1)
        axs    = [axs]
        fields = [fieldname]

    for ax,field in zip(axs,fields):

        data = get_1xt_diagram(simulation, field, cdirection, ccoords,tfs=twindow)

        t   = data['t'] #get in ms
        x   = data['x']
        tsymb = simulation.normalization['tsymbol']; tunit = simulation.normalization['tunits']
        tlabel = tsymb+(' ('+tunit+')')*(1-(tunit==''))
        xlabel = data['xsymbol']+(' ('+data['xunits']+')')*(1-(data['xunits']==''))
        vlabel = data['vsymbol']+(' ('+data['vunits']+')')*(1-(data['vunits']==''))
        if not time_avg:
            if space_time:
                if data['name'] == 'phi':
                        cmap = 'bwr'
                XX, TT = np.meshgrid(x,t)
                vmax = np.max(np.abs(data['values'])); vmin = -vmax * (cmap=='bwr')

                # Create a contour plot or a heatmap of the space-time diagram
                pcm = ax.pcolormesh(XX,TT,data['values'],cmap=cmap,vmin=vmin,vmax=vmax); 
                ax.set_xlabel(xlabel); ax.set_ylabel(tlabel);
                title = data['fulltitle']
                cbar = fig.colorbar(pcm,label=vlabel);
                #-- to change window
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)
                if clim:
                    pcm.set_clim(clim)
            else:
                norm = plt.Normalize(min(t), max(t))
                colormap = cm.viridis  # You can choose any colormap, e.g., 'plasma', 'inferno', etc.
                for it in range(len(t)):
                    ax.plot(x,data['values'][it][:],label=r'$t=%2.2e$ (ms)'%(t[it]),
                            color=colormap(norm(t[it])))
                ax.set_xlabel(xlabel)
                ax.set_ylabel(vlabel)
                title = data['fulltitle']
                # Add a colorbar to the figure
                sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm);sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax);cbar.set_label(tlabel)  # Label for the colorbar
        else:
            # Compute the average of data over the t-axis (axis=1)
            average_data = np.mean(data['values'], axis=0)
            # Compute the standard deviation of data over the t-axis (axis=1)
            std_dev_data = np.std(data['values'], axis=0)
            # Plot with error bars
            ax.errorbar(x, average_data, yerr=std_dev_data, fmt='o', capsize=5, label=vlabel)
            # Labels and title
            ax.set_xlabel(xlabel)
            ax.set_ylabel(vlabel)
            title = data['fulltitle']+tlabel+r'$\in[%2.2e,%2.2e]$'%(t[0],t[-1])
        if not full_plot:
            ax.set_title(title)
    if full_plot:
        fig.suptitle(title)
    fig.tight_layout()

def plot_2D_cut(simulation,cdirection,ccoord,tf,
                fieldname='',spec='e', cmap='inferno',
                xlim=[], ylim=[], clim=[], full_plot=False):
    full_plot = (full_plot or (fieldname=='')) and (not fieldname=='phi')
    if full_plot:
        fig,axs = plt.subplots(2,2,figsize=(8,6))
        axs    = axs.flatten()
        fields = ['n','upar','Tpar','Tperp']
        fields = [f_+spec for f_ in fields]
    else:
        fig,axs = plt.subplots(1,1)
        axs    = [axs]
        fields = [fieldname]

    for ax,field in zip(axs,fields):

        frame = get_2D_slice(simulation, field, cdirection, ccoord, tf)
        YY,XX = np.meshgrid(frame.new_grids[1],frame.new_grids[0])
        pcm = ax.pcolormesh(XX,YY,frame.values,cmap=cmap)
        xlabel = label(frame.new_gsymbols[0],frame.new_gunits[0])
        ylabel = label(frame.new_gsymbols[1],frame.new_gunits[1])
        title  = frame.fulltitle
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = fig.colorbar(pcm,label=label(frame.vsymbol,frame.vunits))
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_xlim(ylim)
        if clim:
            pcm.set_clim(clim)    
        if not full_plot:
            ax.set_title(title)
    if full_plot:
        fig.suptitle(title)
    
    fig.tight_layout()

def compare_GBsource(simulation,species,tf,ix=0):
    simulation.geom_param.compute_bxgradBoB2()
    y      = simulation.geom_param.grids[1]
    Ly     = y[-1] - y[0]
    z      = simulation.geom_param.grids[2]

    vGBz_x = np.trapz(simulation.geom_param.bxgradBoB2[0,ix,:,:], x=y, axis=0)
    tf     = 200
    # build n*T product
    nT_z = 1.0
    for field in ['n','Tpar']:
        field += species.name[0]
        frame  = Frame(simulation,field,tf,load=True)
        nT_z    *= np.trapz(frame.values[ix,:,:],x=y, axis=0)
    # eV to Joules conversion
    nT_z *= simulation.phys_param.eV
    qs     = species.q
    Gammaz = nT_z/qs * vGBz_x
    plt.plot(z,Gammaz,label='Effective source at ' + frame.timetitle)

    # the GB source model
    vGBz_x = simulation.geom_param.GBflux_model()
    n0      = species.n0
    T0      = species.T0
    fz      = n0*T0/qs * vGBz_x * Ly
    plt.plot(z,-fz,label='GB source model')
    plt.legend()
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\Gamma_{\nabla B,x}$')
        
def label(label,units):
    if units:
        label += ' ('+units+')'
    return label
