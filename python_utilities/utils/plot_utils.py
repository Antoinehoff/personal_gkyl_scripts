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
            'slicecoords':frame.slicecoords, 'slicetitle':frame.slicetitle}

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
        ax.set_xlabel(frame.new_gsymbols[0]); ax.set_ylabel(frame.new_gsymbols[1])
        ax.set_title((frame.slicetitle+", t=%2.2e (ms)")%(frame.time*1000))
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
    
def plot_1D_time_evolution(simulation,fieldname,cdirection,ccoords,
                           twindow=[],space_time=False, cmap='inferno',
                           xlim=[], ylim=[], clim=[]):
    fig,ax = plt.subplots()

    data = get_1xt_diagram(simulation, fieldname, cdirection, ccoords,tfs=twindow)

    t   = data['t'] #get in ms
    x   = data['x']
    tlabel = "Time (ms)"
    xlabel = data['xsymbol']+(' ('+data['xunits']+')')*(1-(data['xunits']==''))
    vlabel = data['vsymbol']+(' ('+data['vunits']+')')*(1-(data['vunits']==''))
    if space_time:
        if data['name'] == 'phi':
                cmap = 'bwr'
        XX, TT = np.meshgrid(x,t)
        vmax = np.max(np.abs(data['values'])); vmin = -vmax * (cmap=='bwr')

        # Create a contour plot or a heatmap of the space-time diagram
        pcm = ax.pcolormesh(XX,TT*1000,data['values'],cmap=cmap,vmin=vmin,vmax=vmax); 
        ax.set_xlabel(xlabel); ax.set_ylabel(tlabel);
        ax.set_title(data['slicetitle']);
        cbar = fig.colorbar(pcm,label=vlabel);
        #-- to change window
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if clim:
            pcm.set_clim(clim)
    else:
        norm = plt.Normalize(min(t)*1000, max(t)*1000)
        colormap = cm.viridis  # You can choose any colormap, e.g., 'plasma', 'inferno', etc.
        for it in range(len(t)):
            ax.plot(x,data['values'][it][:],label=r'$t=%2.2e$ (ms)'%(1000*t[it]),
                    color=colormap(norm(t[it]*1000)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(vlabel)
        ax.set_title(data['slicetitle'])
        # Add a colorbar to the figure
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm);sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax);cbar.set_label(tlabel)  # Label for the colorbar
    fig.tight_layout()

def plot_2D_cut(simulation, fieldname, cdirection, ccoord, tf,
                xlim=[], ylim=[], clim=[]):
    frame = get_2D_slice(simulation, fieldname, cdirection, ccoord, tf)
    YY,XX = np.meshgrid(frame.new_grids[1],frame.new_grids[0])
    fig,ax = plt.subplots()
    pcm = ax.pcolormesh(XX,YY,frame.values,cmap='inferno')
    ax.set_xlabel(frame.gsymbols[0]); ax.set_ylabel(frame.gsymbols[1])
    ax.set_title((frame.slicetitle+", t=%2.2e (ms)")%(frame.time*1000))
    cbar = fig.colorbar(pcm,label=label(frame.vsymbol,frame.vunits))
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_xlim(ylim)
    if clim:
        pcm.set_clim(clim)    
    fig.tight_layout()

def label(label,units):
    if units:
        label += ' ('+units+')'
    return label
