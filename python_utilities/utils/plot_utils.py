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
        frame = Frame(simulation,fieldname,tf)
        frame.load()
        frame.slice_1D(cutdirection,ccoords)
        t.append(frame.time)
        vv.append(frame.values)
    frame.free_values() # remove values to free memory
    x = frame.new_grids[0]
    return {'x':x,'t':t,'values':vv,'name':frame.name,
            'xsymbol':frame.new_gsymbols[0], 'xunits':frame.new_gunits[0], 
            'vsymbol':frame.vsymbol, 'vunits':frame.vunits, 'slicetitle':frame.slicetitle,
            'slicecoords':frame.slicecoords, 'fulltitle':frame.fulltitle}
    
def plot_1D_time_evolution(simulation,cdirection,ccoords,fieldname='', spec='e',
                           twindow=[],space_time=False, cmap='inferno',
                           xlim=[], ylim=[], clim=[], time_avg=False, full_plot=False):
    full_plot = (full_plot or (fieldname=='')) and (not fieldname=='phi')
    multi_species = isinstance(spec, list)
    cmap0 = cmap
    if not multi_species:
        spec = [spec]
    else:
        time_avg = True
        space_time = False

    if full_plot:
        fig,axs = plt.subplots(2,2,figsize=(8,6))
        axs    = axs.flatten()
        fields = ['n','upar','Tpar','Tperp']
    else:
        fig,axs = plt.subplots(1,1)
        axs    = [axs]
        fields = [fieldname]

    for s_ in spec:
        for ax,field in zip(axs,fields):
            if not field == 'phi':
                field += s_
            data = get_1xt_diagram(simulation, field, cdirection, ccoords,tfs=twindow)

            t   = data['t'] #get in ms
            x   = data['x']
            tsymb = simulation.normalization['tsymbol'] 
            tunit = simulation.normalization['tunits']
            tlabel = tsymb+(' ('+tunit+')')*(1-(tunit==''))
            xlabel = data['xsymbol']+(' ('+data['xunits']+')')*(1-(data['xunits']==''))
            vlabel = data['vsymbol']+(' ('+data['vunits']+')')*(1-(data['vunits']==''))
            if not time_avg:
                if space_time:
                    if data['name'] == 'phi' or data['name'][:-1] == 'upar':
                        cmap = 'bwr'
                        vmax = np.max(np.abs(data['values'])) 
                        vmin = -vmax
                    else:
                        cmap = cmap0
                        vmax = np.max(np.abs(data['values'])) 
                        vmin = 0.0

                    XX, TT = np.meshgrid(x,t)
                    # Create a contour plot or a heatmap of the space-time diagram
                    pcm = ax.pcolormesh(XX,TT,data['values'],cmap=cmap,vmin=vmin,vmax=vmax); 
                    ax.set_xlabel(xlabel); ax.set_ylabel(tlabel);
                    title = data['slicetitle'][:-2]
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
                    colormap = cm.viridis  # You can choose any colormap
                    for it in range(len(t)):
                        ax.plot(x,data['values'][it][:],label=r'$t=%2.2e$ (ms)'%(t[it]),
                                color=colormap(norm(t[it])))
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(vlabel)
                    title = data['fulltitle']
                    # Add a colorbar to the figure
                    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm);sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax);cbar.set_label(tlabel)
            else:
                # Compute the average of data over the t-axis (axis=1)
                average_data = np.mean(data['values'], axis=0)
                # Compute the standard deviation of data over the t-axis (axis=1)
                std_dev_data = np.std(data['values'], axis=0)
                # Plot with error bars
                ax.errorbar(x, average_data, yerr=std_dev_data, 
                            fmt='o', capsize=5, label=vlabel)
                # Labels and title
                ax.set_xlabel(xlabel)
                if multi_species:
                    ax.set_ylabel(data['vunits'])
                    ax.legend()
                else:
                    ax.set_ylabel(vlabel)
                title = data['fulltitle']+tlabel+r'$\in[%2.2e,%2.2e]$'%(t[0],t[-1])
            if not full_plot:
                ax.set_title(title)
    if full_plot:
        fig.suptitle(title)
    fig.tight_layout()

def plot_2D_cut(simulation,cdirection,ccoord,tf,
                fieldname='',spec='e', cmap='inferno', full_plot=False,
                xlim=[], ylim=[], clim=[], 
                figout=[],cutout=[],):
    full_plot = (full_plot or (fieldname=='')) and (not fieldname=='phi')
    cmap0 = cmap
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
        frame = Frame(simulation,field,tf)
        frame.load()
        frame.slice_2D(cdirection,ccoord)

        if (field == 'phi' or field[:-1] == 'upar') or cmap0 == 'bwr':
            cmap = 'bwr'
            vmax = np.max(np.abs(frame.values)) 
            vmin = -vmax
        else:
            cmap = cmap0
            vmax = np.max(frame.values)
            vmin = np.min(frame.values)

        YY,XX = np.meshgrid(frame.new_grids[1],frame.new_grids[0])
        pcm = ax.pcolormesh(XX,YY,frame.values,cmap=cmap,vmin=vmin,vmax=vmax)
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
    # This allows to return the figure in the arguments line (used for movies)
    figout.append(fig)
    cutout.append(frame.slicecoords)

def make_2D_movie(simulation,cdirection,ccoord,tfs,
                      fieldname='',spec='e', cmap='inferno',
                      xlim=[], ylim=[], clim=[], full_plot=False):
    os.makedirs('gif_tmp', exist_ok=True)
    for tf in tfs:
        figout = []; cutout = []
        plot_2D_cut(simulation,cdirection,ccoord,tf=tf,fieldname=fieldname,
                    spec=spec,cmap=cmap,full_plot=full_plot,
                    xlim=xlim,ylim=ylim,clim=clim,
                    cutout=cutout,figout=figout)
        fig = figout[0]
        fig.tight_layout()
        fig.savefig(f'gif_tmp/plot_{tf}.png')
        plt.close()
    # Naming
    if not fieldname or full_plot:
        fieldname = 'mom'+spec
    cutout=cutout[0]
    cutname = [key+('=%2.2f'%cutout[key]) for key in cutout]
    moviename = 'movie_'+fieldname+'_'+cutname[0]
    if xlim:
        moviename+='_xlim_%2.2d_%2.2d'%(xlim[0],xlim[1])
    if ylim:
        moviename+='_ylim_%2.2d_%2.2d'%(ylim[0],ylim[1])
    if clim:
        moviename+='_clim_%2.2d_%2.2d'%(clim[0],clim[1])
    moviename += '.gif'
    # Compiling the movie images
    images = [Image.open(f'gif_tmp/plot_{tf}.png') for tf in tfs]
    # Save as gif
    images[0].save(moviename, save_all=True, append_images=images[1:], duration=200, loop=1)
    print("movie "+moviename+" created.")

def plot_domain(geometry,geom_type='Miller',vessel_corners=[[0.6,1.2],[-0.7,0.7]]):
    
    geometry.set_domain(geom_type,vessel_corners)

    fig = plt.figure()#(figsize=(4, 3))
    ax  = fig.add_subplot(111)
    ax.plot(geometry.RZ_min[0], geometry.RZ_min[1],'-c')
    ax.plot(geometry.RZ_max[0], geometry.RZ_max[1],'-c')
    ax.plot(geometry.RZ_lcfs[0],geometry.RZ_lcfs[1],'--k')
    vx1 = geometry.vessel_corners[0][0]            
    vx2 = geometry.vessel_corners[0][1]            
    vy1 = geometry.vessel_corners[1][0]            
    vy2 = geometry.vessel_corners[1][1]            
    ax.plot([vx1,vx1],[vy1,vy2],'-k')
    ax.plot([vx2,vx2],[vy1,vy2],'-k')
    ax.plot([vx1,vx2],[vy1,vy1],'-k')
    ax.plot([vx1,vx2],[vy2,vy2],'-k')
    ax.plot(geometry.R_axis,geometry.Z_axis,'x')

    ax.set_xlabel('R (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')

def plot_GBsource(simulation,species,tf=0,ix=0,b=1.2):
    # Set up the simulation geometry and load useful data
    simulation.geom_param.compute_bxgradBoB2()
    vGBz_x = simulation.geom_param.bxgradBoB2[0,ix,:,:]
    qs     = species.q
    ygrid  = simulation.geom_param.y
    Ly     = simulation.geom_param.Ly
    zgrid  = simulation.geom_param.z
    # build n*T product
    nT = 1.0
    for field in ['n','Tperp']:
        field += species.name[0]
        frame  = Frame(simulation,field,tf,load=True)
        nT    *= frame.values[ix,:,:]
    # eV to Joules conversion
    nT *= simulation.phys_param.eV
    # assemble n*T/q*vgradB and integrate over y
    Gammaz = np.trapz(nT*vGBz_x/qs,x=ygrid, axis=0)

    # Compare with the GB source model
    vGBz_x = simulation.geom_param.GBflux_model(b=b)
    n0      = species.n0
    T0      = species.T0
    # y integration is done by Ly multiplication
    fz      = -n0*T0/qs * vGBz_x * Ly

    plt.plot(zgrid,Gammaz,label='Effective source at ' + frame.timetitle)
    plt.plot(zgrid,fz,label='GB source model')
    plt.legend()
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\int \Gamma_{\nabla B,x} dy$')
        
    Ssimul = np.trapz(Gammaz,x=zgrid, axis=0)
    Smodel = np.trapz(fz    ,x=zgrid, axis=0)

    plt.title('Total source simul: %2.2e 1/s, total source model: %2.2e 1/s'\
              %(Ssimul,Smodel))

def label(label,units):
    if units:
        label += ' ('+units+')'
    return label
