"""
plot_utils.py

This module provides various plotting utilities for visualizing Gkeyll simulation data.

Functions:
- get_1xt_diagram: Retrieves 1D time evolution data for a given field and cut direction.
- plot_1D_time_evolution: Plots the 1D time evolution of a field.
- plot_1D: Plots 1D data for a given field and time frames.
- plot_2D_cut: Plots a 2D cut of the simulation domain for a given field and time frame.
- make_2D_movie: Creates a 2D movie from a series of 2D cuts.
- plot_domain: Plots the simulation domain and vessel boundaries.
- plot_GBsource: Plots the GB source model and compares it with the effective source.
- plot_volume_integral_vs_t: Plots the volume integral of a field over time.
- plot_GB_loss: Plots the GB loss over time for different species.
- plot_integrated_moment: Plots integrated moments over time for different species.

"""

# pgkyl to load and interpolate gkyl data
import postgkyl as pg
# personnal classes and routines
from .math_utils import *
from .file_utils import *
from .fig_utils import *
from .data_utils import *
from classes import Frame
# other commonly used libs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tools import pgkyl_interface as pgkyl_

# set the font to be LaTeX
if check_latex_installed(verbose=False):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams.update({'font.size': 14})

import os, re

def get_1xt_diagram(simulation, fieldname, cutdirection, ccoords,
                    tfs):
    # Check if we need to fourier transform
    index = cutdirection.find('k')
    if index > -1:
        fourrier_y = True
        cutdirection = cutdirection.replace('k','')
    else:
        fourrier_y = False

    # to store iteratively times and values
    t  = []
    values = []
    if not isinstance(tfs,list): tfs = [tfs]
    # Fill ZZ with data for each time frame
    for it, tf in enumerate(tfs):
        frame = Frame(simulation,fieldname,tf)
        frame.load()
        if fourrier_y:
            frame.fourrier_y()
        frame.slice_1D(cutdirection,ccoords)
        # if fourrier_y:
            # frame.values = frame.values/np.max(frame.values)
        t.append(frame.time)
        values.append(frame.values)
    values = np.squeeze(values)
    if values.ndim == 1: values = values.reshape(1,-1)
    frame.free_values() # remove values to free memory
    x = frame.new_grids[0]
    tsymb = simulation.normalization['tsymbol'] 
    tunit = simulation.normalization['tunits']
    tlabel = tsymb+(' ('+tunit+')')*(1-(tunit==''))
    xlabel = frame.new_gsymbols[0]+(' ('+frame.new_gunits[0]+')')*(1-(frame.new_gunits[0]==''))
    vlabel = frame.vsymbol+(' ('+frame.vunits+')')*(1-(frame.vunits==''))
    slicetitle = frame.slicetitle
    return x,t,values,xlabel,tlabel,vlabel,frame.vunits,slicetitle, fourrier_y
    
def plot_1D_time_evolution(simulation,cdirection,ccoords,fieldnames='',
                           twindow=[],space_time=False, cmap='inferno',
                           fluctuation=False, bckgrnd_avg_wind=[],
                           xlim=[], ylim=[], clim=[], figout=[]):
    
    if not isinstance(twindow,list): twindow = [twindow]
    cmap0 = cmap
    fields,fig,axs = setup_figure(fieldnames)
    for ax,field in zip(axs,fields):
        x,t,values,xlabel,tlabel,vlabel,vunits,slicetitle,fourrier_y =\
              get_1xt_diagram(simulation,field,cdirection,ccoords,tfs=twindow)
        if fluctuation:
            average_data = np.mean(values, axis=0)
            vlabel = r'$\delta$' + vlabel
            for it in range(len(t)) :
                values[it,:] = (values[it,:] - average_data[:])
            if fluctuation == "relative":
                values = 100.0*values/average_data
                vlabel = re.sub(r'\(.*?\)', '', vlabel)
                vlabel = vlabel + ' (\%)'
        if space_time:
            if ((field in ['phi','upare','upari']) or cmap0=='bwr' or fluctuation) and not fourrier_y:
                cmap = 'bwr'
                vmax = np.max(np.abs(values)) 
                vmin = -vmax
            else:
                cmap = cmap0
                vmax = np.max(np.abs(values)) 
                vmin = 0.0

            if fourrier_y:
                vmin  = np.power(10,np.log10(vmax)-4)
                values = np.clip(values, vmin, None)  # We plot 4 orders of magnitude
                create_norm = mcolors.LogNorm
            else:
                create_norm = mcolors.Normalize
            
            XX, TT = np.meshgrid(x,t)
            # Create a contour plot or a heatmap of the space-time diagram
            norm  = create_norm(vmin=vmin, vmax=vmax)
            pcm = ax.pcolormesh(XX,TT,values,cmap=cmap,norm=norm); 
            cbar = fig.colorbar(pcm)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(tlabel)
            cbar.set_label(vlabel)
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
                ax.plot(x,values[it][:],label=r'$t=%2.2e$ (ms)'%(t[it]),
                        color=colormap(norm(t[it])))
            # Add a colorbar to the figure
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm);sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(vlabel)
            cbar.set_label(tlabel)
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
        if len(axs) > 1:
            fig.suptitle(slicetitle[:-2])
        else:
            ax.set_title(slicetitle[:-2])
    fig.tight_layout()
    figout.append(fig)

def plot_1D(simulation,cdirection,ccoords,fieldnames='',
            time_frames=[], xlim=[], ylim=[], xscale='', yscale = '', periodicity = 0, grid = False,
            figout = [], errorbar = False):
    
    fields,fig,axs = setup_figure(fieldnames)

    if isinstance(time_frames,int) or len(time_frames) == 1:
        time_avg = False
    else:
        time_avg = True

    for ax,field in zip(axs,fields):
        if not isinstance(field,list):
            subfields = [field] #simple plot
        else:
            subfields = field # field is a combined plot
        for subfield in subfields:
            x,t,values,xlabel,tlabel,vlabel,vunits,slicetitle,fourrier_y =\
                get_1xt_diagram(simulation,subfield,cdirection,ccoords,tfs=time_frames)
            # Compute the average of data over the t-axis (axis=1)
            average_data = np.mean(values, axis=0)
            # Compute the standard deviation of data over the t-axis (axis=1)
            std_dev_data = np.std(values, axis=0)
            if time_avg and errorbar:
                # Plot with error bars
                ax.errorbar(x, average_data, yerr=std_dev_data, 
                            fmt='o', capsize=5, label=vlabel)
            else:
                # Classic plot
                ax.plot(x, average_data, label=vlabel)
                if periodicity > 0:
                    ax.plot(x+periodicity, average_data, label=vlabel)

        # Labels and title
        ax.set_xlabel(xlabel)
        if len(subfields)>1:
            ax.set_ylabel(vunits)
            ax.legend()
        else:
            ax.set_ylabel(vlabel)
        #-- to change window
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)
        if grid:
            ax.grid(True)

    if t[0] == t[-1]:
        title = slicetitle+tlabel+r'$=%2.2e$'%(t[0])
    else:
        title = slicetitle+tlabel+r'$\in[%2.2e,%2.2e]$'%(t[0],t[-1])
    if len(axs) > 1:
        fig.suptitle(title)
    else:
        ax.set_title(title)
    fig.tight_layout()
    figout.append(fig)

def plot_2D_cut(simulation,cut_dir,cut_coord,time_frame,
                fieldnames='', cmap='inferno', full_plot=False,
                time_average=False, fluctuation=False,
                xlim=[], ylim=[], clim=[], 
                figout=[],cutout=[], val_out=[], frames_to_plot = None):
    # Check if we provide multiple time frames (time average or fluctuation plot)
    if isinstance(time_frame, int):
        time_frame = [time_frame]
    if clim:
        clim = [clim] if not isinstance(clim[0], list) else clim

    # Check if we need to fourier transform
    index = cut_dir.find('k')
    if index > -1:
        fourrier_y = True
        cut_dir = cut_dir.replace('k','')
    else:
        fourrier_y = False

    cmap0 = cmap    
    fields,fig,axs = setup_figure(fieldnames)
    kf = 0 # field counter
    for ax,field in zip(axs,fields):
        if frames_to_plot:
            frame = frames_to_plot[kf]
            plot_data = frame.values
        else:
            # Load and process data for all time frames
            frames = []
            for t in time_frame:
                frame = Frame(simulation, field, t)
                frame.load()
                
                if fourrier_y:
                    frame.fourrier_y()
                    
                frame.slice_2D(cut_dir, cut_coord)
                frames.append(frame)

            # Determine what to plot
            if fluctuation and len(time_frame) > 1:
                mean_values = np.mean([frame.values for frame in frames], axis=0)
                plot_data = frames[-1].values - mean_values[np.newaxis, ...]
                if fluctuation == "relative":
                    plot_data = 100.0*plot_data/mean_values[np.newaxis, ...]
            elif time_average and len(time_frame) > 1:
                plot_data = np.mean([frame.values for frame in frames], axis=0)  # Time-averaged data
            else:
                plot_data = frames[-1].values  # Single time frame data
            frame = frames[-1] # keep only the last one

        vsymbol = frame.vsymbol
        if ((field == 'phi' or field[:-1] == 'upar') or cmap0 == 'bwr'\
            or fluctuation) and not fourrier_y:
            cmap = 'bwr'
            vmax = np.max(np.abs(plot_data)) 
            vmin = -vmax
        else:
            cmap = cmap0
            vmax = np.max(plot_data)
            vmin = np.min(plot_data)

        if fourrier_y:
            vmin  = np.power(10,np.log10(vmax)-4)
            plot_data = np.clip(plot_data, vmin, None)  # We plot 4 orders of magnitude
            create_norm = mcolors.LogNorm
        else:
            create_norm = mcolors.Normalize

        if fluctuation:
            vsymbol = r'$\delta$'+ vsymbol
        elif time_average:
            vsymbol = r'$\langle$'+vsymbol + r'$\rangle$'
        YY,XX = np.meshgrid(frame.new_grids[1],frame.new_grids[0])
        norm  = create_norm(vmin=vmin, vmax=vmax)
        pcm = ax.pcolormesh(XX,YY,np.squeeze(plot_data),cmap=cmap,norm=norm)
        xlabel = label(frame.new_gsymbols[0],frame.new_gunits[0])
        ylabel = label(frame.new_gsymbols[1],frame.new_gunits[1])
        title  = frame.fulltitle
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if fourrier_y and False:
            title = r'FFT$_y($'+ frame.vsymbol + '), '+ title
        else:
            lbl = label(vsymbol,frame.vunits)
            if fluctuation == "relative" :
                lbl = re.sub(r'\(.*?\)', '', lbl)
                lbl = lbl + ' (\%)'               
            #if fluctuation or time_average:
                #lbl += ' (avg %2.2d to %2.2d)'%(frames[0].time,frames[-1].time)
            cbar = fig.colorbar(pcm,label=lbl)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        if clim: pcm.set_clim(clim[kf])    
        if not full_plot: ax.set_title(title)
        kf += 1 # field counter

    if full_plot:
        fig.suptitle(title)
    
    fig.tight_layout()
    # This allows to return the figure in the arguments line (used for movies)
    figout.append(fig)
    cutout.append(frame.slicecoords)
    val_out.append(np.squeeze(plot_data))

def make_2D_movie(simulation,cut_dir,cut_coord,time_frames, fieldnames,
                      cmap='inferno', xlim=[], ylim=[], clim=[], fluctuation = False,
                      full_plot=False, movieprefix=''):
    os.makedirs('gif_tmp', exist_ok=True)
    
    if isinstance(fieldnames,str):
        dataname = fieldnames + '_'
    else:
        dataname = ''
        for f_ in fieldnames:
            dataname += 'd'+f_+'_' if fluctuation else f_+'_'
    
    movie_frames, vlims = get_2D_movie_data(simulation, cut_dir, cut_coord, time_frames, fieldnames, fluctuation) 
    total_frames = len(time_frames)
    for i, tf in enumerate(time_frames, 1):  # Start the index at 1
        figout = []
        cutout = []
        clim = clim if clim else vlims
        plot_2D_cut(
            simulation, cut_dir=cut_dir, cut_coord=cut_coord, time_frame=tf, fieldnames=fieldnames,
            cmap=cmap, full_plot=full_plot,
            xlim=xlim, ylim=ylim, clim=clim, fluctuation=fluctuation,
            cutout=cutout, figout=figout, frames_to_plot=movie_frames[i-1]
        )
        fig = figout[0]
        fig.tight_layout()
        fig.savefig(f'gif_tmp/plot_{tf}.png')
        plt.close()

        # Update progress
        progress = f"Processing frames: {i}/{total_frames}... "
        sys.stdout.write("\r" + progress)
        sys.stdout.flush()
    sys.stdout.write("\n")
    # Naming
    cutout=cutout[0]
    cutname = [key+('=%2.2f'%cutout[key]) for key in cutout]
    moviename = movieprefix+'_movie_'+dataname+cutname[0]
    moviename+='_xlim_%2.2d_%2.2d'%(xlim[0],xlim[1]) if xlim else ''
    moviename+='_ylim_%2.2d_%2.2d'%(ylim[0],ylim[1]) if ylim else ''
    # moviename+='_clim_%2.2d_%2.2d'%(clim[0],clim[1]) if clim else ''
    moviename += '.gif'
    # Compiling the movie images
    images = [Image.open(f'gif_tmp/plot_{tf}.png') for tf in time_frames]
    # Save as gif
    images[0].save(moviename, save_all=True, append_images=images[1:], duration=200, loop=1)
    print("movie "+moviename+" created.")

def plot_domain(geometry,geom_type='Miller',vessel_corners=[[0.6,1.2],[-0.7,0.7]]):
    geometry.set_domain(geom_type,vessel_corners)
    fig = plt.figure(figsize=(default_figsz[0], default_figsz[1]))
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

def plot_GBsource(simulation,species,tf=0,ix=0,b=1.2,figout=[]):
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

    fig,ax = plt.subplots(1,1,figsize=(default_figsz[0],default_figsz[1]))
    ax[0].plot(zgrid,Gammaz,label='Effective source at ' + frame.timetitle)
    ax[0].plot(zgrid,fz,label='GB source model')
    ax[0].set_xlabel(r'$z$')
    ax[0].set_ylabel(r'$\int \Gamma_{\nabla B,x} dy$')
    ax[0].legend()
        
    Ssimul = np.trapz(Gammaz,x=zgrid, axis=0)
    Smodel = np.trapz(fz    ,x=zgrid, axis=0)

    ax[0].title('Total source simul: %2.2e 1/s, total source model: %2.2e 1/s'\
              %(Ssimul,Smodel))
    figout.append(fig)

def plot_volume_integral_vs_t(simulation, fieldnames, time_frames=[], ddt=False,
                              jacob_squared=False, plot_src_input=False,
                              add_GBloss = False, average = False, figout=[], rm_legend=False):
    fields,fig,axs = setup_figure(fieldnames)
    total_fields = len(fields)
    total_frames = len(time_frames)

    for field_idx, (ax, field) in enumerate(zip(axs, fields), 1):
        if not isinstance(field, list):
            subfields = [field]  # Simple plot
        else:
            subfields = field  # Field is a combined plot

        for subfield_idx, subfield in enumerate(subfields, 1):
            ftot_t = []
            time = []

            for frame_idx, tf in enumerate(time_frames, 1):
                f_ = Frame(simulation=simulation, name=subfield, tf=tf)
                f_.load()

                time.append(f_.time)
                ftot_t.append(f_.compute_volume_integral(jacob_squared=jacob_squared, average=average))

                # Update progress for time frame loop
                progress = (
                    f"Processing: Field {field_idx}/{total_fields}, "
                    f"Subfield {subfield_idx}/{len(subfields)}, "
                    f"Frame {frame_idx}/{total_frames}"
                )
                sys.stdout.write("\r" + progress)
                sys.stdout.flush()

            if ddt: # time derivative
                dfdt   = np.gradient(ftot_t,time)
                # we rescale it to obtain a result in seconds
                Ft = dfdt/simulation.normalization['tscale']
            else:
                Ft  = ftot_t
            # Convert to np arrays
            ftot_t = np.array(ftot_t)
            time   = np.array(time)

            # if we are talking about energy, add if needed the GBloss
            if subfield == 'Wtot' and add_GBloss: 
                gbl = np.zeros_like(ftot_t)
                for spec in simulation.species.values():
                    gbl_s, _ = simulation.get_GBloss_t(
                        spec    = spec,
                        twindow = time_frames,
                        ix      = 0,
                        losstype = 'energy',
                        integrate = True)
                    gbl = gbl+np.array(gbl_s)

            # Setup labels
            Flbl = simulation.normalization[subfield+'symbol']
            Flbl = r'$\int$ '+Flbl+r' $d^3x$'
            xlbl = label_from_simnorm(simulation,'t')
            if average:
                Flbl = Flbl + r'$/V$'
            else:
                ylbl = multiply_by_m3_expression(simulation.normalization[subfield+'units'])

            if ddt:
                Flbl = r'$\partial_t$ '+Flbl
                ylbl = ylbl+'/s'
            
            # Plot
            ax.plot(time,Ft,label=Flbl)
            if add_GBloss:
                ax.plot(time,Ft-gbl,label='w/o gB loss')
        # plot eventually the input power for comparison
        if subfield == 'Wtot' and plot_src_input:
            src_power = simulation.get_source_power()
            if ddt:
                ddtWsrc_t = src_power*np.ones_like(time)/simulation.normalization['Wtotscale']
                ax.plot(time,ddtWsrc_t,'--k',label='Source input (%2.2f MW)'%(src_power/1e6))
            else:
                # plot the accumulate energy from the source
                Wsrc_t = ftot_t[0] + src_power*simulation.normalization['tscale']*time/simulation.normalization['Wtotscale']
                ax.plot(time,Wsrc_t,'--k',label='Source input')

        # Print a newline after completing all frames for the current subfield
        sys.stdout.write("\n")

        # add labels and show legend
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        if not rm_legend:
            ax.legend()
        fig.tight_layout()
    figout.append(fig)

def plot_GB_loss(simulation, twindow, losstype = 'particle', integrate = False, figout = []):
    fields,fig,axs = setup_figure('onefield')
    for ax,field in zip(axs,fields):
        for spec in simulation.species.values():
            GBloss_t, time = simulation.get_GBloss_t(
                spec    = spec,
                twindow = twindow,
                ix      = 0,
                losstype = losstype,
                integrate = integrate)
            minus_GBloss = [-g for g in GBloss_t]
            axs[0].plot(time,minus_GBloss,label=r'$-S_{\nabla B %s, loss}$'%spec.nshort)

        if losstype == 'particle':
            ylabel = r'loss of particle'
        elif losstype == 'energy':
            ylabel = r'loss in MJ'
        if not integrate:
            ylabel = ylabel+'/s'
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r'$t$ ($\mu$s)')
        ax.legend()
        # ax.set_title('Particle loss at the inner flux surface')
        fig.tight_layout()
    figout.append(fig)

def plot_integrated_moment(simulation,fieldnames,xlim=[],ylim=[],ddt=False,plot_src_input=False,figout=[]):
    fields,fig,axs = setup_figure(fieldnames)
    for ax,field in zip(axs,fields):
        if not isinstance(field,list):
            subfields = [field] #simple plot
        else:
            subfields = field # field is a combined plot
        for subfield in subfields:
            if subfield[-1] == 'e':
                spec_s = 'elc'
            elif subfield[-1] == 'i':
                spec_s = 'ion'

            if subfield[:-1] in ['n']:
                def receipe(x): return x[:,0]
                scale = 1.0
                units = 'particles'
                symbol = r'$\bar n_%s$'%spec_s[0]
            elif subfield[:-1] in ['upar']:
                def receipe(x): return x[:,1]
                scale = simulation.species[spec_s].m*simulation.species[spec_s].vt
                units = ''
                symbol = r'$\bar u_{\parallel %s}/v_{t %s}$'%(spec_s[0],spec_s[0])
            elif subfield[:-1] in ['Tpar']:
                def receipe(x): return x[:,2]
                scale = simulation.species[spec_s].m
                units = 'eV'
                symbol = r'$\bar T_{\parallel %s}$'%spec_s[0]
            elif subfield[:-1] in ['Tperp']:
                def receipe(x): return x[:,3]
                scale = simulation.species[spec_s].m
                units = 'eV'
                symbol = r'$\bar T_{\perp %s}$'%spec_s[0]
            elif subfield[:-1] in ['T']:
                def receipe(x): return 1/3*(x[:,2]+2*x[:,3])
                scale = simulation.species[spec_s].m
                units = 'eV'
                symbol = r'$\bar T_{%s}$'%spec_s[0]
            elif subfield[:-1] in ['W']:
                def receipe(x): return 1/3*(x[:,2]+2*x[:,3])*x[:,0]
                scale = simulation.species[spec_s].m * simulation.phys_param.eV / 1e6
                units = 'MJ'
                symbol = r'$\bar n_{%s} \bar T_{%s}$'%(spec_s[0],spec_s[0])

            # Load the data from the file
            f_ = simulation.data_param.fileprefix+'-'+spec_s+'_integrated_moms.gkyl'
            Gdata = pg.data.GData(f_)
            int_moms = pgkyl_.get_values(Gdata)
            time = np.squeeze(Gdata.get_grid()) / simulation.normalization['tscale']

            Ft = receipe(int_moms)
            Ft = np.squeeze(Ft)
            # resacle
            Ft = Ft * scale
            # remove double diagnostic
            time, indices = np.unique(time, return_index=True)
            Ft = Ft[indices]
            # Labels
            Flbl = ddt*r'$\partial_t$ '+symbol
            ylbl = units
            if ddt: # time derivative
                dfdt   = np.gradient(Ft,time,edge_order=2)
                # we rescale it to obtain a result in seconds
                Ft = dfdt/simulation.normalization['tscale']
                if ylbl == 'MJ':
                    ylbl = 'MW'
                else:
                    ylbl = units+'/s'
            # Plot
            ax.plot(time,Ft,label=Flbl)

        # plot eventually the input power for comparison
        if subfield == 'Wtot' and plot_src_input:
            src_power = simulation.get_source_power()
            if ddt:
                ddtWsrc_t = src_power*np.ones_like(time)/simulation.normalization['Wtotscale']
                ax.plot(time,ddtWsrc_t,'--k',label='Source input (%2.2f MW)'%(src_power/1e6))
            else:
                # plot the accumulate energy from the source
                Wsrc_t = Ft[0] + src_power*simulation.normalization['tscale']*time/simulation.normalization['Wtotscale']
                ax.plot(time,Wsrc_t,'--k',label='Source input')
        # add labels and show legend
        ax.set_xlabel(simulation.normalization['tunits'])
        ax.set_ylabel(ylbl)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend()
        fig.tight_layout()
    figout.append(fig)
    return time

#----- Retrocompatibility
plot_1D_time_avg = plot_1D
