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
from ..utils import math_utils
from .. utils import file_utils
from ..tools import fig_tools
from ..utils import data_utils
from ..classes import Frame
# other commonly used libs
import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from ..tools import pgkyl_interface as pgkyl_

# set the font to be LaTeX
if file_utils.check_latex_installed(verbose=False):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams.update({'font.size': 14})

import os, re
    
def plot_1D_time_evolution(simulation,cdirection,ccoords,fieldnames='',
                           twindow=[],space_time=False, cmap='inferno',
                           fluctuation=False,
                           xlim=[], ylim=[], clim=[], figout=[]):
    if not isinstance(twindow,list): twindow = [twindow]
    cmap0 = cmap
    fields,fig,axs = fig_tools.setup_figure(fieldnames)
    for ax,field in zip(axs,fields):
        x,t,values,xlabel,tlabel,vlabel,vunits,slicetitle,fourier_y =\
              data_utils.get_1xt_diagram(simulation,field,cdirection,ccoords,tfs=twindow)
        if fluctuation:
            average_data = np.mean(values, axis=1)
            vlabel = r'$\delta$' + vlabel
            for it in range(len(t)) :
                values[:,it] = (values[:,it] - average_data[:])
            if fluctuation == "relative":
                values = 100.0*values/average_data
                vlabel = re.sub(r'\(.*?\)', '', vlabel)
                vlabel = vlabel + ' (\%)'
        if space_time:
            if ((field in ['phi','upare','upari']) or cmap0=='bwr' or fluctuation) and not fourier_y:
                cmap = 'bwr'
                vmax = np.max(np.abs(values)) 
                vmin = -vmax
            else:
                cmap = cmap0
                vmax = np.max(np.abs(values)) 
                vmin = 0.0
            # handle fourier plot
            colorscale = 'linear' if not fourier_y else 'log'
            vmin = np.power(10,np.log10(vmax)-4) if fourier_y else vmin
            values = np.clip(values, vmin, None) if fourier_y else values 
            # make the plot
            fig_tools.plot_2D(fig,ax,x=x,y=t,z=values, xlim=xlim, ylim=ylim, clim=clim,
                              xlabel=xlabel, ylabel=tlabel, clabel=vlabel, title=slicetitle,
                              cmap=cmap, vmin=vmin, vmax=vmax, colorscale=colorscale)
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
    
    fields,fig,axs = fig_tools.setup_figure(fieldnames)

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
            x,t,values,xlabel,tlabel,vlabel,vunits,slicetitle,fourier_y =\
                data_utils.get_1xt_diagram(simulation,subfield,cdirection,ccoords,tfs=time_frames)
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
                time_average=False, fluctuation=False, plot_type='pcolormesh',
                xlim=[], ylim=[], clim=[], colorscale = 'linear',
                figout=[],cutout=[], val_out=[], frames_to_plot = None):
    # Check if we provide multiple time frames (time average or fluctuation plot)
    if isinstance(time_frame, int):
        time_frame = [time_frame]
    if clim:
        clim = [clim] if not isinstance(clim[0], list) else clim

    # Check if we need to fourier transform
    index = cut_dir.find('k')
    if index > -1:
        fourier_y = True
        cut_dir = cut_dir.replace('k','')
    else:
        fourier_y = False

    cmap0 = cmap    
    fields,fig,axs = fig_tools.setup_figure(fieldnames)
    kf = 0 # field counter
    for ax,field in zip(axs,fields):
        if frames_to_plot:
            frame = frames_to_plot[kf]
            plot_data = frame.values
        else:
            # Load and process data for all time frames
            frames = []
            for t in time_frame:
                frame = Frame(simulation, field, t, load=True)                
                if fourier_y: frame.fourier_y()
                frame.slice(cut_dir, cut_coord)
                # frame.slice_2D(cut_dir, cut_coord)
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

        if ((field == 'phi' or field[:-1] == 'upar') or cmap0 == 'bwr'\
            or fluctuation) and not fourier_y:
            cmap = 'bwr'
            vmax = np.max(np.abs(plot_data)) 
            vmin = -vmax
        else:
            cmap = cmap0
            vmax = np.max(plot_data)
            vmin = np.min(plot_data)

        if fourier_y:
            vmin  = np.power(10,np.log10(vmax)-4)
            plot_data = np.clip(plot_data, vmin, None)  # We plot 4 orders of magnitude
            colorscale = 'log'

        vsymbol = frame.vsymbol
        if fluctuation:
            vsymbol = r'$\delta$'+ vsymbol
        elif time_average:
            vsymbol = r'$\langle$'+ vsymbol + r'$\rangle$'
        lbl = fig_tools.label(vsymbol,frame.vunits)

        if fluctuation == "relative" :
            lbl = re.sub(r'\(.*?\)', '', lbl)
            lbl = lbl + ' (\%)'
            if fluctuation or time_average:
                lbl += ' (avg %2.2d to %2.2d)'%(frames[0].time,frames[-1].time)

        climf = clim[kf] if clim else None
        fig_tools.plot_2D(fig,ax,x=frame.new_grids[0],y=frame.new_grids[1],z=plot_data, 
                          cmap=cmap, xlim=xlim, ylim=ylim, clim=climf,
                          xlabel=frame.new_gsymbols[0], ylabel=frame.new_gsymbols[1], 
                          colorscale=colorscale, clabel=lbl, title=frame.fulltitle, 
                          vmin=vmin, vmax=vmax, plot_type=plot_type)
        kf += 1 # field counter
    
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
    
    movie_frames, vlims = data_utils.get_2D_movie_data(simulation, cut_dir, cut_coord, time_frames, fieldnames, fluctuation) 
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
    moviename += '.gif'
    # Compiling the movie images
    images = [Image.open(f'gif_tmp/plot_{tf}.png') for tf in time_frames]
    # Save as gif
    images[0].save(moviename, save_all=True, append_images=images[1:], duration=200, loop=1)
    print("movie "+moviename+" created.")

def plot_domain(geometry,geom_type='Miller',vessel_corners=[[0.6,1.2],[-0.7,0.7]]):
    geometry.set_domain(geom_type,vessel_corners)
    fig = plt.figure(figsize=(fig_tools.default_figsz[0], fig_tools.default_figsz[1]))
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

    fig,ax = plt.subplots(1,1,figsize=(fig_tools.default_figsz[0],fig_tools.default_figsz[1]))
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
    fields,fig,axs = fig_tools.setup_figure(fieldnames)
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
                Ft = dfdt/simulation.normalization.dic['tscale']
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
            Flbl = simulation.normalization.dict[subfield+'symbol']
            Flbl = r'$\int$ '+Flbl+r' $d^3x$'
            xlbl = fig_tools.label_from_simnorm(simulation,'t')
            if average:
                Flbl = Flbl + r'$/V$'
            else:
                ylbl = fig_tools.multiply_by_m3_expression(simulation.normalization.dict[subfield+'units'])

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
                ddtWsrc_t = src_power*np.ones_like(time)/simulation.normalization.dict['Wtotscale']
                ax.plot(time,ddtWsrc_t,'--k',label='Source input (%2.2f MW)'%(src_power/1e6))
            else:
                # plot the accumulate energy from the source
                Wsrc_t = ftot_t[0] + src_power*simulation.normalization.dict['tscale']*time/simulation.normalization.dict['Wtotscale']
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
    fields,fig,axs = fig_tools.setup_figure('onefield')
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
    fields,fig,axs = fig_tools.setup_figure(fieldnames)
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
            elif subfield[-3:] == 'tot':
                spec_s = ['elc','ion']

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
            elif subfield[:-1] in ['W','Pow']:
                def receipe(x): return 1/3*(x[:,2]+2*x[:,3])
                scale = simulation.species[spec_s].m / 1e6                
                units = 'MJ'
                symbol = r'$W_{kin,%s}$'%spec_s[0]
            elif subfield in ['Wtot']:
                def receipe(x): return 1/3*(x[:,2]+2*x[:,3])
                scale = [simulation.species[s].m / 1e6 for s in spec_s]
                units = 'MJ'
                symbol = r'$W_{kin,tot}$'
            elif subfield in ['ntot']:
                def receipe(x): return x[:,0]
                scale = [1.0 for s in spec_s]      
                units = 'particles'
                symbol = r'$\bar n_{tot}$'

            # Load the data from the file(s)
            int_moms = 0
            if isinstance(spec_s,list):
                for s in spec_s:
                    f_ = simulation.data_param.fileprefix+'-'+s+'_integrated_moms.gkyl'
                    Gdata = pg.data.GData(f_)
                    int_moms += pgkyl_.get_values(Gdata) * scale[spec_s.index(s)]
            else:
                f_ = simulation.data_param.fileprefix+'-'+spec_s+'_integrated_moms.gkyl'
                Gdata = pg.data.GData(f_)
                int_moms = pgkyl_.get_values(Gdata) * scale

            time = np.squeeze(Gdata.get_grid()) / simulation.normalization.dict['tscale']

            Ft = receipe(int_moms)
            Ft = np.squeeze(Ft)
            # remove double diagnostic
            time, indices = np.unique(time, return_index=True)
            Ft = Ft[indices]
            # Labels
            Flbl = ddt*r'$\partial_t$ '+symbol
            ylbl = units
            if ddt: # time derivative
                dfdt   = np.gradient(Ft,time,edge_order=2)
                # we rescale it to obtain a result in seconds
                Ft = dfdt/simulation.normalization.dict['tscale']
                if ylbl == 'MJ':
                    ylbl = 'MW'
                else:
                    ylbl = units+'/s'
            # Plot
            ax.plot(time,Ft,label=Flbl)

        # plot eventually the input power for comparison
        if subfield in ['Wtot','ntot'] and plot_src_input:
            src_input = simulation.get_source_power() if subfield == 'Wtot' else simulation.get_source_particle()
            if subfield == 'Wtot': src_input /= simulation.normalization.dict['Wtotscale']
            if ddt:
                ddtWsrc_t = src_input*np.ones_like(time)
                ax.plot(time,ddtWsrc_t,'--k',label='Source input')
            else:
                # plot the accumulate energy from the source
                Wsrc_t = Ft[0] + src_input*simulation.normalization.dict['tscale']*time
                ax.plot(time,Wsrc_t,'--k',label='Source input')
            
        # add labels and show legend
        ax.set_xlabel(simulation.normalization.dict['tunits'])
        ax.set_ylabel(ylbl)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend()
        fig.tight_layout()
    figout.append(fig)
    return time

def plot_sources_info(simulation,x_const=0,z_const=0):
    """
    Plot the profiles of all sources in the sources dictionary using various cuts.
    """
    print("-- Source Informations --")
    simulation.get_source_particle(type=type, remove_GB_loss=True)
    simulation.get_source_power(type=type, remove_GB_loss=True)
    banana_width = simulation.get_banana_width(x=0.0, z=z_const, spec='ion')
    nrow = 1*(x_const != None) + 1*(z_const != None) + 1
    if len(simulation.sources) > 0:
        x_grid, _, z_grid = simulation.geom_param.grids
        y_const = 0.0
        _, axs = plt.subplots(nrow, 2, figsize=(2*fig_tools.default_figsz[0],nrow*fig_tools.default_figsz[1]))
        axs = axs.flatten()
        iplot = 0
        if z_const != None:
            # Plot the density profiles at constant y and z
            for (name, source) in simulation.sources.items():
                n_src = [source.density_src(x, y_const, z_const) for x in x_grid]
                axs[iplot].plot(x_grid, n_src, label=r"$\dot n$ "+name, linestyle="-")
            axs[iplot].axvline(x=banana_width, color='cyan', linestyle='-', label='Banana width' % banana_width, alpha=0.5)
            axs[iplot].axvline(x=simulation.geom_param.x_LCFS, color='gray', linestyle='-', label='LCFS', alpha=0.7)
            axs[iplot].set_xlabel("x-grid [m]")
            axs[iplot].set_ylabel(r"1/m$^3/s$")
            axs[iplot].legend()
            iplot += 1
            # Plot the temperature profiles at constant x and z
            for (name, source) in simulation.sources.items():
                Te_src = [source.temp_profile_elc(x, y_const, z_const)/simulation.phys_param.eV for x in x_grid]
                Ti_src = [source.temp_profile_ion(x, y_const, z_const)/simulation.phys_param.eV for x in x_grid]
                axs[iplot].plot(x_grid, Te_src, label=r"$T_e$ "+name, linestyle="--")
                axs[iplot].plot(x_grid, Ti_src, label=r"$T_i$ "+name, linestyle="-.")
            axs[iplot].axvline(x=banana_width, color='cyan', linestyle='-', label='Banana width' % banana_width, alpha=0.5)
            axs[iplot].axvline(x=simulation.geom_param.x_LCFS, color='gray', linestyle='-', label='LCFS', alpha=0.7)
            axs[iplot].set_xlabel("x-grid [m]")
            axs[iplot].set_ylabel("eV")
            axs[iplot].legend()
            iplot += 1

        # Plot the density and temperature profiles at constant x and y
        if x_const != None:
            for (name, source) in simulation.sources.items():
                n_src = [source.density_src(x_const, y_const, z) for z in z_grid]
                axs[iplot].plot(z_grid, n_src, label=r"$\dot n$ "+name, linestyle="-")
            axs[iplot].set_xlabel("z-grid [rad]")
            axs[iplot].set_ylabel(r"1/m$^3/s$")
            axs[iplot].legend()
            iplot += 1
            for (name, source) in simulation.sources.items():
                Te_src = [source.temp_profile_elc(x_const, y_const, z)/simulation.phys_param.eV for z in z_grid]
                Ti_src = [source.temp_profile_ion(x_const, y_const, z)/simulation.phys_param.eV for z in z_grid]
                axs[iplot].plot(z_grid, Te_src, label=r"$T_e$ "+name, linestyle="--")
                axs[iplot].plot(z_grid, Ti_src, label=r"$T_i$ "+name, linestyle="-.")
            axs[iplot].set_xlabel("z-grid [rad]")
            axs[iplot].set_ylabel("eV")
            axs[iplot].legend()
            iplot += 1
        
        # Build the total source profiles on a xz meshgrid
        XX, ZZ = math_utils.custom_meshgrid(x_grid, z_grid)
        total_dens_src = np.zeros_like(XX)
        total_pow_src = np.zeros_like(XX)
        for (name, source) in simulation.sources.items():
            total_dens_src += source.density_src(XX, y_const, ZZ)
            total_pow_src += source.density_src(XX, y_const, ZZ) * (source.temp_profile_elc(XX, y_const, ZZ) + source.temp_profile_ion(XX, y_const, ZZ))/1e6

        # Plot the total density source profile
        pcm = axs[iplot].pcolormesh(XX, ZZ, total_dens_src, cmap='inferno')
        cbar = plt.colorbar(pcm, ax=axs[iplot])
        cbar.set_label(r"1/m$^3/s$")
        axs[iplot].set_xlabel("x-grid [m]")
        axs[iplot].set_ylabel("z-grid [rad]")
        axs[iplot].set_title(r"Total $\dot n$ src")
        axs[iplot].axvline(x=banana_width, color='cyan', linestyle='-', label='bw' % banana_width, alpha=0.7)
        axs[iplot].axvline(x=simulation.geom_param.x_LCFS, color='gray', linestyle='-', label='LCFS', alpha=0.7)
        iplot += 1

        # Plot the total power
        pcm = axs[iplot].pcolormesh(XX, ZZ, total_pow_src, cmap='inferno')
        cbar = plt.colorbar(pcm, ax=axs[iplot])
        cbar.set_label(r"MW/m$^3$")
        axs[iplot].set_xlabel("x-grid [m]")
        axs[iplot].set_ylabel("z-grid [rad]")
        axs[iplot].set_title(r"Total power src")
        axs[iplot].axvline(x=banana_width, color='cyan', linestyle='-', label='bw' % banana_width, alpha=0.7)
        axs[iplot].axvline(x=simulation.geom_param.x_LCFS, color='gray', linestyle='-', label='LCFS' % banana_width, alpha=0.7)

        plt.tight_layout()
        plt.show()


#----- Retrocompatibility
plot_1D_time_avg = plot_1D
