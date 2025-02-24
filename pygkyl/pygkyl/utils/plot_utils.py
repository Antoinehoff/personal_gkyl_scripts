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

# personnal classes and routines
from ..utils import math_utils
from .. utils import file_utils
from ..tools import fig_tools
from ..utils import data_utils
from ..classes import Frame, IntegratedMoment, PoloidalProjection

# other commonly used libs
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# set the font to be LaTeX
if file_utils.check_latex_installed(verbose=False):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams.update({'font.size': 14})

import os, re
    
def plot_1D_time_evolution(simulation,cdirection,ccoords,fieldnames='',
                           twindow=[],space_time=False, cmap='inferno',
                           fluctuation=False, plot_type='pcolormesh',
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
        # handle fourier plot
        colorscale = 'linear' if not fourier_y else 'log'
        if space_time:
            if ((field in ['phi','upare','upari']) or cmap0=='bwr' or fluctuation) and not fourier_y:
                cmap = 'bwr'
                vmax = np.max(np.abs(values)) 
                vmin = -vmax
            else:
                cmap = cmap0
                vmax = np.max(np.abs(values)) 
                vmin = 0.0
            vmin = np.power(10,np.log10(vmax)-4) if fourier_y else vmin
            values = np.clip(values, vmin, None) if fourier_y else values 
            # make the plot
            fig_tools.plot_2D(fig,ax,x=x,y=t,z=values, xlim=xlim, ylim=ylim, clim=clim,
                              xlabel=xlabel, ylabel=tlabel, clabel=vlabel, title=slicetitle,
                              cmap=cmap, vmin=vmin, vmax=vmax, colorscale=colorscale, plot_type=plot_type)
        else:
            norm = plt.Normalize(min(t), max(t))
            colormap = cm.viridis  # You can choose any colormap
            for it in range(len(t)):
                ax.plot(x,values[:,it],label=r'$t=%2.2e$ (ms)'%(t[it]),
                        color=colormap(norm(t[it])))
            # Add a colorbar to the figure
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm);sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            fig_tools.finalize_plot(ax, fig, title=slicetitle[:-2], figout=figout, xlim=xlim, ylim=ylim,
                                    xlabel=xlabel, ylabel=vlabel, clabel=tlabel, cbar=cbar)

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
            average_data = np.mean(values, axis=1)
            # Compute the standard deviation of data over the t-axis (axis=1)
            std_dev_data = np.std(values, axis=1)
            if time_avg :
                if errorbar:
                    ax.errorbar(x, average_data, yerr=std_dev_data, 
                                fmt='o', capsize=5, label=vlabel)
                else:
                    ax.plot(x, average_data, label=vlabel)
            else:
                # Classic plot
                ax.plot(x, values, label=vlabel)
                if periodicity > 0:
                    ax.plot(x+periodicity, values, label=vlabel)
        # Labels and title
        ylabel = vunits if len(subfields)>1 else vlabel
        show_legend = len(subfields)>1
        title = slicetitle+tlabel+r'$=%2.2e$'%(t[0]) if t[0] == t[-1] else \
                slicetitle+tlabel+r'$\in[%2.2e,%2.2e]$'%(t[0],t[-1])
        fig_tools.finalize_plot(ax, fig, xlabel=xlabel, ylabel=ylabel, title=title, legend=show_legend,
                                figout=figout, grid=grid, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)

def plot_2D_cut(simulation,cut_dir,cut_coord,time_frame,
                fieldnames='', cmap='inferno', time_average=False, fluctuation=False, plot_type='pcolormesh',
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

def make_2D_movie(simulation, cut_dir='xy', cut_coord=0.0, time_frames=[], fieldnames=['phi'],
                  cmap='inferno', xlim=[], ylim=[], clim=[], fluctuation = False,
                  movieprefix='', plot_type='pcolormesh', 
                  colorScale='lin', scaleFac=1.0, logScaleFloor=1e-3,
                  polProj=None):
    # Create a temporary folder to store the movie frames
    movDirTmp = 'movie_frames_tmp'
    os.makedirs(movDirTmp, exist_ok=True)
    
    if isinstance(fieldnames,str):
        dataname = fieldnames + '_'
    else:
        dataname = ''
        for f_ in fieldnames:
            dataname += 'd'+f_+'_' if fluctuation else f_+'_'
    if 'RZ' in cut_dir:
        vlims, vlims_SOL = data_utils.get_minmax_values(simulation, fieldnames, time_frames)
        if cmap == 'inferno': 
            vlims[0] = np.max([0,vlims[0]])
            vlims_SOL[0] = np.max([0,vlims_SOL[0]])
        elif cmap == 'bwr':
            vmax = np.max(np.abs(vlims))
            vlims = [-vmax, vmax]
            vmax_SOL = np.max(np.abs(vlims_SOL))
            vlims_SOL = [-vmax_SOL, vmax_SOL]

        # Harvest a possible number with the cut_dir
        nzInterp = cut_dir.replace('RZ','')
        nzInterp = int(nzInterp) if nzInterp else 32
        # Setup poloidal projection plot
        if not polProj:
            polProj = PoloidalProjection()
            polProj.setup(simulation, fieldName=fieldnames, timeFrame=time_frames, nzInterp=nzInterp)

    else:
        movie_frames, vlims = data_utils.get_2D_movie_data(simulation, cut_dir, cut_coord, 
                                                            time_frames, fieldnames, fluctuation) 
    
    total_frames = len(time_frames)
    frameFileList = []
    for i, tf in enumerate(time_frames, 1):  # Start the index at 1  

        frameFileName = f'movie_frames_tmp/frame_{tf}.png'
        frameFileList.append(f'movie_frames_tmp/frame_{tf}.png')

        figout = []
        cutout = []
        if clim == 'free':
            clim = []
        else:
            clim = clim if clim else vlims

        if 'RZ' in cut_dir:
            polProj.plot(fieldName=fieldnames, timeFrame=tf, outFilename=frameFileName,
                         colorMap = cmap, doInset=True, scaleFac=scaleFac,
                         colorScale=colorScale, logScaleFloor=logScaleFloor,
                         xlim=xlim, ylim=ylim, clim=clim, climSOL=vlims_SOL)
            cutname = ['RZ'+str(nzInterp)]
        else:
            plot_2D_cut(
                simulation, cut_dir=cut_dir, cut_coord=cut_coord, time_frame=tf, fieldnames=fieldnames,
                cmap=cmap, plot_type=plot_type,
                xlim=xlim, ylim=ylim, clim=clim, fluctuation=fluctuation,
                cutout=cutout, figout=figout, frames_to_plot=movie_frames[i-1]
            )
            fig = figout[0]
            fig.tight_layout()
            fig.savefig(frameFileName)
            plt.close()
            cutout=cutout[0]
            cutname = [key+('=%2.2f'%cutout[key]) for key in cutout]

        # Update progress
        progress = f"Processing frames: {i}/{total_frames}... "
        sys.stdout.write("\r" + progress)
        sys.stdout.flush()

    sys.stdout.write("\n")
    
    # Naming

    movieName = movieprefix+'_'+dataname+cutname[0] if movieprefix else dataname+cutname[0]
    movieName+='_xlim_%2.2d_%2.2d'%(xlim[0],xlim[1]) if xlim else ''
    movieName+='_ylim_%2.2d_%2.2d'%(ylim[0],ylim[1]) if ylim else ''

    # Compiling the movie images
    fig_tools.compile_movie(frameFileList, movieName, rmFrames=True)


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
    fig_tools.finalize_plot(ax, fig, xlabel='R (m)', ylabel='Z (m)', aspect='equal')

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
    Ssimul = np.trapz(Gammaz,x=zgrid, axis=0)
    Smodel = np.trapz(fz    ,x=zgrid, axis=0)
    fig_tools.finalize_plot(ax[0], fig, xlabel=r'$z$', ylabel=r'$\int \Gamma_{\nabla B,x} dy$', legend=True,
                            title = 'Total source simul: %2.2e 1/s, total source model: %2.2e 1/s'%(Ssimul,Smodel))

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
        fig_tools.finalize_plot(ax, fig, xlabel=xlbl, ylabel=ylbl, figout=figout, legend=not rm_legend)

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

        if losstype == 'particle': ylabel = r'loss of particle'
        elif losstype == 'energy': ylabel = r'loss in MJ'
        if not integrate: ylabel = ylabel+'/s'
        fig_tools.finalize_plot(ax, fig, xlabel=r'$t$ ($\mu$s)', ylabel=ylabel, figout=figout, legend=True)

def plot_integrated_moment(simulation,fieldnames,xlim=[],ylim=[],ddt=False,plot_src_input=False,figout=[],twindow=[]):
    fields,fig,axs = fig_tools.setup_figure(fieldnames)
    for ax,field in zip(axs,fields):
        if not isinstance(field,list):
            subfields = [field] #simple plot
        else:
            subfields = field # field is a combined plot
        for subfield in subfields:
            int_mom = IntegratedMoment(simulation=simulation, name=subfield, ddt=ddt)
            # select the time window
            if twindow:
                it0 = np.argmin(abs(int_mom.time-twindow[0]))
                it1 = np.argmin(abs(int_mom.time-twindow[1]))
                int_mom.time = int_mom.time[it0:it1]
                int_mom.values = int_mom.values[it0:it1]
            # Plot
            ax.plot(int_mom.time,int_mom.values,label=int_mom.symbol)
        # plot eventually the input power for comparison
        if subfield in ['Wtot','ntot'] and plot_src_input:
            src_input = simulation.get_source_power() if subfield == 'Wtot' else simulation.get_source_particle()
            if subfield == 'Wtot': src_input /= simulation.normalization.dict['Wtotscale']
            if ddt:
                ddtWsrc_t = src_input*np.ones_like(int_mom.time)
                ax.plot(int_mom.time,ddtWsrc_t,'--k',label='Source input')
            else:
                # plot the accumulate energy from the source
                Wsrc_t = int_mom.values[0] + src_input*simulation.normalization.dict['tscale']*int_mom.time
                ax.plot(int_mom.time,Wsrc_t,'--k',label='Source input')
            
        # add labels and show legend
        fig_tools.finalize_plot(ax, fig, xlabel=int_mom.tunits, ylabel=int_mom.vunits, figout=figout, xlim=xlim, ylim=ylim, legend=True)
    return int_mom.time

def plot_sources_info(simulation,x_const=None,z_const=None,show_LCFS=False,profileORgkyldata='profile'):
    """
    Plot the profiles of all sources in the sources dictionary using various cuts.
    """
    print("-- Source Informations --")
    simulation.get_source_particle(profileORgkyldata=profileORgkyldata, remove_GB_loss=True)
    simulation.get_source_power(profileORgkyldata=profileORgkyldata, remove_GB_loss=True)
    banana_width = simulation.get_banana_width(x=0.0, z=z_const, spec='ion')
    nrow = 1*(x_const != None) + 1*(z_const != None) + 1
    if len(simulation.sources) > 0:
        x_grid, _, z_grid = simulation.geom_param.grids
        y_const = 0.0
        x0 = x_grid[0]
        fig, axs = plt.subplots(nrow, 2, figsize=(2*fig_tools.default_figsz[0],nrow*fig_tools.default_figsz[1]))
        axs = axs.flatten()
        iplot = 0
        if z_const != None:
            # Plot the density profiles at constant y and z
            for (name, source) in simulation.sources.items():
                n_src = [source.density_src(x, y_const, z_const) for x in x_grid]
                axs[iplot].plot(x_grid, n_src, label=r"$\dot n$ "+name, linestyle="-")
            axs[iplot].axvline(x=x0+banana_width, color='cyan', linestyle='-', label='Banana width' % banana_width, alpha=0.5)
            if show_LCFS:
                axs[iplot].axvline(x=simulation.geom_param.x_LCFS, color='gray', linestyle='-', label='LCFS', alpha=0.7)
            fig_tools.finalize_plot(ax=axs[iplot], fig=fig, xlabel="x-grid [m]", ylabel=r"1/m$^3/s$", legend=True)
            iplot += 1
            # Plot the temperature profiles at constant x and z
            for (name, source) in simulation.sources.items():
                Te_src = [source.temp_profile_elc(x, y_const, z_const)/simulation.phys_param.eV for x in x_grid]
                Ti_src = [source.temp_profile_ion(x, y_const, z_const)/simulation.phys_param.eV for x in x_grid]
                axs[iplot].plot(x_grid, Te_src, label=r"$T_e$ "+name, linestyle="--")
                axs[iplot].plot(x_grid, Ti_src, label=r"$T_i$ "+name, linestyle="-.")
            axs[iplot].axvline(x=x0+banana_width, color='cyan', linestyle='-', label='Banana width' % banana_width, alpha=0.5)
            if show_LCFS:
                axs[iplot].axvline(x=simulation.geom_param.x_LCFS, color='gray', linestyle='-', label='LCFS', alpha=0.7)
            fig_tools.finalize_plot(ax=axs[iplot], fig=fig, xlabel="x-grid [m]", ylabel="eV", legend=True)
            iplot += 1

        # Plot the density and temperature profiles at constant x and y
        if x_const != None:
            for (name, source) in simulation.sources.items():
                n_src = [source.density_src(x_const, y_const, z) for z in z_grid]
                axs[iplot].plot(z_grid, n_src, label=r"$\dot n$ "+name, linestyle="-")
            fig_tools.finalize_plot(ax=axs[iplot], fig=fig, xlabel="z-grid [rad]", ylabel=r"1/m$^3/s$", legend=True)
            iplot += 1
            for (name, source) in simulation.sources.items():
                Te_src = [source.temp_profile_elc(x_const, y_const, z)/simulation.phys_param.eV for z in z_grid]
                Ti_src = [source.temp_profile_ion(x_const, y_const, z)/simulation.phys_param.eV for z in z_grid]
                axs[iplot].plot(z_grid, Te_src, label=r"$T_e$ "+name, linestyle="--")
                axs[iplot].plot(z_grid, Ti_src, label=r"$T_i$ "+name, linestyle="-.")
            fig_tools.finalize_plot(ax=axs[iplot], fig=fig, xlabel="z-grid [rad]", ylabel="eV", legend=True)
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
        axs[iplot].axvline(x=x0+banana_width, color='cyan', linestyle='-', label='bw' % banana_width, alpha=0.7)
        if show_LCFS:
            axs[iplot].axvline(x=simulation.geom_param.x_LCFS, color='gray', linestyle='-', label='LCFS', alpha=0.7)
        fig_tools.finalize_plot(ax=axs[iplot], fig=fig, xlabel="x-grid [m]", ylabel="z-grid [rad]", title=r"Total $\dot n$ src",
                                cbar=cbar, clabel=r"1/m$^3$", legend=False)
        iplot += 1

        # Plot the total power
        pcm = axs[iplot].pcolormesh(XX, ZZ, total_pow_src, cmap='inferno')
        cbar = plt.colorbar(pcm, ax=axs[iplot])
        axs[iplot].axvline(x=x0+banana_width, color='cyan', linestyle='-', label='bw' % banana_width, alpha=0.7)
        if show_LCFS:
            axs[iplot].axvline(x=simulation.geom_param.x_LCFS, color='gray', linestyle='-', label='LCFS' % banana_width, alpha=0.7)
        fig_tools.finalize_plot(ax=axs[iplot], fig=fig, xlabel="x-grid [m]", ylabel="z-grid [rad]", title="Total power src",
                                cbar=cbar, clabel=r"MW/m$^3$", legend=False)


def plot_DG_representation(simulation, fieldname, sim_frame, cutdir='x', cutcoord=[0.0,0.0], xlim=[], show_cells=True, figout=[],
                           derivative=False):
    """
    Plot the DG representation of a field at a given time frame.
    """
    if derivative in ['x','y','z']:
        id = 0 * (derivative == 'x') + 1 * (derivative == 'y') + 2 * (derivative == 'z')
    else :
        id = None

    frame = Frame(simulation, fieldname,tf=sim_frame, load=True)
    frame.slice(cutdir, cutcoord)
    # get the coordinates of the slice
    slice_coords = [c for c in frame.slicecoords.values()]
    field_DG = frame.get_DG_coeff()
    if cutdir == 'x':
        ix = 0
        slice_coords[0] *= simulation.normalization.dict['yscale']
        slice_coords[1] *= simulation.normalization.dict['zscale']
        def coord_swap(x): return [x,c0,c1]
    elif cutdir == 'y':
        ix = 1
        slice_coords[0] *= simulation.normalization.dict['xscale']
        slice_coords[1] *= simulation.normalization.dict['zscale']
        def coord_swap(x): return [c0,x,c1]
    elif cutdir == 'z':
        ix = 2
        slice_coords[0] *= simulation.normalization.dict['xscale']
        slice_coords[1] *= simulation.normalization.dict['yscale']
        def coord_swap(x): return [c0,c1,x]
    else:
        raise Exception("Invalid direction")
    cells = field_DG.grid[ix]
    c0 = slice_coords[0]
    c1 = slice_coords[1]
    dx = cells[1]-cells[0]
    DG_proj = []
    x_proj  = []
    xscale = simulation.normalization.dict[cutdir+'scale']
    xshift = simulation.normalization.dict[cutdir+'shift']
    yscale = simulation.normalization.dict[fieldname+'scale']
    for ic in range(len(cells)-1):
        xi = cells[ic]+0.01*dx
        fi = simulation.DG_basis.eval_proj(field_DG, coord_swap(xi), id=id)
        xip1 = cells[ic]+0.99*dx
        fip1 = simulation.DG_basis.eval_proj(field_DG, coord_swap(xip1), id=id)
        DG_proj.append(fi/yscale)
        x_proj.append(xi/xscale - xshift)
        DG_proj.append(fip1/yscale)
        x_proj.append(xip1/xscale - xshift)
        DG_proj.append(None)
        x_proj.append(cells[ic]/xscale - xshift)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_proj, DG_proj,'-')
    # add a vertical line to mark each cell boundary
    if show_cells:
        for xc in cells:
                ax.axvline(xc/xscale - xshift, color='k', linestyle='-', alpha=0.15)

    xlabel = fig_tools.label_from_simnorm(simulation,cutdir)
    ylabel = fig_tools.label_from_simnorm(simulation,fieldname)
    if derivative in ['x','y','z']:
        ylabel.replace(')','')
        ylabel = r'$\partial_%s$'%derivative+ylabel
        if frame.gunits[id] != '':
            ylabel += '/'+ frame.gunits[id] + ')'
    title = frame.slicetitle + ' at ' + frame.timetitle
    fig_tools.finalize_plot(ax, fig, xlabel=xlabel, ylabel=ylabel, title=title, figout=figout, xlim=xlim)
    
#----- Retrocompatibility
plot_1D_time_avg = plot_1D

def plot_poloidal_projection(simulation, fieldName='phi', timeFrame=0, outFilename='',nzInterp=32, scaleFac=1.,
                        colorMap = 'inferno', colorScale = 'lin', doInset=True, xlim=[], ylim=[],clim=[], logScaleFloor=1e-3):
    '''
    This function plots the poloidal projection of a field.

    Inputs:
        simulation: Simulation object.
        fieldName: Name of the field to plot.
        timeFrames: Time frames to plot.
        outFilename: Name of the output file.
        nzInterp: Number of points to interpolate along z.
        scaleFac: Scale factor for the field.
        colorMap: Color map to use.
        doInset: Whether to plot an inset. (not adapted well yet)
        xlim: x-axis limits.
        ylim: y-axis limits.
        clim: Color limits.
    '''
    polproj = PoloidalProjection()

    polproj.setup(simulation, fieldName=fieldName, timeFrame=timeFrame, nzInterp=nzInterp)

    polproj.plot(fieldName=fieldName, timeFrame=timeFrame, colorScale=colorScale,
                 outFilename=outFilename, colorMap=colorMap, doInset=doInset, 
                 scaleFac=scaleFac, xlim=xlim, ylim=ylim, clim=clim, logScaleFloor=logScaleFloor)