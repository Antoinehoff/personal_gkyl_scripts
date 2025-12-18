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
- plot_integrated_moment: Plots integrated moments over time for different species.

"""

# personnal classes and routines
from .. utils import file_utils
from ..tools import fig_tools
from ..utils import data_utils
from ..classes import Frame, IntegratedMoment, TimeSerie
from ..projections import PoloidalProjection, FluxSurfProjection
from ..interfaces import pgkyl_interface as pg_int

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
    
def plot_1D(simulation,cdirection='x',ccoords=[0.0,0.0,0.0],fieldnames='phi',
            time_frames=None, xlim=[], ylim=[], xscale='', yscale = '', periodicity = 0, grid = False,
            figout = [], errorbar = False, show_title = True, show_legend = True, close_fig = False):
    
    fields,fig,axs = fig_tools.setup_figure(fieldnames)
    
    if time_frames is None:
        time_frames = simulation.data_param.get_available_frames(simulation)['field'][-1]

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
        show_legend = len(subfields)>1 or show_legend
        title = slicetitle+tlabel+r'$=%2.2e$'%(t[0]) if t[0] == t[-1] else \
                slicetitle+tlabel+r'$\in[%2.2e,%2.2e]$'%(t[0],t[-1])
        fig_tools.finalize_plot(ax, fig, xlabel=xlabel, ylabel=ylabel, title=title if show_title else '', 
                                legend=show_legend, figout=figout, grid=grid, xlim=xlim, ylim=ylim,
                                xscale=xscale, yscale=yscale)
    if close_fig: plt.close(fig)

def plot_2D_cut(simulation, cut_dir='xy', cut_coord=[0.0,0.0,0.0], time_frame=None,
                fieldnames='phi', cmap=None, time_average=False, fluctuation='', plot_type='pcolormesh',
                xlim=[], ylim=[], clim=[], colorscale = 'linear', show_title=True,
                figout=[],cutout=[], val_out=[], frames_to_plot = None, cmap_period=1,
                close_fig=False):
    if time_frame is None:
        time_frame = simulation.data_param.get_available_frames(simulation)['field'][-1]
    if isinstance(fluctuation,bool): fluctuation = 'tavg' if fluctuation else ''
    if isinstance(time_frame, int): time_frame = [time_frame]
    if isinstance(fieldnames, str): fieldnames = [fieldnames]
        
    # Handle any empty clim
    if not clim or (isinstance(clim, list) and (len(clim) == 0 or all(not c for c in clim))):
        clim = [None] * len(fieldnames)
    elif isinstance(clim, (int, float)):
        clim = [[-clim, clim]] * len(fieldnames)
    elif isinstance(clim, list):
        # If single value in list: symmetric
        if len(clim) == 1 and isinstance(clim[0], (int, float)):
            clim = [[-clim[0], clim[0]]] * len(fieldnames)
        # If two numbers: repeat for all fields
        elif len(clim) == 2 and all(isinstance(c, (int, float)) for c in clim):
            clim = [clim] * len(fieldnames)
        # If list of two-element lists: use as is, pad if needed
        elif all(isinstance(c, list) and len(c) == 2 for c in clim):
            if len(clim) < len(fieldnames):
                clim = clim + [None] * (len(fieldnames) - len(clim))
        else:
            clim = [None] * len(fieldnames)

    # Check if we need to fourier transform
    index = cut_dir.find('k')
    if index > -1:
        fourier_y = True
        cut_dir = cut_dir.replace('k','')
    else:
        fourier_y = False
    
    fields,fig,axs = fig_tools.setup_figure(fieldnames)
    kf = 0 # field counter
    for ax,field in zip(axs,fields):
        if frames_to_plot:
            frame = frames_to_plot[kf]
            plot_data = frame.values
        else:
            serie = TimeSerie(simulation=simulation, fieldname=field, 
                              time_frames=time_frame, load=True, fourier_y=fourier_y,
                              cut_coord=cut_coord, cut_dir=cut_dir)
            if fluctuation: serie.fluctuations(fluctuationType=fluctuation)  # Apply fluctuations if needed
            frame = serie.frames[-1].copy()  # Get the last frame
            # frame.slice(cut_dir, cut_coord)
            plot_data = frame.values
        
        if colorscale == 'log':
            plot_data[plot_data <= 0] = np.nan  # Set negative values to nan for log scale
            
        if (fluctuation) :
            cmap = 'bwr' if not cmap else cmap
            vmax = np.nanmax(np.abs(plot_data)) if not clim[kf] else clim[kf][1]
            vmin = -vmax if not clim[kf] else clim[kf][0]
        else:
            cmap = simulation.data_param.field_info_dict[field+'colormap'] if not cmap else cmap
            vmax = np.nanmax(plot_data) if not clim[kf] else clim[kf][1]
            vmin = np.nanmin(plot_data) if not clim[kf] else clim[kf][0]

        if fourier_y:
            vmin  = np.power(10,np.log10(vmax)-3)
            plot_data = np.clip(plot_data, vmin, None)  # We plot 3 orders of magnitude
            colorscale = 'log'

        vsymbol = frame.vsymbol
        if fluctuation:
            if 'yavg' in fluctuation:
                vsymbol = r'$\delta_y$'+ vsymbol
            if 'tavg' in fluctuation:
                vsymbol = r'$\delta_t$'+ vsymbol
        elif time_average:
            vsymbol = r'$\langle$'+ vsymbol + r'$\rangle$'
        lbl = fig_tools.label(vsymbol,frame.vunits)

        xlabel = frame.new_gsymbols[0] + (' [%s]'%frame.new_gunits[0] if frame.new_gunits[0] else '')
        ylabel = frame.new_gsymbols[1] + (' [%s]'%frame.new_gunits[1] if frame.new_gunits[1] else '')

        if "relative" in fluctuation :
            lbl = re.sub(r'\[.*?\]', '', lbl)
            lbl = lbl + ' [\%]'
            
        fig_tools.plot_2D(fig,ax,x=frame.new_grids[0],y=frame.new_grids[1],z=plot_data, 
                          cmap=cmap, xlim=xlim, ylim=ylim, clim=clim[kf],
                          xlabel=xlabel, ylabel=ylabel, cmap_period=cmap_period,
                          colorscale=colorscale, clabel=lbl, title=frame.fulltitle if show_title else '', 
                          vmin=vmin, vmax=vmax, plot_type=plot_type)
        kf += 1 # field counter
    
    fig.tight_layout()
    # This allows to return the figure in the arguments line (used for movies)
    figout.append(fig)
    cutout.append(frame.slicecoords)
    val_out.append(np.squeeze(plot_data))
    if close_fig: plt.close(fig)

def plot_DG_representation(simulation, fieldname='phi', sim_frame=None, cutdir='x', cutcoord=[0.0,0.0,0.0], xlim=[], ylim=[],
                           show_cells=True, figout=[], derivative=False, close_fig=False, dgcoeffidx=None):
    """
    Plot the DG representation of a field along one direction at a given time frame.
    """
    if sim_frame is None:
        sim_frame = simulation.data_param.get_available_frames(simulation)['field'][-1]
    
    # cut the data
    frame = Frame(simulation, fieldname,tf=sim_frame, load=True)
    frame.slice(cutdir, cutcoord)
    
    # process the coordinates of the cut
    slice_coord_map = ['x','y','z','vpar','mu']
    if simulation.ndim == 5: # 3x2v
        if frame.ndims <= 3: # configuration space frame
            slice_coord_map.remove('vpar')
            slice_coord_map.remove('mu')
    if simulation.ndim == 4: #2x2v
        slice_coord_map.remove('y')
        if frame.ndims == 2: # configuration space frame
            slice_coord_map.remove('vpar')
            slice_coord_map.remove('mu')
    if simulation.ndim == 3: #1x2v
        slice_coord_map.remove('x')
        slice_coord_map.remove('y')
        if frame.ndims == 1: # configuration space frame
            slice_coord_map.remove('vpar')
            slice_coord_map.remove('mu')
        
    if cutdir not in slice_coord_map:
        raise Exception("Invalid cut direction, must be one of %s"%slice_coord_map)
    
    dir = slice_coord_map.index(cutdir)
    
    if derivative in ['x','y','z','vpar','mu']:
        id = slice_coord_map.index(derivative)
    else :
        id = None
        
    slice_coord_map.remove(cutdir)
    
    # get the coordinates of the slice
    slice_coords = [frame.slicecoords[key] for key in slice_coord_map]
    field_DG = frame.get_DG_coeff()
    
    if not dgcoeffidx is None:
        # zero out all other coefficients
        a_new = np.zeros_like(field_DG.values)
        a_new[..., dgcoeffidx] = field_DG.values[..., dgcoeffidx]
        field_DG.values = a_new
        
    def coord_swap(s, ccoords, dir=dir):
        """ insert the coordinate s in position dir in the cut coordinate list """
        coords = ccoords.copy()
        coords.insert(dir, s)
        return coords
        
    # add the species identifier to the cutdir if we are in vspace
    if cutdir in ['vpar','mu']:
        cutdir += fieldname[-1]
    
    # get the numerical coordinates
    for i, name in enumerate(slice_coord_map):
        if name in ['vpar','mu']:
            name += fieldname[-1]
        shift = simulation.normalization.dict[name+'shift']
        scale = simulation.normalization.dict[name+'scale']
        slice_coords[i] = (slice_coords[i] + shift) * scale
                    
    cells = field_DG.grid[dir]
    DG_proj = []
    s_proj  = []
    sscale = simulation.normalization.dict[cutdir+'scale']
    sshift = simulation.normalization.dict[cutdir+'shift']
    yscale = simulation.normalization.dict[fieldname+'scale']
    dint = 1e-6 # interior of the cell
    for ic in range(len(cells)-1):
        ds = cells[ic+1]-cells[ic]
        si = cells[ic]+dint*ds
        # left value eval
        fi = frame.DG_basis.eval_proj(field_DG, coord_swap(si,slice_coords), id=id)
        DG_proj.append(fi/yscale)
        s_proj.append(si/sscale - sshift)
        sip1 = cells[ic]+(1-dint)*ds
        # right value eval
        fip1 = frame.DG_basis.eval_proj(field_DG, coord_swap(sip1,slice_coords), id=id)
        DG_proj.append(fip1/yscale)
        s_proj.append(sip1/sscale - sshift)
        # add a None to break the line
        DG_proj.append(None)
        s_proj.append(cells[ic]/sscale - sshift)

    fig = plt.figure(figsize=(fig_tools.default_figsz[0],fig_tools.default_figsz[1]))
    ax = fig.add_subplot(111)
    ax.plot(s_proj, DG_proj,'-')
    # add a vertical line to mark each cell boundary
    if show_cells:
        for sc in cells:
                ax.axvline(sc/sscale - sshift, color='k', linestyle='-', alpha=0.15)

    xlabel = fig_tools.label_from_simnorm(simulation,cutdir)
    ylabel = fig_tools.label_from_simnorm(simulation,fieldname)
    if derivative in ['x','y','z']:
        ylabel.replace(')','')
        ylabel = r'$\partial_%s$'%derivative+ylabel
        if frame.gunits[id] != '':
            ylabel += '/'+ frame.gunits[id] + ')'
    title = frame.slicetitle + ' at ' + frame.timetitle
    fig_tools.finalize_plot(ax, fig, xlabel=xlabel, ylabel=ylabel, title=title, figout=figout, xlim=xlim, ylim=ylim)
    
    if close_fig: plt.close(fig)

def poloidal_proj(simulation, fieldName='phi', timeFrame=0, outFilename='',nzInterp=32,
                             colorMap = 'inferno', colorScale = 'lin', doInset=True, polproj=None,
                             xlim=[], ylim=[],clim=[], logScaleFloor=1e-3, figout=[], close_fig=False):
    if timeFrame is None:
        timeFrame = simulation.data_param.get_available_frames(simulation)['field'][-1]
    if polproj is None:
        polproj = PoloidalProjection()
        polproj.setup(simulation, timeFrame=timeFrame, nzInterp=nzInterp)

    polproj.plot(fieldName=fieldName, timeFrame=timeFrame, colorScale=colorScale,
                 outFilename=outFilename, colorMap=colorMap, show_inset=doInset,
                 xlim=xlim, ylim=ylim, clim=clim, logScaleFloor=logScaleFloor, 
                 figout=figout, close_fig=close_fig)

def flux_surface_proj(simulation, rho=0.9, fieldName='phi', timeFrame=None, Nint=32, figout=[], close_fig=False,
                      clim=[]):
    if timeFrame is None:
        timeFrame = simulation.data_param.get_available_frames(simulation)['field'][-1]
    fsproj = FluxSurfProjection()
    fsproj.setup(simulation, rho=rho, timeFrame=timeFrame,
                 Nint=Nint)
    fsproj.plot(fieldName=fieldName, timeFrame=timeFrame, figout=figout, close_fig=close_fig, clim=clim)
    
#--- Time independent or time series plot routines
  
def plot_1D_time_evolution(simulation, cdirection='x', ccoords=[0.0,0.0,0.0], fieldnames='phi',
                           twindow=None, space_time=False, cmap='inferno',
                           fluctuation='', plot_type='pcolormesh', yscale='linear',
                           xlim=[], ylim=[], clim=[], figout=[], colorscale='linear',
                           show_title=True, cmap_period=1, close_fig=False):
    if twindow is None:
        twindow = simulation.data_param.get_available_frames(simulation)['field']
        twindow = [twindow[0], twindow[-1]]
    if not isinstance(twindow, list): twindow = [twindow]
    if clim: clim = [clim] if not isinstance(clim[0], list) else clim
    cmap0 = cmap
    fields, fig, axs = fig_tools.setup_figure(fieldnames)
    kf = 0  # field counter
    for ax, field in zip(axs, fields):
        x, t, values, xlabel, tlabel, vlabel, vunits, slicetitle, fourier_y = \
            data_utils.get_1xt_diagram(simulation, field, cdirection, ccoords, tfs=twindow)
        if len(fluctuation) > 0:
            if 'tavg' in fluctuation:
                average_data = np.mean(values, axis=1)
                vlabel = r'$\delta$' + vlabel
                for it in range(len(t)) :
                    values[:,it] = (values[:,it] - average_data[:])
                if 'relative' in fluctuation:
                    values = 100.0*values/average_data
                    vlabel = re.sub(r'\(.*?\)', '', vlabel)
                    vlabel = vlabel + ' (\%)'
            else:
                raise ValueError("Fluctuation type '%s' not recognized. Use 'tavg' or 'tavg_relative'." % fluctuation)
        # handle fourier plot
        # Use the colorscale argument, override to 'log' if fourier_y
        cs = 'log' if fourier_y else colorscale
        if space_time:
            if ((simulation.data_param.field_info_dict[field + 'colormap'] == 'bwr' or cmap0 == 'bwr' or len(fluctuation) > 0)
                and not fourier_y):
                cmap = 'bwr'
                vmax = np.nanmax(np.abs(values))
                vmin = -vmax
            else:
                cmap = cmap0
                vmax = np.nanmax(np.abs(values))
                vmin = 0.0
            if cs == 'log' or fourier_y:
                # set to nan negative values
                values[values <= 0] = np.nan
                vmax = np.nanmax(values)
                # For log scale, clip values and set vmin > 0
                vmin = np.power(10, np.log10(vmax) - 4) if vmax > 0 else 1e-10
                values = np.clip(values, vmin, None)
            clim_ = clim[kf] if clim else None
            fig = fig_tools.plot_2D(
                fig, ax, x=x, y=t, z=values, xlim=xlim, ylim=ylim, clim=clim_,
                xlabel=xlabel, ylabel=tlabel, clabel=vlabel, title=slicetitle,
                cmap=cmap, vmin=vmin, vmax=vmax, colorscale=cs, plot_type=plot_type,
                cmap_period=cmap_period
            )
            figout.append(fig)
        else:
            norm = plt.Normalize(min(t), max(t))
            colormap = cm.viridis  # You can choose any colormap
            for it in range(len(t)):
                ax.plot(x, values[:, it], label=r'$t=%2.2e$ (ms)' % (t[it]),
                        color=colormap(norm(t[it])))
            # Add a colorbar to the figure
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            fig_tools.finalize_plot(ax, fig, title=slicetitle[:-2] if show_title else '', 
                                    xlim=xlim, ylim=ylim, figout=figout,
                                    xlabel=xlabel, ylabel=vlabel, clabel=tlabel, cbar=cbar, yscale=yscale)
        kf += 1  # field counter
    if close_fig: plt.close(fig)

def plot_domain(simulation,geom_type='Miller',vessel_corners=[[0.6,1.2],[-0.7,0.7]], close_fig=False):
    geometry = simulation.geometry
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
    if close_fig: plt.close(fig)

def plot_integrated_moment(simulation,fieldnames='ne',xlim=[],ylim=[],ddt=False,figout=[],twindow=[],data=[],
                           close_fig=False):
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
            data.append((int_mom.time,int_mom.values))
        # add labels and show legend
        fig_tools.finalize_plot(ax, fig, xlabel=int_mom.tunits, ylabel=int_mom.vunits, figout=figout, 
                                xlim=xlim, ylim=ylim, legend=True)
    if close_fig: plt.close(fig)
    return int_mom.time
    
def plot_time_serie(simulation,fieldnames='phi',cut_coords=[0.0,0.0,0.0], time_frames=None,
                    figout=[],xlim=[],ylim=[], ddt = False, data=None, close_fig=False):
    if time_frames is None:
        time_frames = simulation.data_param.get_available_frames(simulation)['field'][-1]
    fields,fig,axs = fig_tools.setup_figure(fieldnames)
    for ax,field in zip(axs,fields):
        if not isinstance(field, list):
            subfields = [field]  # Simple plot
        else:
            subfields = field  # Field is a combined plot

        for subfield in subfields:
            timeserie = TimeSerie(simulation=simulation,fieldname=subfield,time_frames=time_frames,
                                cut_dir='scalar',cut_coord=cut_coords,load=True)
            f0 = timeserie.frames[0]
            t,v = timeserie.get_values()
            label = f0.vsymbol
            units = f0.vunits
            if ddt:
                dt0 = t[1] - t[0]
                v_ext = np.concatenate(([v[0]], v))
                t_ext = np.concatenate(([t[0]-dt0], t))
                dvdt_ext = np.gradient(v_ext, t_ext)/simulation.normalization.dict['tscale']
                v = dvdt_ext
                t = t_ext
                label = r'$\partial_t($' + label + '$)$'
                if units[-1] == '$':
                    units = units[:-1] + r'/s$'
                else:
                    units = units + r'/s'
            ax.plot(t,v,label=label)
            if data is not None:
                data.append((t,v))
        
        # units = math_utils.simplify_units(units) # this is not robuts...
        fig_tools.finalize_plot(ax, fig, xlabel=f0.tunits, ylabel=units, figout=figout,
                                xlim=xlim, ylim=ylim, legend=True, title=f0.slicetitle)
    if close_fig: plt.close(fig)
        
def plot_nodes(simulation, close_fig=False):
    from matplotlib.collections import LineCollection
    import postgkyl as pg
    simName = simulation.data_param.fileprefix
    plt.figure()
    data = pg.GData(simName+"-nodes.gkyl")
    vals = data.get_values()
    X = vals[:,0,:,0]
    Y = vals[:,0,:,1]
    Z = vals[:,0,:,2]
    R=np.sqrt(X**2+Y**2)

    plt.plot(R,Z,marker=".", color="k", linestyle="none")
    plt.scatter(R,Z, marker=".")
    segs1 = np.stack((R,Z), axis=2)
    segs2 = segs1.transpose(1,0,2)
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))
    plt.grid()

    plt.axis('equal')
    plt.show()
    if close_fig: plt.close()

def plot_balance(simulation, balance_type='particle', species=['elc', 'ion'], figout=[], 
                 rm_legend=False, fig_size=(8,6), log_abs=False, close_fig=False):
    from ..ext.gkeyll_gk_balance import plot_balance
    plot_balance(simulation, balance_type=balance_type, species=species, figout=figout, 
                 rm_legend=rm_legend, fig_size=fig_size, log_abs=log_abs)
    if close_fig: plt.close()
    
def plot_loss(simulation, losstype='energy', walls =[], volfrac_scaled=True, show_avg=True,
              title=True, figout=[], xlim=[], ylim=[], showall=False, legend=True,
              data_out = [], close_fig=False):
    def get_int_mom_data(simulation, fieldname):
        try:
            intmom = IntegratedMoment(simulation=simulation, name=fieldname, load=True, ddt=False)
        except KeyError:
            raise ValueError(f"Cannot find field '{fieldname}' in the simulation data. ")
        return intmom.values, intmom.time, intmom.vunits, intmom.tunits
    
    if losstype not in ['particle', 'energy']:
        raise ValueError("Invalid losstype. Choose 'particle' or 'energy'.")
    
    walls = walls if walls else ['x_u','z_l','z_u']
    wall_labels = {'x_l': r'{core}', 'x_u': r'{wall}', 'z_l': r'{lim,low}', 'z_u': r'{lim,up}'}
    symbol = r'\Gamma' if losstype == 'particle' else 'P'
    
    losses = []
    for wall in walls:
        fieldname = f'bflux_{wall}' +  ('_ntot' if losstype == 'particle' else '_Htot')
        loss_, time, vunits, tunits = get_int_mom_data(simulation, fieldname)
        if volfrac_scaled: loss_ = loss_ / simulation.geom_param.vol_frac
        losses.append(loss_)
    
    # Replace J/s to W or particle by 1
    vunits = vunits.replace('J/s', 'W')
    vunits = vunits.replace('particles', '1')

    total_loss = np.sum(losses, axis=0)    

    nt = len(time)
    loss_avg = np.mean(total_loss[-nt//3:])
        
    fig, ax = plt.subplots(figsize=(fig_tools.default_figsz[0], fig_tools.default_figsz[1]))
    if showall:
        for iw,wall in zip(range(len(walls)), walls):
            ax.plot(time, losses[iw], label=r'$%s_{%s}$'%(symbol,wall_labels[wall]))
            data_out.append((time, losses[iw], r'$%s_{%s}$'%(symbol,wall_labels[wall])))
    labeltot = r'$%s_{SOL}$'%symbol if walls == ['x_u','z_l','z_u'] else r'$%s_{tot}$'%symbol
    ax.plot(time, total_loss, label=labeltot)
    data_out.append((time, total_loss, labeltot))
    if show_avg:
        # Add horizontal line at average balance value
        ax.plot([time[-nt//3], time[-1]], [loss_avg, loss_avg],
                '--k', alpha=0.5, label='%s %s' % (fig_tools.optimize_str_format(loss_avg), vunits))
        data_out.append((time, [loss_avg]*len(time), '%s %s' % (fig_tools.optimize_str_format(loss_avg), vunits)))
    xlabel = r'$t$ [%s]' % tunits if  tunits else r'$t$'
    ylabel = vunits
    title_ = f'%s Loss' % losstype.capitalize() if  title else ''
        
    fig_tools.finalize_plot(ax, fig, xlabel=xlabel, ylabel=ylabel, figout=figout,
                            title=title_, legend=legend, xlim=xlim, ylim=ylim)
    
    if close_fig: plt.close(fig)

def plot_adapt_src_data(simulation, figout=[], xlim=[], ylim=[], subsrc_labels=[], close_fig=False):
    elc_src_part_filename = simulation.data_param.fileprefix+'-elc_adapt_sources_particle.gkyl'
    elc_src_temp_filename = simulation.data_param.fileprefix+'-elc_adapt_sources_temperature.gkyl'
    ion_src_part_filename = simulation.data_param.fileprefix+'-ion_adapt_sources_particle.gkyl'
    ion_src_temp_filename = simulation.data_param.fileprefix+'-ion_adapt_sources_temperature.gkyl'
    files = [elc_src_part_filename, elc_src_temp_filename,
             ion_src_part_filename, ion_src_temp_filename]
    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Required file '{f}' not found.")
        
    ylabels = [r'$\Gamma_{src,e}$ [1/s]', r'$T_{src,e}$ [eV]', r'$\Gamma_{src,i}$ [1/s]', r'$T_{src,i}$ [eV]']
    scale = [1.0, 1.609e-19, 1.0, 1.609e-19]
    tscale = simulation.normalization.dict['tscale']

    data = []
    time = []
    for f in files:
        Gdata = pg_int.get_gkyl_data(f)
        data.append(np.squeeze(pg_int.get_values(Gdata)))
        time.append(np.squeeze(Gdata.get_grid()))
        
    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(2*fig_tools.default_figsz[0], 2*fig_tools.default_figsz[1]))
    axs = axs.flatten()
    
    nsubsources = data[0].shape[1]
    if not len(subsrc_labels) == nsubsources:
        subsrc_labels = [r'$S^{%d}$' % (i+1) for i in range(nsubsources)]
    
    for i in range(4):
        ax = axs[i]
        for j in range(nsubsources):
            ax.plot(time[i]/tscale, data[i][:,j]/scale[i], label=subsrc_labels[j])
        # ax.set_ylabel(ylabels[i])
        # ax.set_xlabel(simulation.normalization.dict['tunits'])
        
        fig_tools.finalize_plot(ax, fig, xlabel=simulation.normalization.dict['tunits'], 
                                ylabel=ylabels[i], figout=[], xlim=xlim, ylim=ylim, legend=True)
    figout.append(fig)
    if close_fig: plt.close(fig)

def make_2D_movie(simulation, cut_dir='xy', cut_coord=0.0, time_frames=None, fieldnames=['phi'],
                  cmap=None, xlim=[], ylim=[], clim=None, fluctuation = '',
                  movieprefix='', plot_type='pcolormesh', fourier_y=False,colorScale='lin'):
    if time_frames is None:
        time_frames = simulation.data_param.get_available_frames(simulation)['field'][-2:]
    # Create a temporary folder to store the movie frames
    movDirTmp = 'movie_frames_tmp'
    os.makedirs(movDirTmp, exist_ok=True)
    
    if isinstance(fieldnames,str):
        dataname = fieldnames + '_'
        fieldnames = [fieldnames]
    else:
        dataname = ''
        for f_ in fieldnames:
            dataname += 'd'+f_+'_' if len(fluctuation)>0 else f_+'_'

    movie_frames, vlims = data_utils.get_2D_movie_time_serie(
        simulation, cut_dir, cut_coord, time_frames, fieldnames, fluctuation, fourier_y) 
    
    if clim == 'free':
        clim = None
    else:
        clim = clim if clim else vlims
        
    total_frames = len(time_frames)
    frameFileList = []
        
    for i, tf in enumerate(time_frames, 1):  # Start the index at 1  

        frameFileName = f'movie_frames_tmp/frame_{tf}.png'
        frameFileList.append(f'movie_frames_tmp/frame_{tf}.png')

        figout = []
        cutout = []

        plot_2D_cut(
            simulation, cut_dir=cut_dir, cut_coord=cut_coord, time_frame=tf, fieldnames=fieldnames,
            cmap=cmap, plot_type=plot_type, colorscale=colorScale,
            xlim=xlim, ylim=ylim, clim=clim, fluctuation=fluctuation,
            cutout=cutout, figout=figout, frames_to_plot=movie_frames[i-1]
        )
        fig = figout[0]
        fig.tight_layout()
        fig.savefig(frameFileName)
        plt.close()
        cutout=cutout[0]
        cutname = []
        for key in cutout:
            if isinstance(cutout[key], float):
                cutname.append(key+('=%2.2f'%cutout[key]))
            elif isinstance(cutout[key], str):
                cutname.append(key+cutout[key])

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
 
def make_movie(plot_function, frames, **kwargs):
    return