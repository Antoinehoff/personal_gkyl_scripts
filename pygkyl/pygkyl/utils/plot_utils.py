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
import os
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from .. utils import file_utils
from ..classes import Frame, IntegratedMoment, TimeSerie
from ..interfaces import pgkyl_interface as pg_int
from ..projections import FluxSurfProjection, PoloidalProjection
from ..tools import fig_tools, math_tools
from ..utils import data_utils

# set the font to be LaTeX
if file_utils.check_latex_installed(verbose=False):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams.update({'font.size': 14})


def _as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def _iter_subfields(field):
    return field if isinstance(field, list) else [field]


def _get_default_field_frame(simulation):
    return simulation.data_param.get_available_frames(simulation)['field'][-1]


def _normalize_clim(clim, nfields):
    if not clim or (isinstance(clim, list) and (len(clim) == 0 or all(not value for value in clim))):
        return [None] * nfields

    if isinstance(clim, (int, float)):
        return [[-clim, clim]] * nfields

    if isinstance(clim, list):
        if len(clim) == 1 and isinstance(clim[0], (int, float)):
            return [[-clim[0], clim[0]]] * nfields

        if len(clim) == 2 and all(isinstance(value, (int, float)) for value in clim):
            return [clim] * nfields

        if all(isinstance(value, list) and len(value) == 2 for value in clim):
            if len(clim) < nfields:
                return clim + [None] * (nfields - len(clim))
            return clim

    return [None] * nfields


def _parse_cut_direction(cut_dir):
    fourier_y = 'k' in cut_dir
    return cut_dir.replace('k', ''), fourier_y


def _format_frame_axis_label(symbol, unit):
    return symbol + (f' [{unit}]' if unit else '')


def _format_time_window_title(slicetitle, tlabel, time_values):
    if time_values[0] == time_values[-1]:
        return slicetitle + tlabel + r'$=%2.2e$' % time_values[0]
    return slicetitle + tlabel + r'$\in[%2.2e,%2.2e]$' % (time_values[0], time_values[-1])


def _apply_fluctuation_label(vsymbol, fluctuation, time_average=False):
    if fluctuation:
        if 'yavg' in fluctuation:
            vsymbol = r'$\delta_y$' + vsymbol
        if 'tavg' in fluctuation:
            vsymbol = r'$\delta_t$' + vsymbol
        return vsymbol

    if time_average:
        return r'$\langle$' + vsymbol + r'$\rangle$'

    return vsymbol


def _apply_relative_percent_label(label):
    label = re.sub(r'\[.*?\]', '', label)
    return label + ' [\%]'


def _load_2d_frame(simulation, field, time_frame, cut_coord, cut_dir, fourier_y, fluctuation, frames_to_plot, field_index):
    if frames_to_plot:
        frame = frames_to_plot[field_index]
        return frame, frame.values

    serie = TimeSerie(
        simulation=simulation,
        fieldname=field,
        time_frames=time_frame,
        load=True,
        fourier_y=fourier_y,
        cut_coord=cut_coord,
        cut_dir=cut_dir,
    )
    if fluctuation:
        serie.fluctuations(fluctuationType=fluctuation)
    frame = serie.frames[-1].copy()
    return frame, frame.values


def _get_2d_color_settings(simulation, field, plot_data, fluctuation, clim_value, cmap, fourier_y, colorscale):
    plot_values = plot_data.copy()
    color_scale = colorscale

    if color_scale == 'log':
        plot_values[plot_values <= 0] = np.nan

    if fluctuation:
        cmap_name = cmap if cmap else 'bwr'
        vmax = np.nanmax(np.abs(plot_values)) if not clim_value else clim_value[1]
        vmin = -vmax if not clim_value else clim_value[0]
    else:
        cmap_name = cmap if cmap else simulation.data_param.field_info_dict[field + 'colormap']
        vmax = np.nanmax(plot_values) if not clim_value else clim_value[1]
        vmin = np.nanmin(plot_values) if not clim_value else clim_value[0]

    if fourier_y:
        vmin = np.power(10, np.log10(vmax) - 3)
        plot_values = np.clip(plot_values, vmin, None)
        color_scale = 'log'

    return plot_values, cmap_name, vmin, vmax, color_scale


def _get_slice_coord_map(simulation_ndim, frame_ndims):
    slice_coord_map = ['x', 'y', 'z', 'vpar', 'mu']

    if simulation_ndim == 5 and frame_ndims <= 3:
        slice_coord_map.remove('vpar')
        slice_coord_map.remove('mu')
    if simulation_ndim == 4:
        slice_coord_map.remove('y')
        if frame_ndims == 2:
            slice_coord_map.remove('vpar')
            slice_coord_map.remove('mu')
    if simulation_ndim == 3:
        slice_coord_map.remove('x')
        slice_coord_map.remove('y')
        if frame_ndims == 1:
            slice_coord_map.remove('vpar')
            slice_coord_map.remove('mu')
    if simulation_ndim == 2:
        slice_coord_map.remove('x')
        slice_coord_map.remove('y')
        slice_coord_map.remove('vpar')
        if frame_ndims == 1:
            slice_coord_map.remove('mu')

    return slice_coord_map


def _normalize_slice_coordinates(simulation, slice_coord_map, slice_coords, fieldname):
    for index, name in enumerate(slice_coord_map):
        coord_name = name + fieldname[-1] if name in ['vpar', 'mu'] else name
        shift = simulation.normalization.dict[coord_name + 'shift']
        scale = simulation.normalization.dict[coord_name + 'scale']
        slice_coords[index] = (slice_coords[index] + shift) * scale


def _trim_integrated_moment_window(int_mom, twindow):
    if not twindow:
        return int_mom

    it0 = np.argmin(abs(int_mom.time - twindow[0]))
    it1 = np.argmin(abs(int_mom.time - twindow[1]))
    int_mom.time = int_mom.time[it0:it1]
    int_mom.values = int_mom.values[it0:it1]
    return int_mom


def _compute_time_series_derivative(simulation, time_values, values, units, label):
    dt0 = time_values[1] - time_values[0]
    values_ext = np.concatenate(([values[0]], values))
    time_ext = np.concatenate(([time_values[0] - dt0], time_values))
    derivative_values = np.gradient(values_ext, time_ext) / simulation.normalization.dict['tscale']

    if units[-1] == '$':
        units = units[:-1] + r'/s$'
    else:
        units = units + r'/s'

    return time_ext, derivative_values, units, r'$\partial_t($' + label + '$)$'


def _load_loss_field(simulation, fieldname):
    try:
        intmom = IntegratedMoment(simulation=simulation, name=fieldname, load=True, ddt=False)
    except KeyError:
        raise ValueError(f"Cannot find field '{fieldname}' in the simulation data.")
    return intmom.values, intmom.time, intmom.vunits, intmom.tunits


def _fl_build_seeds(flp, grids):
    """Pop seed parameters from flp and return (xl0_a, yl0_a) in 3D (x,y) space."""
    xl0_a   = flp.pop('xl0_a', None)
    yl0_a   = flp.pop('yl0_a', None)
    nxl     = flp.pop('nxl',   2)
    nyl     = flp.pop('nyl',   4)
    xlmin_a = flp.pop('xlmin', None)
    xlmax_a = flp.pop('xlmax', None)
    ylmin_a = flp.pop('ylmin', None)
    ylmax_a = flp.pop('ylmax', None)
    if xl0_a is None:
        xlmin = xlmin_a if xlmin_a is not None else grids[0][len(grids[0])//4]
        xlmax = xlmax_a if xlmax_a is not None else grids[0][3*len(grids[0])//4]
        xl0_a = np.linspace(xlmin, xlmax, nxl)
    else:
        xl0_a = np.atleast_1d(xl0_a)
    if yl0_a is None:
        ylmin = ylmin_a if ylmin_a is not None else grids[1][1]
        ylmax = ylmax_a if ylmax_a is not None else grids[1][-2]
        yl0_a = np.linspace(ylmin, ylmax, nyl)
    else:
        yl0_a = np.atleast_1d(yl0_a)
    return xl0_a, yl0_a


def _fl_build_colors(flp, n_lines):
    """Pop color/cmap from flp and return a list of n_lines colours."""
    user_color = flp.pop('color', None)
    user_cmap  = flp.pop('cmap',  None)
    if user_cmap is not None:
        cmap_obj = cm.get_cmap(user_cmap)
        return [cmap_obj(k / max(n_lines - 1, 1)) for k in range(n_lines)]
    if user_color is not None:
        return [user_color] * n_lines
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return [prop_cycle[k % len(prop_cycle)] for k in range(n_lines)]


def _fl_insert_nans(h, v, L_h, L_v):
    """Insert NaN at the union of periodic wrap-around positions in h and v.

    Both arrays receive NaNs at the same indices so they stay the same length.
    """
    break_idx = set()
    if L_h is not None:
        L = L_h[1] - L_h[0]
        break_idx |= set((np.where(np.abs(np.diff(h)) > L / 2)[0] + 1).tolist())
    if L_v is not None:
        L = L_v[1] - L_v[0]
        break_idx |= set((np.where(np.abs(np.diff(v)) > L / 2)[0] + 1).tolist())
    for bi in sorted(break_idx, reverse=True):
        h = np.insert(h, bi, np.nan)
        v = np.insert(v, bi, np.nan)
    return h, v


def _fl_draw(ax, xfl, yfl, zfl, cut_dir, L_periodicity, line_colors, flp):
    """Plot all field lines on ax, inserting NaN breaks at periodic wraps."""
    dir_to_idx = {'x': 0, 'y': 1, 'z': 2}
    coord_map  = {'x': xfl, 'y': yfl, 'z': zfl}
    hcoord = coord_map[cut_dir[0]]
    vcoord = coord_map[cut_dir[1]]
    L_h = L_periodicity[dir_to_idx[cut_dir[0]]]
    L_v = L_periodicity[dir_to_idx[cut_dir[1]]]
    line_idx = 0
    for i in range(xfl.shape[0]):
        for j in range(xfl.shape[1]):
            c = line_colors[line_idx];  line_idx += 1
            h, v = _fl_insert_nans(
                hcoord[i, j].astype(float).copy(),
                vcoord[i, j].astype(float).copy(),
                L_h, L_v,
            )
            ax.plot(h, v, color=c, **flp)
            ax.plot(hcoord[i, j,  0], vcoord[i, j,  0], '.', color=c)
            ax.plot(hcoord[i, j, -1], vcoord[i, j, -1], 'x', color=c)


def plot_1D(simulation,cdirection='x',ccoords=[0.0,0.0,0.0],fieldnames='phi',
            time_frames=None, xlim=[], ylim=[], xscale='', yscale = '', periodicity = 0, grid = False,
            figout = [], errorbar = False, show_title = True, show_legend = True, close_fig = False,
            plot_data = [], figsize=None):
    fields, fig, axs = fig_tools.setup_figure(fieldnames, figsize=figsize)

    if time_frames is None:
        time_frames = _get_default_field_frame(simulation)

    time_avg = not isinstance(time_frames, int) and len(time_frames) != 1

    for ax, field in zip(axs, fields):
        subfields = _iter_subfields(field)
        axis_legend = len(subfields) > 1 and show_legend

        for subfield in subfields:
            x_plot, t, values, xlabel, tlabel, vlabel, vunits, slicetitle, _ = data_utils.get_1xt_diagram(
                simulation,
                subfield,
                cdirection,
                ccoords,
                tfs=time_frames,
            )
            y_plot = np.mean(values, axis=1) if time_avg else values

            if time_avg and errorbar:
                std_dev_data = np.std(values, axis=1)
                ax.errorbar(x_plot, y_plot, yerr=std_dev_data, fmt='o', capsize=5, label=vlabel)
            else:
                ax.plot(x_plot, y_plot, label=vlabel)

            if periodicity > 0:
                ax.plot(x_plot + periodicity, y_plot, label=vlabel)

            plot_data.append((x_plot, y_plot, [xlabel, vlabel, vunits]))

        ylabel = vunits if len(subfields) > 1 else vlabel
        title = _format_time_window_title(slicetitle, tlabel, t)
        fig_tools.finalize_plot(
            ax,
            fig,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title if show_title else '',
            legend=axis_legend,
            figout=figout,
            grid=grid,
            xlim=xlim,
            ylim=ylim,
            xscale=xscale,
            yscale=yscale,
        )
    if close_fig: plt.close(fig)

def plot_2D_cut(simulation, cut_dir='xy', cut_coord=[0.0,0.0,0.0], time_frame=None,
                fieldnames='phi', cmap=None, time_average=False, fluctuation='', plot_type='pcolormesh',
                xlim=[], ylim=[], clim=[], colorscale = 'linear', show_title=True,
                figout=[],cutout=[], val_out=[], frames_to_plot = None, cmap_period=1,
                close_fig=False, aspect='auto', figsize=None, fig_dpi=None,
                quiver_params=None, field_line_params=None):
    """
    quiver_params : dict or list of dict or None
        Optional quiver overlay per subplot.  A single dict is broadcast to all
        subplots; a list (with None entries to skip individual subplots) gives
        per-subplot control.  Two keys are consumed before passing the rest
        straight to ax.quiver():
            fieldname_1 : str -- x-component field name
            fieldname_2 : str -- y-component field name
        All remaining keys (e.g. scale, width, color, alpha, ...) are forwarded
        directly to ax.quiver(). 'color' defaults to 'white' if not provided.
    """
    if time_frame is None:
        time_frame = _get_default_field_frame(simulation)
    if isinstance(fluctuation, bool):
        fluctuation = 'tavg' if fluctuation else ''

    time_frame = _as_list(time_frame)
    fieldnames = _as_list(fieldnames)
    clim = _normalize_clim(clim, len(fieldnames))
    cut_dir, fourier_y = _parse_cut_direction(cut_dir)

    # Normalise quiver_params to a per-subplot list
    nfields = len(fieldnames)
    if quiver_params is None:
        quiver_list = [None] * nfields
    elif isinstance(quiver_params, dict):
        quiver_list = [quiver_params] * nfields
    else:
        quiver_list = list(quiver_params)
        if len(quiver_list) == 1:
            quiver_list = quiver_list * nfields

    # Normalise field_line_params to a per-subplot list
    if field_line_params is None:
        field_line_list = [None] * nfields
    elif isinstance(field_line_params, dict):
        field_line_list = [field_line_params] * nfields
    else:
        field_line_list = list(field_line_params)
        if len(field_line_list) == 1:
            field_line_list = field_line_list * nfields

    fields, fig, axs = fig_tools.setup_figure(fieldnames, figsize=figsize, fig_dpi=fig_dpi)
    for field_index, (ax, field) in enumerate(zip(axs, fields)):
        frame, plot_data = _load_2d_frame(
            simulation,
            field,
            time_frame,
            cut_coord,
            cut_dir,
            fourier_y,
            fluctuation,
            frames_to_plot,
            field_index,
        )
        plot_data, cmap_name, vmin, vmax, color_scale = _get_2d_color_settings(
            simulation,
            field,
            plot_data,
            fluctuation,
            clim[field_index],
            cmap,
            fourier_y,
            colorscale,
        )

        vsymbol = _apply_fluctuation_label(frame.vsymbol, fluctuation, time_average=time_average)
        label = fig_tools.label(vsymbol, frame.vunits)
        if 'relative' in fluctuation:
            label = _apply_relative_percent_label(label)

        xlabel = _format_frame_axis_label(frame.new_gsymbols[0], frame.new_gunits[0])
        ylabel = _format_frame_axis_label(frame.new_gsymbols[1], frame.new_gunits[1])
        fig_tools.plot_2D(
            fig,
            ax,
            x=frame.new_grids[0],
            y=frame.new_grids[1],
            z=plot_data,
            cmap=cmap_name,
            xlim=xlim,
            ylim=ylim,
            clim=clim[field_index],
            xlabel=xlabel,
            ylabel=ylabel,
            cmap_period=cmap_period,
            colorscale=color_scale,
            clabel=label,
            title=frame.fulltitle if show_title else '',
            vmin=vmin,
            vmax=vmax,
            plot_type=plot_type,
            aspect=aspect,
        )

        # Optionally overlay quiver arrows for this subplot
        qp = quiver_list[field_index]
        if qp is not None:
            qp = dict(qp)  # shallow copy so we never mutate the caller's dict
            fn1 = qp.pop('fieldname_1', None)
            fn2 = qp.pop('fieldname_2', None)
            if fn1 is not None and fn2 is not None:
                _, qdata_1 = _load_2d_frame(simulation, fn1, time_frame, cut_coord, cut_dir,
                                            fourier_y, fluctuation, None, 0)
                _, qdata_2 = _load_2d_frame(simulation, fn2, time_frame, cut_coord, cut_dir,
                                            fourier_y, fluctuation, None, 0)
                qp.setdefault('color', 'white')
                max_val = np.nanmax(np.sqrt(qdata_1**2 + qdata_2**2))
                qp.setdefault('scale', 1.25*max_val)  # Default scale based on max vector magnitude
                Y, X = np.meshgrid(frame.new_grids[1], frame.new_grids[0])
                ax.quiver(X, Y, np.squeeze(qdata_1), np.squeeze(qdata_2), **qp)
                
        # Optionally overlay field lines for this subplot
        flp = field_line_list[field_index]
        if flp is not None:
            flp = dict(flp)  # shallow copy to avoid mutating caller's dict
            dBx_fieldname  = flp.pop('dBx_fieldname',  'dB_perp_x')
            dBy_fieldname  = flp.pop('dBy_fieldname',  'dB_perp_y')
            Bmag_fieldname = flp.pop('Bmag_fieldname', 'Bmag')
            periodicity    = flp.pop('periodicity', (False, True, False))
            flp.setdefault('linestyle', '-')
            tf_fl = time_frame[-1]
            dBx_frame  = Frame(simulation, dBx_fieldname,  tf=tf_fl, load=True, normalize=False)
            dBy_frame  = Frame(simulation, dBy_fieldname,  tf=tf_fl, load=True, normalize=False)
            Bmag_frame = Frame(simulation, Bmag_fieldname, tf=tf_fl, load=True, normalize=False)
            grids = dBx_frame.new_grids
            xl0_a, yl0_a = _fl_build_seeds(flp, grids)
            line_colors   = _fl_build_colors(flp, len(xl0_a) * len(yl0_a))
            L_periodicity = [None if not p else (grids[i][0], grids[i][-1])
                             for i, p in enumerate(periodicity)]
            xfl, yfl, zfl = math_tools.field_line_tracer(
                xl0_a, yl0_a, grids[2],
                interp_dBx=dBx_frame.eval_interp,
                interp_dBy=dBy_frame.eval_interp,
                interp_Bmag=Bmag_frame.eval_interp,
                L_periodicity=L_periodicity,
            )
            _fl_draw(ax, xfl, yfl, zfl, cut_dir, L_periodicity, line_colors, flp)

    fig.tight_layout()
    figout.append(fig)
    cutout.append(frame.slicecoords)
    val_out.append(np.squeeze(plot_data))
    if close_fig: plt.close(fig)

def plot_DG_representation(simulation, fieldname='phi', sim_frame=None, cutdir='x', cutcoord=[0.0,0.0,0.0], xlim=[], ylim=[],
                           show_cells=True, figout=[], derivative=False, close_fig=False, dgcoeffidx=None, 
                           figsize=fig_tools.default_figsz, fig_dpi=fig_tools.default_fig_dpi):
    """
    Plot the DG representation of a field along one direction at a given time frame.
    """
    if sim_frame is None:
        sim_frame = simulation.data_param.get_available_frames(simulation)['field'][-1]
    
    # cut the data
    frame = Frame(simulation, fieldname,tf=sim_frame, load=True)
    frame.slice(cutdir, cutcoord)

    slice_coord_map = _get_slice_coord_map(simulation.ndim, frame.ndims)
    if cutdir not in slice_coord_map:
        raise Exception("Invalid cut direction, must be one of %s"%slice_coord_map)

    cutdir_index = slice_coord_map.index(cutdir)
    derivative_index = slice_coord_map.index(derivative) if derivative in ['x','y','z','vpar','mu'] else None
    slice_coord_map.remove(cutdir)

    slice_coords = [frame.slicecoords[key] for key in slice_coord_map]
    field_DG = frame.get_DG_coeff()

    if dgcoeffidx is not None:
        a_new = np.zeros_like(field_DG.values)
        a_new[..., dgcoeffidx] = field_DG.values[..., dgcoeffidx]
        field_DG.values = a_new

    def coord_swap(s, ccoords, direction_index=cutdir_index):
        coords = ccoords.copy()
        coords.insert(direction_index, s)
        return coords

    if cutdir in ['vpar','mu']:
        cutdir += fieldname[-1]

    _normalize_slice_coordinates(simulation, slice_coord_map, slice_coords, fieldname)

    cells = field_DG.grid[cutdir_index]
    DG_proj = []
    s_proj  = []
    sscale = simulation.normalization.dict[cutdir+'scale']
    sshift = simulation.normalization.dict[cutdir+'shift']
    yscale = simulation.normalization.dict[fieldname+'scale']
    dint = 1e-6 # interior of the cell
    for ic in range(len(cells)-1):
        ds = cells[ic+1]-cells[ic]
        si = cells[ic]+dint*ds
        fi = frame.DG_basis.eval_proj(field_DG, coord_swap(si,slice_coords), id=derivative_index)
        DG_proj.append(fi/yscale)
        s_proj.append(si/sscale - sshift)
        sip1 = cells[ic]+(1-dint)*ds
        fip1 = frame.DG_basis.eval_proj(field_DG, coord_swap(sip1,slice_coords), id=derivative_index)
        DG_proj.append(fip1/yscale)
        s_proj.append(sip1/sscale - sshift)
        DG_proj.append(None)
        s_proj.append(cells[ic]/sscale - sshift)

    _, fig, ax = fig_tools.setup_figure(fieldname, figsize=figsize, fig_dpi=fig_dpi)
    ax = ax[0]
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
        if frame.gunits[derivative_index] != '':
            ylabel += '/'+ frame.gunits[derivative_index] + ')'
    title = frame.slicetitle + ' at ' + frame.timetitle
    fig_tools.finalize_plot(ax, fig, xlabel=xlabel, ylabel=ylabel, title=title, figout=figout, xlim=xlim, ylim=ylim)
    
    if close_fig: plt.close(fig)

def poloidal_proj(simulation, fieldName='phi', timeFrame=0, outFilename='',nzInterp=32, polproj=None, cmap_period=1,
                             colorMap = None, colorScale = 'lin', fig_dpi=300, limiterColor='gray', cutoutLimiter=False,
                             showInset=True, showLCFS=True, showVessel=False, showLimiter=True, showAxis=True,
                             xlim=[], ylim=[],clim=[], logScaleFloor=1e-3, figout=[], close_fig=False, figsize=None):
    if timeFrame is None:
        timeFrame = simulation.data_param.get_available_frames(simulation)['field'][-1]
    if polproj is None:
        polproj = PoloidalProjection()
        polproj.setup(simulation, timeFrame=timeFrame, nzInterp=nzInterp)

    polproj.plot(fieldName=fieldName, timeFrame=timeFrame, colorScale=colorScale,
                 outFilename=outFilename, colorMap=colorMap, show_inset=showInset,
                 xlim=xlim, ylim=ylim, clim=clim, logScaleFloor=logScaleFloor, cmap_period=cmap_period,
                 show_LCFS=showLCFS, show_vessel=showVessel, show_limiter=showLimiter, show_axis=showAxis,
                 figout=figout, close_fig=close_fig, cutout_limiter=cutoutLimiter, limiter_color=limiterColor, fig_dpi=fig_dpi, 
                 figsize=figsize)

def flux_surface_proj(simulation, rho=0.9, fieldName='phi', timeFrame=None, Nint=32, figout=[], close_fig=False, clim=[], figsize=None, fig_dpi=150):
    if timeFrame is None:
        timeFrame = simulation.data_param.get_available_frames(simulation)['field'][-1]
    fsproj = FluxSurfProjection()
    fsproj.setup(simulation, rho=rho, timeFrame=timeFrame,
                 Nint=Nint)
    fsproj.plot(fieldName=fieldName, timeFrame=timeFrame, figout=figout, close_fig=close_fig, clim=clim, figsize=figsize, fig_dpi=fig_dpi)
    
#--- Time independent or time series plot routines
  
def plot_1D_time_evolution(simulation, cdirection='x', ccoords=[0.0,0.0,0.0], fieldnames='phi',
                           twindow=None, space_time=False, cmap=None, data_dict={},
                           fluctuation='', plot_type='pcolormesh', yscale='linear',
                           xlim=[], ylim=[], clim=[], figout=[], colorscale='linear',
                           show_title=True, cmap_period=1, close_fig=False,
                           figsize=None):
    fieldnames_list = _as_list(fieldnames)
    if twindow is None:
        twindow = simulation.data_param.get_available_frames(simulation)['field']
        twindow = [twindow[0], twindow[-1]]
    if not isinstance(twindow, list): twindow = [twindow]
    if clim: clim = [clim] if not isinstance(clim[0], list) else clim
    cmap0 = cmap if cmap else simulation.data_param.field_info_dict[fieldnames_list[0]+'colormap']
    fields, fig, axs = fig_tools.setup_figure(fieldnames, figsize=figsize)
    for kf, (ax, field) in enumerate(zip(axs, fields)):
        x, t, values, xlabel, tlabel, vlabel, vunits, slicetitle, fourier_y = \
            data_utils.get_1xt_diagram(simulation, field, cdirection, ccoords, tfs=twindow)
        if fluctuation:
            if 'tavg' in fluctuation:
                average_data = np.mean(values, axis=1)
                vlabel = r'$\delta$' + vlabel
                values -= average_data[:, np.newaxis]
                if 'relative' in fluctuation:
                    values = 100.0 * values / average_data[:, np.newaxis]
                    vlabel = re.sub(r'\(.*?\)', '', vlabel)
                    vlabel = vlabel + ' (\%)'
            else:
                raise ValueError("Fluctuation type '%s' not recognized. Use 'tavg' or 'tavg_relative'." % fluctuation)
        cs = 'log' if fourier_y else colorscale
        if space_time:
            if (cmap0 == 'bwr' or fluctuation) and not fourier_y:
                cmap_name = 'bwr'
                vmax = np.nanmax(np.abs(values))
                vmin = -vmax
            else:
                cmap_name = cmap0
                vmax = np.nanmax(np.abs(values))
                vmin = 0.0
            if cs == 'log' or fourier_y:
                values[values <= 0] = np.nan
                vmax = np.nanmax(values)
                vmin = np.power(10, np.log10(vmax) - 4) if vmax > 0 else 1e-10
                values = np.clip(values, vmin, None)
            clim_ = clim[kf] if clim else None
            fig = fig_tools.plot_2D(
                fig, ax, x=x, y=t, z=values, xlim=xlim, ylim=ylim, clim=clim_,
                xlabel=xlabel, ylabel=tlabel, clabel=vlabel, title=slicetitle if show_title else '',
                cmap=cmap_name, vmin=vmin, vmax=vmax, colorscale=cs, plot_type=plot_type,
                cmap_period=cmap_period
            )
            data_dict[field] = (x, t, values, xlabel, tlabel, vlabel, vunits)
            figout.append(fig)
        else:
            norm = plt.Normalize(min(t), max(t))
            colormap = cm.viridis
            for it in range(len(t)):
                ax.plot(x, values[:, it], label=r'$t=%2.2e$ (ms)' % (t[it]),
                        color=colormap(norm(t[it])))
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            fig_tools.finalize_plot(ax, fig, title=slicetitle[:-2] if show_title else '',
                                    xlim=xlim, ylim=ylim, figout=figout,
                                    xlabel=xlabel, ylabel=vlabel, clabel=tlabel, cbar=cbar, yscale=yscale)
    if close_fig: plt.close(fig)

def plot_domain(simulation, geom_type='Miller', vessel_corners=[[0.6,1.2],[-0.7,0.7]], close_fig=False):
    geometry = simulation.geometry
    geometry.set_domain(geom_type, vessel_corners)
    fig, ax = plt.subplots(figsize=fig_tools.default_figsz)
    ax.plot(geometry.RZ_min[0],  geometry.RZ_min[1],  '-c')
    ax.plot(geometry.RZ_max[0],  geometry.RZ_max[1],  '-c')
    ax.plot(geometry.RZ_lcfs[0], geometry.RZ_lcfs[1], '--k')
    vx1, vx2 = geometry.vessel_corners[0]
    vy1, vy2 = geometry.vessel_corners[1]
    ax.plot([vx1, vx1], [vy1, vy2], '-k')
    ax.plot([vx2, vx2], [vy1, vy2], '-k')
    ax.plot([vx1, vx2], [vy1, vy1], '-k')
    ax.plot([vx1, vx2], [vy2, vy2], '-k')
    ax.plot(geometry.R_axis, geometry.Z_axis, 'x')
    fig_tools.finalize_plot(ax, fig, xlabel='R (m)', ylabel='Z (m)', aspect='equal')
    if close_fig: plt.close(fig)

def plot_integrated_moment(simulation,fieldnames='ne',xlim=[],ylim=[],ddt=False,figout=[],twindow=[],data_dict={},
                           close_fig=False, figsize=None):
    fields,fig,axs = fig_tools.setup_figure(fieldnames, figsize=figsize)
    for ax,field in zip(axs,fields):
        subfields = _iter_subfields(field)
        for subfield in subfields:
            int_mom = IntegratedMoment(simulation=simulation, name=subfield, ddt=ddt)
            int_mom = _trim_integrated_moment_window(int_mom, twindow)
            marker = 'o' if len(int_mom.time) == 1 else None
            ax.plot(int_mom.time,int_mom.values,label=int_mom.symbol,marker=marker)
            data_dict[subfield] = (int_mom.time, int_mom.values, int_mom.tunits, int_mom.vunits, int_mom.symbol)

        fig_tools.finalize_plot(ax, fig, xlabel=int_mom.tunits, ylabel=int_mom.vunits, figout=figout, 
                                xlim=xlim, ylim=ylim, legend=True)
    if close_fig: plt.close(fig)
    return int_mom.time
    
def plot_time_serie(simulation,fieldnames='phi',cut_coords=[0.0,0.0,0.0], time_frames=None,
                    figout=[],xlim=[],ylim=[], ddt = False, data_dict={}, close_fig=False, figsize=None):
    if time_frames is None:
        time_frames = _get_default_field_frame(simulation)
    fields,fig,axs = fig_tools.setup_figure(fieldnames, figsize=figsize)
    for ax,field in zip(axs,fields):
        subfields = _iter_subfields(field)

        for subfield in subfields:
            timeserie = TimeSerie(simulation=simulation,fieldname=subfield,time_frames=time_frames,
                                cut_dir='scalar',cut_coord=cut_coords,load=True)
            f0 = timeserie.frames[0]
            t,v = timeserie.get_values()
            label = f0.vsymbol
            units = f0.vunits
            if ddt:
                t, v, units, label = _compute_time_series_derivative(simulation, t, v, units, label)
            ax.plot(t,v,label=label)
            if data_dict is not None:
                data_dict[subfield] = (t, v, f0.tunits, f0.vunits, f0.vsymbol)

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
                 rm_legend=False, figsize=(8,6), log_abs=False, close_fig=False, data=[], 
                 xlim=None, ylim=None):
    from ..ext.gkeyll_gk_balance import plot_balance
    plot_balance(simulation, balance_type=balance_type, species=species, figout=figout, 
                 rm_legend=rm_legend, figsize=figsize, log_abs=log_abs, data=data, xlim=xlim, ylim=ylim)
    if close_fig: plt.close()
    
def plot_loss(simulation, losstype='energy', walls=[], volfrac_scaled=True, show_avg=True,
              title=True, figout=[], xlim=[], ylim=[], showall=False, legend=True,
              data_out=[], close_fig=False):
    if losstype not in ['particle', 'energy']:
        raise ValueError("Invalid losstype. Choose 'particle' or 'energy'.")
    
    walls = walls if walls else ['x_u','z_l','z_u']
    wall_labels = {'x_l': r'{core}', 'x_u': r'{wall}', 'z_l': r'{lim,low}', 'z_u': r'{lim,up}'}
    symbol = r'\Gamma' if losstype == 'particle' else 'P'
    
    losses = []
    times = []
    npoint_min = None
    for wall in walls:
        fieldname = f'bflux_{wall}' +  ('_ntot' if losstype == 'particle' else '_Htot')
        loss_, time_, vunits, tunits = _load_loss_field(simulation, fieldname)
        if volfrac_scaled: loss_ = loss_ / simulation.geom_param.vol_frac
        npoint_min = len(loss_) if npoint_min is None else min(npoint_min, len(loss_))
        losses.append(loss_)
        times.append(time_)
    
    # Truncate all arrays to the minimum length to handle inhomogeneous shapes
    losses = [loss[:npoint_min] for loss in losses]
    time = times[0][:npoint_min]
    
    # Replace J/s to W or particle by 1
    vunits = vunits.replace('J/s', 'W')
    vunits = vunits.replace('particles', '1')

    total_loss = np.sum(losses, axis=0)    

    nt = len(time)
    loss_avg = np.mean(total_loss[-nt//3:])
        
    fig, ax = plt.subplots(figsize=(fig_tools.default_figsz[0], fig_tools.default_figsz[1]))
    if showall:
        for iw, wall in enumerate(walls):
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
    xlabel = r'$t$ [%s]' % tunits if tunits else r'$t$'
    ylabel = vunits
    title_ = '%s Loss' % losstype.capitalize() if title else ''
        
    fig_tools.finalize_plot(ax, fig, xlabel=xlabel, ylabel=ylabel, figout=figout,
                            title=title_, legend=legend, xlim=xlim, ylim=ylim)
    
    if close_fig: plt.close(fig)

def plot_adapt_src_data(simulation, figout=[], xlim=[], ylim=[], subsrc_labels=[], close_fig=False):
    prefix = simulation.data_param.fileprefix
    files = [
        prefix + '-elc_adapt_sources_particle.gkyl',
        prefix + '-elc_adapt_sources_temperature.gkyl',
        prefix + '-ion_adapt_sources_particle.gkyl',
        prefix + '-ion_adapt_sources_temperature.gkyl',
    ]
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

    fig, axs = plt.subplots(2, 2, figsize=(2*fig_tools.default_figsz[0], 2*fig_tools.default_figsz[1]))
    axs = axs.flatten()

    nsubsources = data[0].shape[1]
    if len(subsrc_labels) != nsubsources:
        subsrc_labels = [r'$S^{%d}$' % (i+1) for i in range(nsubsources)]

    for i in range(4):
        ax = axs[i]
        for j in range(nsubsources):
            ax.plot(time[i]/tscale, data[i][:,j]/scale[i], label=subsrc_labels[j])
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
        frameFileList.append(frameFileName)

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

        print(f"\rProcessing frames: {i}/{total_frames}...", end='', flush=True)

    print()
    
    # Naming

    movieName = movieprefix+'_'+dataname+cutname[0] if movieprefix else dataname+cutname[0]
    movieName+='_xlim_%2.2d_%2.2d'%(xlim[0],xlim[1]) if xlim else ''
    movieName+='_ylim_%2.2d_%2.2d'%(ylim[0],ylim[1]) if ylim else ''

    # Compiling the movie images
    fig_tools.compile_movie(frameFileList, movieName, rmFrames=True)