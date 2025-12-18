#[ ........................................................... ]#
#[
#[ Check particle and energy balance in a Gkeyll gyrokinetic simulation.
#[
#[ Adapted from Manaure Francisquez.
#[
#[ ........................................................... ]#
import numpy as np
import matplotlib.pyplot as plt
import os
from ..tools import fig_tools
from ..utils.file_utils import does_file_exist
from ..interfaces.pgkyl_interface import read_dyn_vector

# Some RGB colors. These are MATLAB-like.
defaultBlue    = [0, 0.4470, 0.7410]
defaultOrange  = [0.8500, 0.3250, 0.0980]
defaultGreen   = [0.4660, 0.6740, 0.1880]
defaultPurple  = [0.4940, 0.1840, 0.5560]
defaultRed     = [0.6350, 0.0780, 0.1840]
defaultSkyBlue = [0.3010, 0.7450, 0.9330]
grey           = [0.5, 0.5, 0.5]
defaultColors = [defaultBlue,defaultOrange,defaultGreen,defaultPurple,defaultRed,defaultSkyBlue,grey,'black']

# LineStyles in a single array.
lineStyles = ['-','--',':','-.','None','None','None','None']
markers    = ['None','None','None','None','o','d','s','+']

# Some fontsizes used in plots.
xyLabelFontSize       = 17
titleFontSize         = 17
colorBarLabelFontSize = 17
tickFontSize          = 14
legendFontSize        = 14

def setTickFontSize(axIn,fontSizeIn):
  axIn.tick_params(axis='both',labelsize=fontSizeIn)
  offset_txt = axIn.yaxis.get_offset_text()
  if offset_txt: offset_txt.set_size(fontSizeIn)
  offset_txt = axIn.xaxis.get_offset_text()
  if offset_txt: offset_txt.set_size(fontSizeIn)

def get_balance_data(data_path, species_names, moment_idx):
    fdot_total = None
    src_total = None
    bflux_total = None
    intmoms_total = None
    time_fdot = None
    time_intmoms = None
    time_src = None
    time_bflux = None
    has_fdot = False
    has_intmoms = False
    has_source = False
    has_bflux = False
    if not isinstance(species_names, list):
        species_names = [species_names]
    for sI, species in enumerate(species_names):
        # fdot
        fdot_file = f"{data_path}{species}_fdot_integrated_moms.gkyl"
        if does_file_exist(fdot_file):
            has_fdot = True
            time_fdot_s, fdot_s = read_dyn_vector(fdot_file)
        else:
            fdot_s = np.zeros((1, 1))  # Placeholder for missing data
            time_fdot_s = np.zeros((1, 1))  # Placeholder for missing time data
        
        # integrated moms
        intmoms_file = f"{data_path}{species}_integrated_moms.gkyl"
        if does_file_exist(intmoms_file):
            has_intmoms = True
            time_intmoms_s, intmoms_all = read_dyn_vector(intmoms_file)
            intmoms_s = intmoms_all[:, moment_idx]
        else:
            intmoms_s = np.zeros_like(fdot_s[:, moment_idx])
            time_intmoms_s = np.zeros_like(time_fdot_s)

        # source
        source_file = f"{data_path}{species}_source_integrated_moms.gkyl"
        if does_file_exist(source_file):
            has_source = True
            time_src_s, src_s_all = read_dyn_vector(source_file)
            src_s = src_s_all[:, moment_idx]
            src_s[0] = 0.0
        else:
            src_s = np.zeros_like(fdot_s[:, moment_idx])

        # boundary flux
        nbflux = 0
        bflux_s_list = []
        time_bflux_list = []
        for d in ["x", "y", "z"]:
            for e in ["lower", "upper"]:
                bflux_file = f"{data_path}{species}_bflux_{d}{e}_integrated_HamiltonianMoments.gkyl"
                if does_file_exist(bflux_file):
                    has_bflux = True
                    time_bflux_tmp, bflux_tmp_all = read_dyn_vector(bflux_file)
                    bflux_s_list.append(bflux_tmp_all[:, moment_idx])
                    time_bflux_list.append(time_bflux_tmp)
                    nbflux += 1
        
        bflux_tot_s = np.sum(bflux_s_list, axis=0) if nbflux > 0 else np.zeros_like(fdot_s[:, moment_idx])

        if sI == 0:
            fdot_total = fdot_s[:, moment_idx]
            intmoms_total = intmoms_s
            src_total = src_s
            bflux_total = bflux_tot_s
            time_fdot = time_fdot_s
            time_intmoms = time_intmoms_s
            if has_source: time_src = time_src_s
            if has_bflux: time_bflux = time_bflux_list[0]
        else:
            fdot_total += fdot_s[:, moment_idx]
            intmoms_total += intmoms_s
            src_total += src_s
            bflux_total += bflux_tot_s

    return (time_fdot, fdot_total, has_fdot, time_intmoms, intmoms_total, has_intmoms, time_src, src_total, has_source, 
            time_bflux, bflux_total, has_bflux)

def plot_balance(simulation, balance_type='particle', species=['elc', 'ion'], figout=[], 
                 rm_legend=False, fig_size=(8,6), log_abs=False):
    moment_idx = 0 if balance_type == 'particle' else 2
    symbol = 'N' if balance_type == 'particle' else r'\mathcal{E}'
     
    data_path = f"{simulation.data_param.fileprefix}-"
    (time_fdot, fdot, has_fdot, _, _, _, time_src, src, has_source, 
     time_bflux, bflux_tot, has_bflux) = get_balance_data(data_path, species, moment_idx)

    field, fig, axs = fig_tools.setup_figure(balance_type,fig_size=fig_size)
    ax = axs[0]
    def fp(x): return np.abs(x) if log_abs else x
       
    # Field energy for energy balance
    field_dot = np.zeros_like(fdot)
    has_field = False
    if balance_type == 'energy':
        field_file = f"{data_path}field_energy_dot.gkyl"
        if does_file_exist(field_file):
            has_field = True
            time_field_dot, field_dot_raw = read_dyn_vector(field_file)
            # Ensure field_dot has same shape as fdot
            field_dot = np.interp(time_fdot, time_field_dot, field_dot_raw)
            
    # EM energy for energy balance
    apardot = np.zeros_like(fdot)
    has_apardot = False
    if balance_type == 'energy':
        apardot_file = f"{data_path}apar_energy_dot.gkyl"
        if does_file_exist(apardot_file):
            has_apardot = True
            time_apardot, apardot_raw = read_dyn_vector(apardot_file)
            apardot = np.interp(time_fdot, time_apardot, apardot_raw)

    mom_err = 0.0
    if has_source : 
        mom_err += src
        time_err = time_src
    if has_bflux: 
        mom_err -= bflux_tot
        time_err = time_bflux
    if has_fdot: 
        mom_err -= fdot
        time_err = time_fdot
    if has_field: 
        mom_err += field_dot
        time_err = time_field_dot
    if has_apardot: 
        mom_err += apardot
        time_err = time_apardot
    
    legendStrings = []
    if has_fdot:
        lbl = r'$\dot{f}$'
        ax.plot(time_fdot, fp(fdot), label=lbl)
        legendStrings.append(lbl)

    if has_source:
        lbl = r'$\mathcal{S}$'
        ax.plot(time_src, fp(src), label=lbl)
        legendStrings.append(lbl)

    if has_bflux:
        lbl = r'$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$'
        ax.plot(time_bflux, fp(-bflux_tot), label=lbl)
        legendStrings.append(lbl)

    if has_field:
        lbl = r'$\dot{\phi}$'
        ax.plot(time_fdot, fp(field_dot), label=lbl)
        legendStrings.append(lbl)
        
    if has_apardot:
        lbl = r'$\partial_t A_\parallel$'
        ax.plot(time_fdot, fp(apardot), label=lbl)
        legendStrings.append(lbl)

    lbl = r'$E_{\dot{'+symbol+'}}=\mathcal{S}-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f'
    lbl += r'-\dot{f}'
    if balance_type == 'energy':
        if has_field:
            lbl += r'+\dot{\phi}' 
        if has_apardot:
            lbl += r'-\partial_t A_\parallel'
    lbl += r'$'
    
    ax.plot(time_err, fp(mom_err), label=lbl)
    legendStrings.append(lbl)

    xlbl = r'Time ($s$)'
    ylbl = ''
    
    # add labels and show legend
    fig_tools.finalize_plot(ax, fig, xlabel=xlbl, ylabel=ylbl, figout=figout, 
                            legend=not rm_legend, yscale='log' if log_abs else 'linear')

def plot_relative_error(simulation, balance_type='particle', species=['elc', 'ion'], figout=[], 
                        rm_legend=True, data =[], show_plot=True):
    moment_idx = 0 if balance_type == 'particle' else 2
    symbol = 'N' if balance_type == 'particle' else r'\mathcal{E}'

    data_path = f"{simulation.data_param.fileprefix}-"
    (time_fdot, fdot, time_distf, distf, time_src, src, has_source, 
     time_bflux, bflux_tot, has_bflux) = get_balance_data(data_path, species, moment_idx)

    # Remove t=0 data point
    fdot = fdot[1:]
    distf = distf[1:]
    src = src[1:] if has_source else np.zeros_like(fdot)
    bflux_tot = bflux_tot[1:] if has_bflux else np.zeros_like(fdot)
    
    time_dt, dt = read_dyn_vector(f"{data_path}dt.gkyl")

    # Field energy for energy balance
    field_dot = np.zeros_like(fdot)
    field = np.zeros_like(distf)
    if balance_type == 'energy':
        if does_file_exist(f"{data_path}field_energy_dot.gkyl"):
            time_field_dot, field_dot_raw = read_dyn_vector(f"{data_path}field_energy_dot.gkyl")
            field_dot = np.interp(time_dt, time_field_dot[1:], field_dot_raw[1:])
        if does_file_exist(f"{data_path}field_energy.gkyl"):
            time_field, field_raw = read_dyn_vector(f"{data_path}field_energy.gkyl")
            field = np.interp(time_distf, time_field[1:], field_raw[1:])

    mom_err = src - bflux_tot - (fdot - field_dot)
    denominator = distf - field if balance_type == 'energy' else distf
    mom_err_norm = mom_err * dt / denominator

    field, fig, axs = fig_tools.setup_figure(balance_type)
    ax = axs[0]
    
    ax.semilogy(time_dt, np.abs(mom_err_norm), color=defaultColors[0], linestyle=lineStyles[0], linewidth=2)
    
    xlbl = r'Time ($s$)'
    ylbl = r'$|E_{\dot{'+symbol+'}}~\Delta t/'+symbol+'|$'
    ax.set_xlim(time_fdot[0], time_fdot[-1])
    setTickFontSize(ax, tickFontSize)

    if show_plot:
        # add labels and show legend
        fig_tools.finalize_plot(ax, fig, xlabel=xlbl, ylabel=ylbl, figout=figout, legend=not rm_legend)
    else:
        plt.close(fig)
    
    data.append(time_dt)
    data.append(np.abs(mom_err_norm))
