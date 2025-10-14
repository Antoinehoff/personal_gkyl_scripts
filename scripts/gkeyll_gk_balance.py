#[ ........................................................... ]#
#[
#[ Check particle and energy balance in a Gkeyll gyrokinetic simulation.
#[
#[ Adapted from Manaure Francisquez.
#[
#[ ........................................................... ]#
import numpy as np
import postgkyl as pg
import matplotlib.pyplot as plt
import os

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

def does_file_exist(fileIn):
  return os.path.exists(fileIn)

def read_dyn_vector(dataFile):
  pgData = pg.GData(dataFile)
  time   = pgData.get_grid()
  val    = pgData.get_values()
  return np.squeeze(time), np.squeeze(val)

def get_balance_data(data_path, species_names, moment_idx):
    fdot_total = None
    src_total = None
    bflux_total = None
    distf_total = None
    time_fdot = None
    time_distf = None
    time_src = None
    time_bflux = None
    has_source = False
    has_bflux = False

    for sI, species in enumerate(species_names):
        # fdot
        time_fdot_s, fdot_s = read_dyn_vector(f"{data_path}{species}_fdot_integrated_moms.gkyl")
        
        # integrated moms
        time_distf_s, distf_s_all = read_dyn_vector(f"{data_path}{species}_integrated_moms.gkyl")
        distf_s = distf_s_all[:, moment_idx]

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
            distf_total = distf_s
            src_total = src_s
            bflux_total = bflux_tot_s
            time_fdot = time_fdot_s
            time_distf = time_distf_s
            if has_source: time_src = time_src_s
            if has_bflux: time_bflux = time_bflux_list[0]
        else:
            fdot_total += fdot_s[:, moment_idx]
            distf_total += distf_s
            src_total += src_s
            bflux_total += bflux_tot_s

    return (time_fdot, fdot_total, time_distf, distf_total, time_src, src_total, has_source, 
            time_bflux, bflux_total, has_bflux)

def plot_balance(balance_type, simdir, fileprefix, species=['elc', 'ion'], save_fig=False, fig_dir='./', fig_format='.png'):
    moment_idx = 0 if balance_type == 'particle' else 2
    title = 'Particle' if balance_type == 'particle' else 'Energy'
    symbol = 'N' if balance_type == 'particle' else r'\mathcal{E}'

    data_path = f"{simdir}/{fileprefix}-"
    (time_fdot, fdot, _, _, time_src, src, has_source, 
     time_bflux, bflux_tot, has_bflux) = get_balance_data(data_path, species, moment_idx)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.add_axes([0.09, 0.15, 0.87, 0.78])
    
    ax.plot([-1.0,1.0], [0.0,0.0], color='grey', linestyle=':', linewidth=1)

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


    mom_err = src - bflux_tot - (fdot - field_dot)
    
    hpl = []
    legendStrings = []

    hpl.append(ax.plot(time_fdot, fdot, color=defaultColors[0], linestyle=lineStyles[0], linewidth=2, marker=markers[0]))
    legendStrings.append(r'$\dot{f}$')

    if has_source:
        hpl.append(ax.plot(time_src, src, color=defaultColors[2], linestyle=lineStyles[2], linewidth=2, marker=markers[2]))
        legendStrings.append(r'$\mathcal{S}$')

    if has_bflux:
        hpl.append(ax.plot(time_bflux, -bflux_tot, color=defaultColors[1], linestyle=lineStyles[1], linewidth=2, marker=markers[1]))
        legendStrings.append(r'$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$')

    if has_field:
        hpl.append(ax.plot(time_fdot, field_dot, color=defaultColors[4], linestyle=':', linewidth=2, marker='+', markevery=8))
        legendStrings.append(r'$\dot{\phi}$')

    hpl.append(ax.plot(time_fdot, mom_err, color=defaultColors[3], linestyle=lineStyles[3], linewidth=2, marker=markers[3]))
    err_label = r'$E_{\dot{'+symbol+'}}=\mathcal{S}-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f'
    err_label += r'-(\dot{f}-\dot{\phi})$' if balance_type == 'energy' else r'-\dot{f}$'
    legendStrings.append(err_label)

    ax.set_xlabel(r'Time ($s$)', fontsize=xyLabelFontSize, labelpad=+4)
    ax.set_title(f'{title} balance', fontsize=titleFontSize)
    ax.set_xlim(time_fdot[0], time_fdot[-1])
    ax.legend([h[0] for h in hpl], legendStrings, fontsize=legendFontSize, frameon=False, loc='lower right')
    setTickFontSize(ax, tickFontSize)

    if save_fig:
        plt.savefig(f"{fig_dir}{fileprefix}_{balance_type}_balance{fig_format}")
    else:
        plt.show()

def plot_relative_error(balance_type, simdir, fileprefix, species=['elc', 'ion'], save_fig=False, fig_dir='./', fig_format='.png'):
    moment_idx = 0 if balance_type == 'particle' else 2
    symbol = 'N' if balance_type == 'particle' else r'\mathcal{E}'

    data_path = f"{simdir}/{fileprefix}-"
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

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.add_axes([0.12, 0.15, 0.87, 0.77])
    
    ax.plot([-1.0,1.0], [0.0,0.0], color='grey', linestyle=':', linewidth=1)
    ax.semilogy(time_dt, np.abs(mom_err_norm), color=defaultColors[0], linestyle=lineStyles[0], linewidth=2)
    
    ax.set_xlabel(r'Time ($s$)', fontsize=xyLabelFontSize, labelpad=+4)
    ax.set_ylabel(r'$|E_{\dot{'+symbol+'}}~\Delta t/'+symbol+'|$', fontsize=xyLabelFontSize, labelpad=0)
    ax.set_xlim(time_fdot[0], time_fdot[-1])
    setTickFontSize(ax, tickFontSize)

    if save_fig:
        plt.savefig(f"{fig_dir}{fileprefix}_{balance_type}_conservation_rel_error{fig_format}")
    else:
        plt.show()


