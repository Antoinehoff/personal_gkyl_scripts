# pgkyl to load and interpolate gkyl data
import postgkyl as pg
# personnal classes and routines
from .math_utils import *
from .file_utils import *
from classes import Frame
# other commonly used libs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# set the font to be LaTeX
if check_latex_installed(verbose=True):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams.update({'font.size': 14})
default_figsz = [5,3.5]

import os, re

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
    if isinstance(tfs,int):
        tfs = [tfs]
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
                           xlim=[], ylim=[], clim=[], figout=[]):
    cmap0 = cmap
    fields,fig,axs = setup_figure(fieldnames)
    for ax,field in zip(axs,fields):
        x,t,values,xlabel,tlabel,vlabel,vunits,slicetitle,fourrier_y =\
              get_1xt_diagram(simulation,field,cdirection,ccoords,tfs=twindow)
        if space_time:
            if ((field in ['phi','upare','upari']) or cmap0=='bwr') and not fourrier_y:
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
            tfs=[], xlim=[], ylim=[], xscale='', yscale = '', periodicity = 0, grid = False,
            figout = [], errorbar = False):
    
    fields,fig,axs = setup_figure(fieldnames)

    if isinstance(tfs,int) or len(tfs) == 1:
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
                get_1xt_diagram(simulation,subfield,cdirection,ccoords,tfs=tfs)
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

def plot_2D_cut(simulation,cdirection,ccoord,tf,
                fieldnames='', cmap='inferno', full_plot=False,
                xlim=[], ylim=[], clim=[], 
                figout=[],cutout=[]):
    
    # Check if we need to fourier transform
    index = cdirection.find('k')
    if index > -1:
        fourrier_y = True
        cdirection = cdirection.replace('k','')
    else:
        fourrier_y = False

    cmap0 = cmap    
    fields,fig,axs = setup_figure(fieldnames)
    for ax,field in zip(axs,fields):
        frame = Frame(simulation,field,tf)
        frame.load()
        
        if fourrier_y:
            frame.fourrier_y()

        frame.slice_2D(cdirection,ccoord)

        if ((field == 'phi' or field[:-1] == 'upar') or cmap0 == 'bwr')\
            and not fourrier_y:
            cmap = 'bwr'
            vmax = np.max(np.abs(frame.values)) 
            vmin = -vmax
        else:
            cmap = cmap0
            vmax = np.max(frame.values)
            vmin = np.min(frame.values)

        if fourrier_y:
            vmin  = np.power(10,np.log10(vmax)-4)
            frame.values = np.clip(frame.values, vmin, None)  # We plot 4 orders of magnitude
            create_norm = mcolors.LogNorm
        else:
            create_norm = mcolors.Normalize

        YY,XX = np.meshgrid(frame.new_grids[1],frame.new_grids[0])
        norm  = create_norm(vmin=vmin, vmax=vmax)
        pcm = ax.pcolormesh(XX,YY,frame.values,cmap=cmap,norm=norm)
        xlabel = label(frame.new_gsymbols[0],frame.new_gunits[0])
        ylabel = label(frame.new_gsymbols[1],frame.new_gunits[1])
        title  = frame.fulltitle
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if fourrier_y and False:
            title = r'FFT$_y($'+ frame.vsymbol + '), '+ title
        else:
            cbar = fig.colorbar(pcm,label=label(frame.vsymbol,frame.vunits))
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
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
                      fieldnames='', cmap='inferno',
                      xlim=[], ylim=[], clim=[], full_plot=False,
                      fourrier_y=False):
    os.makedirs('gif_tmp', exist_ok=True)
    
    if isinstance(fieldnames,str):
        dataname = fieldnames + '_'
    else:
        dataname = ''
        for f_ in fieldnames:
            dataname += f_+'_'

    for tf in tfs:
        figout = []; cutout = []
        plot_2D_cut(simulation,cdirection,ccoord,tf=tf,fieldnames=fieldnames,
                    cmap=cmap,full_plot=full_plot,
                    xlim=xlim,ylim=ylim,clim=clim,
                    cutout=cutout,figout=figout)
        fig = figout[0]
        fig.tight_layout()
        fig.savefig(f'gif_tmp/plot_{tf}.png')
        plt.close()
    # Naming
    cutout=cutout[0]
    cutname = [key+('=%2.2f'%cutout[key]) for key in cutout]
    moviename = 'movie_'+dataname+cutname[0]
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

def plot_volume_integral_vs_t(simulation, fieldnames, tfs=[], ddt=False,
                              jacob_squared=False, plot_src_input=False,
                              add_GBloss = False, average = False, figout=[], rm_legend=False):
    fields,fig,axs = setup_figure(fieldnames)
    for ax,field in zip(axs,fields):
        if not isinstance(field,list):
            subfields = [field] #simple plot
        else:
            subfields = field # field is a combined plot
        for subfield in subfields:
            ftot_t = []
            time  = []
            for tf in tfs:
                f_ = Frame(simulation=simulation,name=subfield,tf=tf)
                f_.load()

                time.append(f_.time)
                ftot_t.append(f_.compute_volume_integral(jacob_squared=jacob_squared,average=average))

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
                        twindow = tfs,
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
            src_power = simulation.get_input_power()
            if ddt:
                ddtWsrc_t = src_power*np.ones_like(time)/simulation.normalization['Wtotscale']
                ax.plot(time,ddtWsrc_t,'--k',label='Source input (%2.2f MW)'%(src_power/1e6))
            else:
                # plot the accumulate energy from the source
                Wsrc_t = ftot_t[0] + src_power*simulation.normalization['tscale']*time/simulation.normalization['Wtotscale']
                ax.plot(time,Wsrc_t,'--k',label='Source input')
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
            addscale = 1
            if subfield[-1] == 'e':
                spec_s = 'elc'
            elif subfield[-1] == 'i':
                spec_s = 'ion'
            if subfield[:-1] in ['n']:
                comp = 0
            elif subfield[:-1] in ['upar']:
                comp = 1
            elif subfield[:-1] in ['Tpar']:
                comp = 2
                addscale = simulation.phys_param.eV
            elif subfield[:-1] in ['Tperp']:
                comp = 3
                addscale = simulation.phys_param.eV

            f_ = simulation.data_param.fileprefix+'-'+spec_s+'_integrated_moms.gkyl'
            # Load the data from the file
            Gdata = pg.data.GData(f_)
            int_moms = Gdata.get_values()
            time = np.squeeze(Gdata.get_grid()) / simulation.normalization['tscale']
            Ft = np.squeeze(int_moms[:,comp]) / simulation.normalization[subfield+'scale'] * addscale
            Flbl = ddt*r'$\partial_t$ '+simulation.normalization[subfield+'symbol']
            # Plot
            ax.plot(time,Ft,label=Flbl)

        # plot eventually the input power for comparison
        if subfield == 'Wtot' and plot_src_input:
            src_power = simulation.get_input_power()
            if ddt:
                ddtWsrc_t = src_power*np.ones_like(time)/simulation.normalization['Wtotscale']
                ax.plot(time,ddtWsrc_t,'--k',label='Source input (%2.2f MW)'%(src_power/1e6))
            else:
                # plot the accumulate energy from the source
                Wsrc_t = Ft[0] + src_power*simulation.normalization['tscale']*time/simulation.normalization['Wtotscale']
                ax.plot(time,Wsrc_t,'--k',label='Source input')
        # add labels and show legend
        ax.set_xlabel(simulation.normalization['tunits'])
        ax.set_ylabel(simulation.normalization[subfield+'units'])
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend()
        fig.tight_layout()
    figout.append(fig)
    
def label_from_simnorm(simulation,name):
    return label(simulation.normalization[name+'symbol'],simulation.normalization[name+'units'])

def label(label,units):
    if units:
        label += ' ('+units+')'
    return label

def multiply_by_m3_expression(expression):
    
    if expression[-6:]=='/m$^3$':
        expression_new = expression[:-6]
    elif expression[-6:]=='m$^{-3}$':
        expression_new = expression[:-8]
    else:
        expression_new = expression + r'm$^3$'
    return expression_new

def setup_figure(fieldnames):
    if fieldnames == '':
        ncol = 2
        fields = ['ne','upari','Tpari','Tperpi']
    elif not isinstance(fieldnames,list):
        ncol   = 1
        fields = [fieldnames]
    else:
        ncol = 1 * (len(fieldnames) == 1) + 2 * (len(fieldnames) > 1)
        fields = fieldnames
    nrow = len(fields)//ncol + len(fields)%ncol
    fig,axs = plt.subplots(nrow,ncol,figsize=(default_figsz[0]*ncol,default_figsz[1]*nrow))
    if ncol == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    return fields,fig,axs

def get_figdatadict(fig):
    """
    Get data from all curves in a figure
    
    Parameters:
    fig : matplotlib.figure.Figure
        The figure containing the curves.
    """
    figdatadict = []
    # Loop through each Axes in the Figure
    for ax in fig.get_axes():
        a_ = {}
        # axis labels
        a_['xlabel'] = ax.get_xlabel()
        a_['ylabel'] = ax.get_ylabel()
        a_['curves'] = []
        # Write data for each line in the Axes
        for line in ax.get_lines():
            l_ = {}
            l_['xdata'] = line.get_xdata()  # Extract x data
            l_['ydata'] = line.get_ydata()  # Extract y data
            l_['label'] = line.get_label()  # Extract label
            a_['curves'].append(l_)
        figdatadict.append(a_)

    return figdatadict

def plot_figdatadict(figdatadict):
    naxes = len(figdatadict)
    if naxes == 1:
        ncol   = 1
    else:
        ncol = 1 * (naxes == 1) + 2 * (naxes > 1)
    nrow = naxes//ncol + naxes%ncol
    fig,axs = plt.subplots(nrow,ncol,figsize=(default_figsz[0]*ncol,default_figsz[1]*nrow))
    if ncol == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
        
    n_ = 0
    for ax in axs:
        a_ = figdatadict[n_]
        for l_ in a_['curves']:
            ax.plot(l_['xdata'], l_['ydata'], label=l_['label'])
        ax.set_xlabel(a_['xlabel'])
        ax.set_ylabel(a_['ylabel'])
        ax.legend()
        n_ = n_ + 1

def save_figout(figout,fname):
    import pickle
    figdatadict = get_figdatadict(figout[0])
    # Save the dictionary to a JSON file
    with open(fname+'.pkl', 'wb') as f:
        pickle.dump(figdatadict, f)
    print(fname+' saved.')

def load_figout(fname):
    import pickle
    # Load the dictionary from the pickle file
    with open(fname + '.pkl', 'rb') as f:
        figdatadict = pickle.load(f)
    print(fname + ' loaded.')
    return figdatadict

#----- Retrocompatibility
plot_1D_time_avg = plot_1D
