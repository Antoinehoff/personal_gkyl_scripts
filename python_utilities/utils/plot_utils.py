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

# set the font to be LaTeX
if check_latex_installed(verbose=True):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']

import matplotlib.cm as cm
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

def get_1xt_diagram(simulation, fieldname, cutdirection, ccoords,tfs ):
    # to store iteratively times and values
    t  = []
    values = []
    if isinstance(tfs,int):
        tfs = [tfs]
    # Fill ZZ with data for each time frame
    for it, tf in enumerate(tfs):
        frame = Frame(simulation,fieldname,tf)
        frame.load()
        frame.slice_1D(cutdirection,ccoords)
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
    return x,t,values,xlabel,tlabel,vlabel,frame.vunits,slicetitle
    return {'x':x,'t':t,'values':values,'name':frame.name,
            'xsymbol':frame.new_gsymbols[0], 'xunits':frame.new_gunits[0], 
            'vsymbol':frame.vsymbol, 'vunits':frame.vunits, 'slicetitle':frame.slicetitle,
            'slicecoords':frame.slicecoords, 'fulltitle':frame.fulltitle}
    
def plot_1D_time_evolution(simulation,cdirection,ccoords,fieldnames='',
                           twindow=[],space_time=False, cmap='inferno',
                           xlim=[], ylim=[], clim=[], time_avg=False):
    cmap0 = cmap
    fields,fig,axs = setup_figure(fieldnames)
    for ax,field in zip(axs,fields):
        x,t,values,xlabel,tlabel,vlabel,vunits,slicetitle =\
              get_1xt_diagram(simulation,field,cdirection,ccoords,tfs=twindow)
        if space_time:
            if (field in ['phi','upare','upari']) or cmap0=='bwr':
                cmap = 'bwr'
                vmax = np.max(np.abs(values)) 
                vmin = -vmax
            else:
                cmap = cmap0
                vmax = np.max(np.abs(values)) 
                vmin = 0.0
            XX, TT = np.meshgrid(x,t)
            # Create a contour plot or a heatmap of the space-time diagram
            pcm = ax.pcolormesh(XX,TT,values,cmap=cmap,vmin=vmin,vmax=vmax); 
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
        if len(axs) > 1:
            fig.suptitle(slicetitle[:-2])
        else:
            ax.set_title(slicetitle[:-2])
    fig.tight_layout()

def plot_1D_time_avg(simulation,cdirection,ccoords,fieldnames='',
                           tfs=[], xlim=[], ylim=[], multi_species = True):
    fields,fig,axs = setup_figure(fieldnames)
    print('the function plot_1D_time_avg is now depreciated, use plot_1D instead')
    for ax,field in zip(axs,fields):
        if not isinstance(field,list):
            subfields = [field] #simple plot
        else:
            subfields = field # field is a combined plot
        for subfield in subfields:
            x,t,values,xlabel,tlabel,vlabel,vunits,slicetitle =\
                get_1xt_diagram(simulation,subfield,cdirection,ccoords,tfs=tfs)
            # Compute the average of data over the t-axis (axis=1)
            average_data = np.mean(values, axis=0)
            # Compute the standard deviation of data over the t-axis (axis=1)
            std_dev_data = np.std(values, axis=0)
            # Plot with error bars
            ax.errorbar(x, average_data, yerr=std_dev_data, 
                        fmt='o', capsize=5, label=vlabel)
            
        # Labels and title
        ax.set_xlabel(xlabel)
        if multi_species:
            ax.set_ylabel(vunits)
            ax.legend()
        else:
            ax.set_ylabel(vlabel)
        #-- to change window
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    title = slicetitle+tlabel+r'$\in[%2.2e,%2.2e]$'%(t[0],t[-1])
    if len(axs) > 1:
        fig.suptitle(title)
    else:
        ax.set_title(title)
    fig.tight_layout()

def plot_1D(simulation,cdirection,ccoords,fieldnames='',
                           tfs=[], xlim=[], ylim=[], multi_species = True):
    
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
            x,t,values,xlabel,tlabel,vlabel,vunits,slicetitle =\
                get_1xt_diagram(simulation,subfield,cdirection,ccoords,tfs=tfs)
            # Compute the average of data over the t-axis (axis=1)
            average_data = np.mean(values, axis=0)
            # Compute the standard deviation of data over the t-axis (axis=1)
            std_dev_data = np.std(values, axis=0)
            if time_avg:
                # Plot with error bars
                ax.errorbar(x, average_data, yerr=std_dev_data, 
                            fmt='o', capsize=5, label=vlabel)
            else:
                # Classic plot
                ax.plot(x, average_data, label=vlabel)
            
        # Labels and title
        ax.set_xlabel(xlabel)
        if multi_species:
            ax.set_ylabel(vunits)
            ax.legend()
        else:
            ax.set_ylabel(vlabel)
        #-- to change window
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
    if t[0] == t[-1]:
        title = slicetitle+tlabel+r'$=%2.2e$'%(t[0])
    else:
        title = slicetitle+tlabel+r'$\in[%2.2e,%2.2e]$'%(t[0],t[-1])
    if len(axs) > 1:
        fig.suptitle(title)
    else:
        ax.set_title(title)
    fig.tight_layout()

def plot_2D_cut(simulation,cdirection,ccoord,tf,
                fieldnames='', cmap='inferno', full_plot=False,
                xlim=[], ylim=[], clim=[], 
                figout=[],cutout=[],):
    cmap0 = cmap    
    fields,fig,axs = setup_figure(fieldnames)
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
                      fieldname='', cmap='inferno',
                      xlim=[], ylim=[], clim=[], full_plot=False):
    os.makedirs('gif_tmp', exist_ok=True)
    if fieldname in simulation.data_param.spec_undep_quantities:
        spec = ''
    for tf in tfs:
        figout = []; cutout = []
        plot_2D_cut(simulation,cdirection,ccoord,tf=tf,fieldnames=fieldname,
                    cmap=cmap,full_plot=full_plot,
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
    fig,axs = plt.subplots(nrow,ncol,figsize=(4*ncol,3*nrow))
    if ncol == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    return fields,fig,axs

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

def plot_volume_integral_vs_t(simulation, fieldnames, tfs=[], ddt=False,
                              jacob_squared=False, plot_src_input=False):
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
                ftot_t.append(f_.compute_volume_integral(jacob_squared=jacob_squared))
            if ddt: # time derivative
                dfdt   = np.gradient(ftot_t,time)
                # we rescale it to obtain a result in seconds
                Ft = dfdt/simulation.normalization['tscale']
            else:
                Ft  = ftot_t
            
            # Convert to np arrays
            ftot_t = np.array(ftot_t)
            time   = np.array(time)

            # Setup labels
            Flbl = simulation.normalization[subfield+'symbol']
            Flbl = r'$\int$ '+Flbl+r' $d^3x$'
            xlbl = label_from_simnorm(simulation,'t')
            ylbl = multiply_by_m3_expression(simulation.normalization[subfield+'units'])

            if ddt:
                Flbl = r'$\partial_t$ '+Flbl
                ylbl = ylbl+'/s'
            # Plot
            ax.plot(time,Ft,label=Flbl)

        # plot eventually the input power for comparison
        if subfield == 'Wtot' and plot_src_input:
            src_power = simulation.get_input_power()
            if ddt:
                ddtWsrc_t = src_power*np.ones_like(time)/simulation.normalization['Wtotscale']
                plt.plot(time,ddtWsrc_t,'--k',label='Source input')
            else:
                # plot the accumulate energy from the source
                Wsrc_t = ftot_t[0] + src_power*simulation.normalization['tscale']*time/simulation.normalization['Wtotscale']
                plt.plot(time,Wsrc_t,'--k',label='Source input')
        # add labels and show legend
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.legend()
        fig.tight_layout()
    
    
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