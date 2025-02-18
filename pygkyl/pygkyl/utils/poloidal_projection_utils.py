import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
#.Scipy is used for interpolation and integration.
import scipy.integrate as integrate
from scipy.interpolate import griddata
from scipy.interpolate import pchip_interpolate
#.JAX is used for automatic differentiation (AD).
import jax.numpy as jnp
from jax import grad, jit, vmap
from matplotlib.patches import Rectangle

from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                mark_inset)
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib import ticker

from ..classes import Frame, Simulation

def poloidal_projection_def(simulation, fieldName='', timeFrames=0, outFilename='',nzInterp=32, scaleFac=1.,
                        colorMap = 'inferno', doInset=True, xlim=[],ylim=[],clim=[]):
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

    #................................................................................#
    #.
    #.Plot the color map of a field on the poloidal plane given the flux-tube data.
    #.There are two options:
    #.  a) Perform all interpolations in field aligned coordinates and use an FFT
    #.     This may only be valid for the potential which is FEM and not DG.
    #.  b) Interpolate in the parallel direction onto a finer grid, then transform
    #.     to cylindrical and perform another interpolation onto the plotting points.
    #.
    #.Manaure Francisquez.
    #.
    #................................................................................#

    # Geometry and magnetic field.
    Z_axis     = simulation.geom_param.Z_axis
    R_axis     = simulation.geom_param.R_axis
    R_LCFSmid  = simulation.geom_param.R_LCFSmid
    x_in = simulation.geom_param.x_LCFS
    x_out = simulation.geom_param.Lx - x_in
    Rmid_min   = R_LCFSmid-x_in # Minimum midplane major radius of simulation box [m].
    Rmid_max   = R_LCFSmid+x_out # Maximum midplane major radius of simulation box [m].
    R0         = 0.5*(Rmid_min+Rmid_max)  # Major radius of the simulation box [m].
    r0         = R0-R_axis          # Minor radius of the simulation box [m].

    #.Miller geometry parameters.
    kappa = simulation.geom_param.kappa
    delta = simulation.geom_param.delta

    #.Magnetic safety factor profile.
    qprofile = simulation.geom_param.qprofile

    #.Some fontsizes used in plots.
    xyLabelFontSize       = 18
    titleFontSize         = 18
    colorBarLabelFontSize = 18
    tickFontSize          = 17

    field_frame = Frame(simulation, name=fieldName, tf=timeFrames, load=True)
    dataInterp = field_frame.values
    xNodal = field_frame.xNodal
    field = np.squeeze(dataInterp)
    
    dimInt = len(xNodal)

    xInt = xNodal
    nxInt = np.zeros(dimInt, dtype='int')
    lxInt = np.zeros(dimInt, dtype='double')
    dxInt = np.zeros(dimInt, dtype='double')
    for i in range(dimInt):
        nxInt[i] = np.size(xInt[i])
        lxInt[i] = xInt[i][-1]-xInt[i][0]
        dxInt[i] = xInt[i][ 1]-xInt[i][0]

    xIntC = [[] for i in range(dimInt)]
    for i in range(dimInt):
        nNodes  = len(xNodal[i])
        xIntC[i] = np.zeros(nNodes-1)
        xIntC[i] = np.multiply(0.5,xNodal[i][0:nNodes-1]+xNodal[i][1:nNodes])
    nxIntC = np.zeros(dimInt, dtype='int')
    lxIntC = np.zeros(dimInt, dtype='double')
    dxIntC = np.zeros(dimInt, dtype='double')
    for i in range(dimInt):
        nxIntC[i] = np.size(xIntC[i])
        lxIntC[i] = xIntC[i][-1]-xIntC[i][0]
        dxIntC[i] = xIntC[i][ 1]-xIntC[i][0]

    xi_lcfs = int(nxIntC[0]*1/3)
    zNumInt = nzInterp*nxIntC[2]

    #.Magnetic safety factor in the middle of the simulation box.
    q0 = qprofile(r0)

    toroidal_mode_number = 2.*np.pi*r0/q0/lxIntC[1]  #.n_0 in Goerler et al.

    #.......................................................................#
    #.Precompute grids and arrays needed in transforming/plotting data below.
    #.Approach: FFT along y, then follow a procedure similar to that in pseudospectral
    #.codes (e.g. GENE, see Xavier Lapillonne's PhD thesis 2010, section 3.2.2, page 55).

    field_frame = Frame(simulation, name=fieldName, tf=timeFrames, load=True)
    dataInterp = field_frame.values
    xNodal = field_frame.xNodal
    field = np.squeeze(dataInterp)

    field_ky = np.fft.rfft(field, axis=1, norm="forward")
    kxIntC = field_ky.shape

    #.Extend along z by in each direction by applying twist-shift BCs in the 
    #.closed-flux region, and just copying the last values (along z) in the SOL.
    z_ex = np.concatenate(([xIntC[2][0]-0.5*dxIntC[2]],xIntC[2],[xIntC[2][-1]+0.5*dxIntC[2]])) # TEST
    esPotInt_kex = np.zeros(kxIntC+np.array([0,0,2]), dtype=np.cdouble)

    bcPhaseShift = 1j*2.0*np.pi*toroidal_mode_number*qprofile(simulation.geom_param.r_x(xIntC[0][:xi_lcfs]))
    
    #.Interpolate onto a finer mesh along z.
    z_int = np.linspace(z_ex[0],z_ex[-1],zNumInt)
    esPotInt_kintPos = np.zeros((kxIntC[0],kxIntC[1],zNumInt), dtype=np.cdouble)

    #.Append negative ky values.
    esPotInt_kint = np.zeros((kxIntC[0],2*kxIntC[1],zNumInt), dtype=np.cdouble)

    #.Compute R(x,z) and Z(x,z).
    numInt = [nxIntC[0], zNumInt]
    RInt, ZInt = np.zeros((nxIntC[0],zNumInt)), np.zeros((nxIntC[0],zNumInt))
    for i in range(nxIntC[0]):
        for k in range(zNumInt):
            x, z = simulation.geom_param.r_x(xIntC[0][i]), z_int[k]

            RInt[i,k] = R_axis + x*np.cos(z + delta*np.sin(z))
            ZInt[i,k] = Z_axis + kappa*x*np.sin(z)

    #.Calculate R,Z for LCFS plotting
    RInt_lcfs, ZInt_lcfs = np.zeros(zNumInt), np.zeros(zNumInt)
    for k in range(zNumInt):
        x, z = simulation.geom_param.r_x(xIntC[0][xi_lcfs]), z_int[k]
        
        RInt_lcfs[k] = R_axis + x*np.cos(z + delta*np.sin(z))
        ZInt_lcfs[k] = Z_axis + kappa*x*np.sin(z)
        
    #.Compute alpha(r,z,phi=0) which is independent of y:
    alpha_rz_phi0 = np.zeros([nxIntC[0],zNumInt])
    for i in range(nxIntC[0]):
        for k in range(zNumInt):
            alpha_rz_phi0[i,k]  = simulation.geom_param.alpha_f(simulation.geom_param.r_x(xIntC[0][i]),z_int[k],0.)
        
    #.Convert (x,y,z) data to (R,Z):
    xyz2RZ = np.zeros([nxIntC[0],2*kxIntC[1],zNumInt], dtype=np.cdouble)
    for j in range(kxIntC[1]):
        for k in range(zNumInt):
            #.Positive ky's.
            xyz2RZ[:,j,k]  = np.exp(2.*np.pi*1j*j*(-(r0/q0)*(0. + alpha_rz_phi0[:,k])/lxInt[1]))  #.phi=0. lx-->lxInt
            #.Negative ky's.
            xyz2RZ[:,-j,k] = np.conj(xyz2RZ[:,j,k])

    esPotInt_RZ = np.zeros([nxIntC[0],zNumInt])

    #.Construct nodal coordinates needed for pcolormesh.
    RIntN, ZIntN = np.zeros((numInt[0]+1,numInt[1]+1)), np.zeros((numInt[0]+1,numInt[1]+1))
    for j in range(numInt[1]):
        for i in range(numInt[0]):
            RIntN[i,j] = RInt[i,j]-0.5*(RInt[1,j]-RInt[0,j])
        RIntN[numInt[0],j] = RInt[-1,j]+0.5*(RInt[-1,j]-RInt[-2,j])
        RIntN[:,numInt[1]] = RIntN[:,-2]

    for i in range(numInt[0]):
        for j in range(numInt[1]):
            ZIntN[i,j] = ZInt[i,j]-0.5*(ZInt[i,1]-ZInt[i,0])
        ZIntN[i,numInt[1]] = ZInt[i,-1]+0.5*(ZInt[i,-1]-ZInt[i,-2])
        ZIntN[numInt[0],:] = ZIntN[-2,:]

    del RInt, ZInt

    #.Finished precomputing grids and arrays needed in transforming/plotting.
    #.......................................................................#
    if not isinstance(timeFrames, list):
        timeFrames = [timeFrames]

    for t in timeFrames:
        if t%10 ==0:
            print('t=%d'%t)

        field_frame = Frame(simulation, fieldName, t, load=True)
        dataInterp = field_frame.values
        xNodal = field_frame.xNodal
        field = np.squeeze(dataInterp)

        minSOL = np.amin(field[xi_lcfs:])
        maxSOL = np.amax(field[xi_lcfs:])
        
        np.seterr(invalid='ignore')
        #.......................................................................#
        #.Approach: FFT along y, then follow a procedure similar to that in pseudospectral
        #.codes (e.g. GENE, see Xavier Lapillonne's PhD thesis 2010, section 3.2.2, page 55).

        field_ky = np.fft.rfft(field, axis=1, norm="forward")

        #.Extend along z by in each direction by applying twist-shift BCs in the 
        #.closed-flux region, and just copying the last values (along z) in the SOL.
        esPotInt_kex[:,:,1:-1] = field_ky
        for j in range(kxIntC[1]):
            esPotInt_kex[:xi_lcfs,j,0]  = field_ky[:xi_lcfs,j,-1]*np.exp( bcPhaseShift*j)
            esPotInt_kex[:xi_lcfs,j,-1] = field_ky[:xi_lcfs,j, 0]*np.exp(-bcPhaseShift*j)
            esPotInt_kex[xi_lcfs:,j,0]  = field_ky[xi_lcfs:,j, 1]
            esPotInt_kex[xi_lcfs:,j,-1] = field_ky[xi_lcfs:,j,-2]

        #.Interpolate onto a finer mesh along z.
        for i in range(kxIntC[0]):
            for j in range(kxIntC[1]):
                esPotInt_kintPos[i,j,:] = pchip_interpolate(z_ex, esPotInt_kex[i,j,:], z_int)

        #.Append negative ky values.
        for i in range(kxIntC[0]):
            for j in range(kxIntC[1]):
                esPotInt_kint[i,j,:]  = esPotInt_kintPos[i,j,:]
                esPotInt_kint[i,-j,:] = np.conj(esPotInt_kintPos[i,j,:])

        #.Convert (x,y,z) data to (R,Z):
        for i in range(nxIntC[0]):
            for k in range(zNumInt):
                esPotInt_RZ[i,k] = np.real(np.sum(xyz2RZ[i,:,k]*esPotInt_kint[i,:,k]))

        fldMin = np.amin(esPotInt_RZ)/scaleFac
        fldMax = np.amax(esPotInt_RZ)/scaleFac

        #.Find max for inset
        
        #.Finished transforming data and setting up grids.
        #.......................................................................#

        #.Create the figure.
        figProp1a = (8.5,9.5)
        ax1aPos   = [ [0.10, 0.08, 0.76, 0.88] ]
        cax1aPos  = [0.88, 0.08, 0.02, 0.88]
        fig1a     = plt.figure(figsize=figProp1a)
        ax1a      = list()
        for i in range(len(ax1aPos)):
            ax1a.append(fig1a.add_axes(ax1aPos[i]))
        cbar_ax1a = fig1a.add_axes(cax1aPos)
        
        hpl1a = list()
        if colorMap == 'bwr':
            fldMax = max(np.abs(fldMin),np.abs(fldMax))
            fldMin = -fldMax
        pcm1 = ax1a[0].pcolormesh(RIntN, ZIntN, np.squeeze(esPotInt_RZ)/scaleFac, shading='auto',cmap=colorMap,
                                        vmin=fldMin,vmax=fldMax)
        hpl1a.append(pcm1)

        #fig1a.suptitle
        ax1a[0].set_title('t = %.2f'%(field_frame.time)+' '+simulation.normalization.dict['tunits'],fontsize=titleFontSize) 
        ax1a[0].set_xlabel(r'$R$ (m)',fontsize=xyLabelFontSize, labelpad=-2)
        #setTickFontSize(ax1a[0],tickFontSize)
        ax1a[0].set_ylabel(r'$Z$ (m)',fontsize=xyLabelFontSize, labelpad=-10)
        cbar = plt.colorbar(hpl1a[0],ax=ax1a,cax=cbar_ax1a)
        cbar.ax.tick_params(labelsize=10)#tickFontSize)
        cbar.set_label(field_frame.vsymbol+r'$(R,\varphi=0,Z)$'+'['+field_frame.vunits+']', 
                       rotation=270, labelpad=18, fontsize=colorBarLabelFontSize)
        hmag = cbar.ax.yaxis.get_offset_text().set_size(tickFontSize)

        #.Plot lcfs
        ax1a[0].plot(RInt_lcfs,ZInt_lcfs,linewidth=1.5,linestyle='--',color='white',alpha=.8)
        ax1a[0].add_patch(Rectangle((0.6,Z_axis-0.01),0.085,0.02,color='gray'))

        if doInset: 
            #.inset data
            axins2 = zoomed_inset_axes(ax1a[0], 1.5, loc='lower left', bbox_to_anchor=(0.42,0.3),bbox_transform=ax1a[0].transAxes)
            img_in = axins2.pcolormesh(RIntN, ZIntN, np.squeeze(esPotInt_RZ)/scaleFac, cmap=colorMap, shading='auto',vmin=minSOL,vmax=maxSOL)
            axins2.plot(RInt_lcfs,ZInt_lcfs,linewidth=1.5,linestyle='--',color='white',alpha=.6)
            cax = inset_axes(axins2,
                            width="10%",  # width = 10% of parent_bbox width
                            height="100%",  # height : 50%
                            loc='lower left',
                            bbox_to_anchor=(1.05, 0., 1, 1),
                            bbox_transform=axins2.transAxes,
                            borderpad=0,)
            fig1a.colorbar(img_in,cax=cax)
            
            # sub region of the original image
            x1, x2, y1, y2 = 1.07, 1.16, Z_axis-.1, Z_axis+.1
            axins2.set_xlim(x1, x2)
            axins2.set_ylim(y1, y2)
            # fix the number of ticks on the inset axes
            axins2.yaxis.get_major_locator().set_params(nbins=7)
            axins2.xaxis.get_major_locator().set_params(nbins=2)
            axins2.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
        
            # draw a bbox of the region of the inset axes in the parent axes and
            # connecting lines between the bbox and the inset axes area
            mark_inset(ax1a[0], axins2, loc1=1, loc2=4, fc="none", ec="0.5")

        ax1a[0].set_aspect('equal',adjustable='datalim')

        if xlim: ax1a.set_xlim(xlim)
        if ylim: ax1a.set_ylim(ylim)
        if clim: pcm1.set_clim(clim)

        if outFilename:
            plt.savefig(outFilename)
            plt.close()
        else:
            plt.show()