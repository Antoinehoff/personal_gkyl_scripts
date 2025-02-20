import numpy as np
import matplotlib.pyplot as plt
import os, sys

from scipy.interpolate import pchip_interpolate
from matplotlib.patches import Rectangle

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib import ticker
from matplotlib import colors

from . import Frame
from ..utils import data_utils
from ..tools import fig_tools

#.Some fontsizes used in plots.
xyLabelFontSize       = 18
titleFontSize         = 18
colorBarLabelFontSize = 18
tickFontSize          = 17

class PoloidalProjection:
  def __init__(self):
    self.sim = None
    self.geom = None
    self.xi_lcfs = 0
    self.bcPhaseShift = 0
    self.kxIntC = 0
    self.z_ex = None
    self.z_int = None
    self.nxIntC = 0
    self.zNumInt = 0
    self.xyz2RZ = None
    self.RIntN = None
    self.ZIntN = None
    self.RInt_lcfs = None
    self.ZInt_lcfs = None
    self.nzInterp = 0

  def setup(self, simulation, fieldName='phi', timeFrame=0, nzInterp=16):
    '''
    This function sets up the grids and interpolation tools for the poloidal projection of a field.
    We FFT along y, then follow a procedure similar to that in pseudospectral codes (see, e.g., 
    Xavier Lapillonne's PhD thesis 2010, section 3.2.2, page 55).
    This code is mainly adapted from M. Francisquez's.

    Inputs:
      simulation: Simulation object.
      fieldName: Name of a field to load and get dimensions for fourier. (default: 'phi')
      timeFrame: frame id of the field to load. (default: 0)
      nzInterp: Number of points to interpolate along z. (default: 16)
    '''

    # Store simulation and a link to goeometry objects
    self.sim = simulation
    self.geom = simulation.geom_param
    self.nzInterp = nzInterp

    # Load a frame to get the grid
    field_frame = Frame(self.sim, name=fieldName, tf=timeFrame, load=True)
    self.xNodal = field_frame.xNodal
    self.dimInt = len(self.xNodal)

    nxInt = np.zeros(self.dimInt, dtype='int')
    lxInt = np.zeros(self.dimInt, dtype='double')
    dxInt = np.zeros(self.dimInt, dtype='double')
    for i in range(self.dimInt):
        nxInt[i] = np.size(self.xNodal[i])
        lxInt[i] = self.xNodal[i][-1]-self.xNodal[i][0]
        dxInt[i] = self.xNodal[i][ 1]-self.xNodal[i][0]

    xIntC = [[] for i in range(self.dimInt)]
    for i in range(self.dimInt):
        nNodes  = len(self.xNodal[i])
        xIntC[i] = np.zeros(nNodes-1)
        xIntC[i] = np.multiply(0.5,self.xNodal[i][0:nNodes-1]+self.xNodal[i][1:nNodes])
    self.nxIntC = np.zeros(self.dimInt, dtype='int')
    lxIntC = np.zeros(self.dimInt, dtype='double')
    dxIntC = np.zeros(self.dimInt, dtype='double')
    for i in range(self.dimInt):
        self.nxIntC[i] = np.size(xIntC[i])
        lxIntC[i] = xIntC[i][-1]-xIntC[i][0]
        dxIntC[i] = xIntC[i][ 1]-xIntC[i][0]

    # xi_lcfs = int(nxIntC[0]*1/3)
    self.xi_lcfs = np.argmin(np.abs(self.geom.grids[0] - self.geom.x_LCFS))

    self.zNumInt = nzInterp*self.nxIntC[2]

    toroidal_mode_number = 2.*np.pi*self.geom.r0/self.geom.q0/lxIntC[1]  #.n_0 in Goerler et al.

    #.......................................................................#
    #.Precompute grids and arrays needed in transforming/plotting data below.
    #.Approach: FFT along y, then follow a procedure similar to that in pseudospectral
    #.codes (e.g. GENE, see Xavier Lapillonne's PhD thesis 2010, section 3.2.2, page 55).

    field = np.squeeze(field_frame.values)
    field_ky = np.fft.rfft(field, axis=1, norm="forward")
    self.kxIntC = field_ky.shape

    #.Extend along z by in each direction by applying twist-shift BCs in the 
    #.closed-flux region, and just copying the last values (along z) in the SOL.
    self.z_ex = np.concatenate(([xIntC[2][0]-0.5*dxIntC[2]],xIntC[2],[xIntC[2][-1]+0.5*dxIntC[2]])) # TEST

    self.bcPhaseShift = 1j*2.0*np.pi*toroidal_mode_number*self.geom.qprofile(self.geom.r_x(xIntC[0][:self.xi_lcfs]))
    
    #.Interpolate onto a finer mesh along z.
    self.z_int = np.linspace(self.z_ex[0],self.z_ex[-1],self.zNumInt)

    #.Compute R(x,z) and Z(x,z).
    numInt = [self.nxIntC[0], self.zNumInt]
    RInt, ZInt = np.zeros((self.nxIntC[0],self.zNumInt)), np.zeros((self.nxIntC[0],self.zNumInt))
    for i in range(self.nxIntC[0]):
        for k in range(self.zNumInt):
            x, z = self.geom.r_x(xIntC[0][i]), self.z_int[k]

            RInt[i,k] = self.geom.R_axis + x*np.cos(z + self.geom.delta*np.sin(z))
            ZInt[i,k] = self.geom.Z_axis + self.geom.kappa*x*np.sin(z)

    #.Calculate R,Z for LCFS plotting
    self.RInt_lcfs, self.ZInt_lcfs = np.zeros(self.zNumInt), np.zeros(self.zNumInt)
    for k in range(self.zNumInt):
        x, z = self.geom.r_x(xIntC[0][self.xi_lcfs]), self.z_int[k]
        
        self.RInt_lcfs[k] = self.geom.R_axis + x*np.cos(z + self.geom.delta*np.sin(z))
        self.ZInt_lcfs[k] = self.geom.Z_axis + self.geom.kappa*x*np.sin(z)
        
    #.Compute alpha(r,z,phi=0) which is independent of y:
    alpha_rz_phi0 = np.zeros([self.nxIntC[0],self.zNumInt])
    for i in range(self.nxIntC[0]):
        for k in range(self.zNumInt):
            alpha_rz_phi0[i,k]  = self.geom.alpha_f(self.geom.r_x(xIntC[0][i]),self.z_int[k],0.)
        
    #.Convert (x,y,z) data to (R,Z):
    self.xyz2RZ = np.zeros([self.nxIntC[0],2*self.kxIntC[1],self.zNumInt], dtype=np.cdouble)
    for j in range(self.kxIntC[1]):
        for k in range(self.zNumInt):
            #.Positive ky's.
            self.xyz2RZ[:,j,k]  = np.exp(2.*np.pi*1j*j*(-(self.geom.r0/self.geom.q0)*(0. + alpha_rz_phi0[:,k])/lxInt[1]))  #.phi=0. lx-->lxInt
            #.Negative ky's.
            self.xyz2RZ[:,-j,k] = np.conj(self.xyz2RZ[:,j,k])

    #.Construct nodal coordinates needed for pcolormesh.
    self.RIntN, self.ZIntN = np.zeros((numInt[0]+1,numInt[1]+1)), np.zeros((numInt[0]+1,numInt[1]+1))
    for j in range(numInt[1]):
        for i in range(numInt[0]):
            self.RIntN[i,j] = RInt[i,j]-0.5*(RInt[1,j]-RInt[0,j])
        self.RIntN[numInt[0],j] = RInt[-1,j]+0.5*(RInt[-1,j]-RInt[-2,j])
        self.RIntN[:,numInt[1]] = self.RIntN[:,-2]

    for i in range(numInt[0]):
        for j in range(numInt[1]):
            self.ZIntN[i,j] = ZInt[i,j]-0.5*(ZInt[i,1]-ZInt[i,0])
        self.ZIntN[i,numInt[1]] = ZInt[i,-1]+0.5*(ZInt[i,-1]-ZInt[i,-2])
        self.ZIntN[numInt[0],:] = self.ZIntN[-2,:]

  def plot(self, fieldName, timeFrame, outFilename='', colorMap = '', doInset=True, scaleFac=1., 
           xlim=[],ylim=[],clim=[],climSOL=[], colorScale='linear', logScaleFloor = 1e-3):
    '''
    Plot the color map of a field on the poloidal plane given the flux-tube data.
    There are two options:
      a) Perform all interpolations in field aligned coordinates and use an FFT
         This may only be valid for the potential which is FEM and not DG.
      b) Interpolate in the parallel direction onto a finer grid, then transform
         to cylindrical and perform another interpolation onto the plotting points.

    Inputs:
        fieldName: Name of the field to plot.
        timeFrame: Time frame to plot.
        outFilename: If not empty, save the figure to this file.
        colorMap: Colormap to use. (optional)
        doInset: If True, plot an inset of the SOL region. (default: True)
        scaleFac: Scale factor for the field. (default: 1.)
        xlim: x-axis limits. (optional)
        ylim: y-axis limits. (optional)
        clim: Color limits. (optional)
        climSOL: Color limits for the inset. (optional)
        colorScale: Color scale. (default: 'linear')
    '''

    field_frame = Frame(self.sim, name=fieldName, tf=timeFrame, load=True)
    dataInterp = field_frame.values
    field = np.squeeze(dataInterp)
    
    np.seterr(invalid='ignore') # cursed

    #.Approach: FFT along y, then follow a procedure similar to that in pseudospectral
    #.codes (e.g. GENE, see Xavier Lapillonne's PhD thesis 2010, section 3.2.2, page 55).

    field_ky = np.fft.rfft(field, axis=1, norm="forward")

    #.Extend along z by in each direction by applying twist-shift BCs in the 
    #.closed-flux region, and just copying the last values (along z) in the SOL.
    field_kex = np.zeros(self.kxIntC+np.array([0,0,2]), dtype=np.cdouble)
    field_kex[:,:,1:-1] = field_ky
    for j in range(self.kxIntC[1]):
        field_kex[:self.xi_lcfs,j,0]  = field_ky[:self.xi_lcfs,j,-1]*np.exp( self.bcPhaseShift*j)
        field_kex[:self.xi_lcfs,j,-1] = field_ky[:self.xi_lcfs,j, 0]*np.exp(-self.bcPhaseShift*j)
        field_kex[self.xi_lcfs:,j,0]  = field_ky[self.xi_lcfs:,j, 1]
        field_kex[self.xi_lcfs:,j,-1] = field_ky[self.xi_lcfs:,j,-2]

    #.Interpolate onto a finer mesh along z.
    field_kintPos = np.zeros((self.kxIntC[0],self.kxIntC[1],self.zNumInt), dtype=np.cdouble)
    for i in range(self.kxIntC[0]):
        for j in range(self.kxIntC[1]):
            field_kintPos[i,j,:] = pchip_interpolate(self.z_ex, field_kex[i,j,:], self.z_int)

    #.Append negative ky values.
    field_kint = np.zeros((self.kxIntC[0],2*self.kxIntC[1],self.zNumInt), dtype=np.cdouble)
    for i in range(self.kxIntC[0]):
        for j in range(self.kxIntC[1]):
            field_kint[i,j,:]  = field_kintPos[i,j,:]
            field_kint[i,-j,:] = np.conj(field_kintPos[i,j,:])

    #.Convert (x,y,z) data to (R,Z):
    field_RZ = np.zeros([self.nxIntC[0],self.zNumInt])
    for i in range(self.nxIntC[0]):
        for k in range(self.zNumInt):
            field_RZ[i,k] = np.real(np.sum(self.xyz2RZ[i,:,k]*field_kint[i,:,k]))

    # handle colormap and limits
    colorMap = colorMap if colorMap else self.sim.fields_info[fieldName+'colormap']

    vlims, vlims_SOL = data_utils.get_minmax_values(self.sim, fieldName, [timeFrame])
    if colorMap == 'inferno': 
        vlims[0] = np.max([0,vlims[0]])
        vlims_SOL[0] = np.max([0,vlims_SOL[0]])
        
    elif colorMap == 'bwr':
        vmax = np.max(np.abs(vlims))
        vlims = [-vmax, vmax]
        vmax_SOL = np.max(np.abs(vlims_SOL))
        vlims_SOL = [-vmax_SOL, vmax_SOL]

    if clim:
      fldMin = clim[0]
      fldMax = clim[1]
    else:
      fldMin = vlims[0]
      fldMax = vlims[1]

    if climSOL:
      minSOL = climSOL[0]
      maxSOL = climSOL[1]
    else:
      minSOL = vlims_SOL[0]
      maxSOL = vlims_SOL[1]

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
    pcm1 = ax1a[0].pcolormesh(self.RIntN, self.ZIntN, np.squeeze(field_RZ)/scaleFac, shading='auto',cmap=colorMap,
                              vmin=fldMin,vmax=fldMax)
    hpl1a.append(pcm1)

    #fig1a.suptitle
    ax1a[0].set_title('t = %.2f'%(field_frame.time)+' '+self.sim.normalization.dict['tunits'],fontsize=titleFontSize) 
    ax1a[0].set_xlabel(r'$R$ (m)',fontsize=xyLabelFontSize, labelpad=-2)
    #setTickFontSize(ax1a[0],tickFontSize)
    ax1a[0].set_ylabel(r'$Z$ (m)',fontsize=xyLabelFontSize, labelpad=-10)
    cbar = plt.colorbar(hpl1a[0],ax=ax1a,cax=cbar_ax1a)
    cbar.ax.tick_params(labelsize=10)#tickFontSize)
    cbar.set_label(field_frame.vsymbol+r'$(R,\varphi=0,Z)$'+'['+field_frame.vunits+']', 
                    rotation=270, labelpad=18, fontsize=colorBarLabelFontSize)
    hmag = cbar.ax.yaxis.get_offset_text().set_size(tickFontSize)

    #.Plot lcfs
    ax1a[0].plot(self.RInt_lcfs,self.ZInt_lcfs,linewidth=1.5,linestyle='--',color='white',alpha=.8)

    #.Plot the limiter
    xWidth = np.min(self.RInt_lcfs) - np.min(self.RIntN)
    xCorner = np.min(self.RIntN)
    yWidth = 0.01
    yCorner = self.geom.Z_axis - 0.5*yWidth
    ax1a[0].add_patch(Rectangle((xCorner,yCorner),xWidth,yWidth,color='gray'))

    if doInset: 
      #.inset data
      axins2 = zoomed_inset_axes(ax1a[0], 1.5, loc='lower left', 
                                  bbox_to_anchor=(0.42,0.3),bbox_transform=ax1a[0].transAxes)
      img_in = axins2.pcolormesh(self.RIntN, self.ZIntN, np.squeeze(field_RZ)/scaleFac, 
                                  cmap=colorMap, shading='auto',vmin=minSOL,vmax=maxSOL)
      axins2.plot(self.RInt_lcfs,self.ZInt_lcfs,linewidth=1.5,linestyle='--',color='white',alpha=.6)
      cax = inset_axes(axins2,
                      width="10%",  # width = 10% of parent_bbox width
                      height="100%",  # height : 50%
                      loc='lower left',
                      bbox_to_anchor=(1.05, 0., 1, 1),
                      bbox_transform=axins2.transAxes,
                      borderpad=0,)
      fig1a.colorbar(img_in,cax=cax)
      
      # sub region of the original image
      x1, x2, y1, y2 = 1.07, 1.16, self.geom.Z_axis-.1, self.geom.Z_axis+.1
      axins2.set_xlim(x1, x2)
      axins2.set_ylim(y1, y2)
      if colorScale == 'log':
        colornorm = colors.LogNorm(vmax=maxSOL, vmin=logScaleFloor*maxSOL) if minSOL > 0 \
            else colors.SymLogNorm(vmax=maxSOL, vmin=minSOL, linscale=1.0, linthresh=logScaleFloor*maxSOL)
        img_in.set_norm(colornorm)
      if climSOL: img_in.set_clim(climSOL)
      axins2.set_xticks([])
      axins2.set_yticks([])
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
    if colorScale == 'log':
        colornorm = colors.LogNorm(vmax=fldMax, vmin=logScaleFloor*fldMax) if minSOL > 0 \
            else colors.SymLogNorm(vmax=fldMax, vmin=fldMin, linscale=1.0, linthresh=logScaleFloor*fldMax)
        pcm1.set_norm(colornorm)
    if clim: pcm1.set_clim(clim)

    if outFilename:
        plt.savefig(outFilename)
        plt.close()
    else:
        plt.show()

  def movie(self, fieldName, timeFrames, moviePrefix='', colorMap =None, doInset=True, scaleFac=1., 
          xlim=[],ylim=[],clim=[],climSOL=[], colorScale='linear', logScaleFloor = 1e-3):
      # Create a temporary folder to store the movie frames
      movDirTmp = 'movie_frames_tmp'
      os.makedirs(movDirTmp, exist_ok=True)   

      # handle color limits to fix the colorbar
      vlims, vlims_SOL = data_utils.get_minmax_values(self.sim, fieldName, timeFrames)
      colorMap = colorMap if colorMap else self.sim.fields_info[fieldName+'colormap']
      if colorMap == 'inferno': 
        vlims[0] = np.max([logScaleFloor if colorScale=='log' else 0, vlims[0]])
        vlims_SOL[0] = np.max([logScaleFloor if colorScale=='log' else 0, vlims_SOL[0]])
      elif colorMap == 'bwr':
          vmax = np.max(np.abs(vlims))
          vlims = [-vmax, vmax]
          vmax_SOL = np.max(np.abs(vlims_SOL))
          vlims_SOL = [-vmax_SOL, vmax_SOL]
      clim = clim if clim else vlims
      climSOL = climSOL if climSOL else vlims_SOL
      
      frameFileList = []
      total_frames = len(timeFrames)
      for i, tf in enumerate(timeFrames, 1):  # Start the index at 1  
          frameFileName = f'movie_frames_tmp/frame_{tf}.png'
          frameFileList.append(f'movie_frames_tmp/frame_{tf}.png')

          self.plot(fieldName=fieldName, timeFrame=tf, outFilename=frameFileName,
                          colorMap = colorMap, doInset=doInset, scaleFac=scaleFac,
                          colorScale=colorScale, logScaleFloor=logScaleFloor,
                          xlim=xlim, ylim=ylim, clim=clim, climSOL=climSOL)
          cutname = ['RZ'+str(self.nzInterp)]

          # Update progress
          progress = f"Processing frames: {i}/{total_frames}... "
          sys.stdout.write("\r" + progress)
          sys.stdout.flush()

      sys.stdout.write("\n")

      # Naming
      movieName = fieldName+'_RZ'
      if colorScale == 'log': movieName = 'log'+movieName
      movieName = moviePrefix + movieName
      movieName+='_xlim_%2.2d_%2.2d'%(xlim[0],xlim[1]) if xlim else ''
      movieName+='_ylim_%2.2d_%2.2d'%(ylim[0],ylim[1]) if ylim else ''

      # Compiling the movie images
      fig_tools.compile_movie(frameFileList, movieName, rmFrames=True)