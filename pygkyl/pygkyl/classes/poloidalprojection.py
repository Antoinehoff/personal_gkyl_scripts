import numpy as np
import matplotlib.pyplot as plt
import os, sys

from scipy.interpolate import pchip_interpolate
from matplotlib.patches import Rectangle

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib import ticker
from matplotlib import colors

from . import Frame, TimeSerie
from ..utils import data_utils
from ..tools import fig_tools, math_tools

#.Some fontsizes used in plots.
xyLabelFontSize       = 18
titleFontSize         = 18
colorBarLabelFontSize = 18
tickFontSize          = 17

class PoloidalProjection:
  def __init__(self):
    self.sim = None
    self.geom = None
    self.ixLCFS_C = 0
    self.bcPhaseShift = 0
    self.kyDimsC = 0
    self.zGridEx = None
    self.zgridI = None
    self.dimsC = 0
    self.nzI = 0
    self.xyz2RZ = None
    self.RIntN = None
    self.ZIntN = None
    self.Rlcfs = None
    self.Zlcfs = None
    self.nzInterp = 0
    self.figSize = None
    self.inset = None

  def setup(self, simulation, fieldName='phi', timeFrame=0, nzInterp=16,
            intMethod='trapz32',figSize = (8,9)):

    # Store simulation and a link to geometry objects
    self.sim = simulation
    self.geom = simulation.geom_param
    self.nzInterp = nzInterp
    self.figSize = figSize
    if self.sim.polprojInset is not None:
      self.inset = self.sim.polprojInset
    else:
      self.inset = Inset()

    # Load a frame to get the grid
    field_frame = Frame(self.sim, name=fieldName, tf=timeFrame, load=True)
    self.gridsN = field_frame.xNodal # nodal grids
    self.ndim = len(self.gridsN) # Dimensionality

    # Write down directions of conf space axis for readability
    xDir = 0
    yDir = 1
    zDir = self.ndim-1

    LyN = self.gridsN[yDir][-1]-self.gridsN[yDir][0] # length in the y direction of nodal grid

    # Centered mesh creation
    meshC = [[] for i in range(self.ndim)] 
    for i in range(self.ndim):
        nNodes  = len(self.gridsN[i])
        meshC[i] = np.zeros(nNodes-1)
        meshC[i] = np.multiply(0.5,self.gridsN[i][0:nNodes-1]+self.gridsN[i][1:nNodes])
    self.dimsC = [np.size(meshC[i]) for i in range(self.ndim)]

    # Radial index of the last closed flux surface on the centered mesh
    self.ixLCFS_C = np.argmin(np.abs(meshC[0] - self.geom.x_LCFS))

    #.n_0 in Goerler et al.
    LyC = meshC[yDir][-1]-meshC[yDir][0] # length in the y direction
    torModNum = 2.*np.pi * self.geom.r0 / self.geom.q0 / LyC # torroidal mode number (n_0 in Goerler thesis 2009)
    xGridCore = meshC[xDir][:self.ixLCFS_C] # x grid on in the core region
    self.bcPhaseShift = 1j*2.0*np.pi * torModNum*self.geom.qprofile(self.geom.r_x(xGridCore))

    #.Precompute grids and arrays needed in transforming/plotting data
    field = np.squeeze(field_frame.values)
    field_ky = np.fft.rfft(field, axis=yDir, norm="forward")
    self.kyDimsC = field_ky.shape

    #.Extend along z by in each direction by applying twist-shift BCs in the 
    #.closed-flux region, and just copying the last values (along z) in the SOL.
    # Number of points for the z interpolation (BIG)
    self.nzI = nzInterp*self.dimsC[zDir]

    z1, zN, dz = meshC[zDir][0], meshC[zDir][-1], meshC[zDir][1] - meshC[zDir][0]
    self.zGridEx = np.concatenate( ([z1-0.5*dz], meshC[zDir], [zN+0.5*dz]) ) # TEST (??)

    #.Interpolate onto a finer mesh along z.
    self.zgridI = np.linspace(self.zGridEx[0],self.zGridEx[-1],self.nzI)

    #.Compute R(x,z) and Z(x,z) (this starts to be big)
    xxI, zzI = math_tools.custom_meshgrid(meshC[xDir],self.zgridI)
    dimsI = np.shape(xxI) # interpolation plane dimensions (R,Z)
    RInt = self.geom.R_axis + self.geom.r_x(xxI) * np.cos(zzI + self.geom.delta * np.sin(zzI))
    ZInt = self.geom.Z_axis + self.geom.kappa * self.geom.r_x(xxI) * np.sin(zzI)

    #.Calculate R,Z for LCFS plotting
    rLCFS = self.geom.r_x(meshC[xDir][self.ixLCFS_C])
    self.Rlcfs = self.geom.R_axis + rLCFS * np.cos(self.zgridI + self.geom.delta * np.sin(self.zgridI))
    self.Zlcfs = self.geom.Z_axis + self.geom.kappa * rLCFS * np.sin(self.zgridI)
        
    #.Compute alpha(r,z,phi=0) which is independent of y.
    alpha_rz_phi0 = np.zeros([self.dimsC[0],self.nzI])
    errAbs = 1.e-8
    for i in range(self.dimsC[0]): # we do it point by point because we integrate over r for each point
        for k in range(self.nzI):
            alpha_rz_phi0[i,k]  = self.geom.alpha_f(self.geom.r_x(meshC[0][i]),self.zgridI[k],0.,method=intMethod)
        
    # #.Convert (x,y,z) data to (R,Z):
    phiTor = 0 #.phi=0. lx-->lxInt
    # this is a very big array
    self.xyz2RZ = np.zeros([self.dimsC[xDir],2*self.kyDimsC[yDir],self.nzI], dtype=np.cfloat)
    exponent_fact = -2.*np.pi*1j * (self.geom.r0 / self.geom.q0) / LyN
    for j in range(self.kyDimsC[1]):
        for k in range(self.nzI):
            #.Positive ky's.
            self.xyz2RZ[:,j,k]  = np.exp(exponent_fact* j * (phiTor + alpha_rz_phi0[:,k]))  #.phi=0. lx-->lxInt
            #.Negative ky's.
            self.xyz2RZ[:,-j,k] = np.conj(self.xyz2RZ[:,j,k])

    #.Construct nodal coordinates needed for pcolormesh.
    self.RIntN, self.ZIntN = np.zeros((dimsI[0]+1,dimsI[1]+1)), np.zeros((dimsI[0]+1,dimsI[1]+1))
    for j in range(dimsI[1]):
        for i in range(dimsI[0]):
            self.RIntN[i,j] = RInt[i,j]-0.5*(RInt[1,j]-RInt[0,j])
        self.RIntN[dimsI[0],j] = RInt[-1,j]+0.5*(RInt[-1,j]-RInt[-2,j])
        self.RIntN[:,dimsI[1]] = self.RIntN[:,-2]

    for i in range(dimsI[0]):
        for j in range(dimsI[1]):
            self.ZIntN[i,j] = ZInt[i,j]-0.5*(ZInt[i,1]-ZInt[i,0])
        self.ZIntN[i,dimsI[1]] = ZInt[i,-1]+0.5*(ZInt[i,-1]-ZInt[i,-2])
        self.ZIntN[dimsI[0],:] = self.ZIntN[-2,:]
        
  def reset_inset(self):
    self.inset = Inset()
        
  def project_field(self,field):
    
    #.Approach: FFT along y, then follow a procedure similar to that in pseudospectral
    #.codes (e.g. GENE, see Xavier Lapillonne's PhD thesis 2010, section 3.2.2, page 55).
    field_ky = np.fft.rfft(field, axis=1, norm="forward")

    #.Extend along z by in each direction by applying twist-shift BCs in the 
    #.closed-flux region, and just copying the last values (along z) in the SOL.
    field_kex = np.zeros(self.kyDimsC+np.array([0,0,2]), dtype=np.cdouble)
    field_kex[:,:,1:-1] = field_ky
    for j in range(self.kyDimsC[1]):
        field_kex[:self.ixLCFS_C,j,0]  = field_ky[:self.ixLCFS_C,j,-1]*np.exp( self.bcPhaseShift*j)
        field_kex[:self.ixLCFS_C,j,-1] = field_ky[:self.ixLCFS_C,j, 0]*np.exp(-self.bcPhaseShift*j)
        field_kex[self.ixLCFS_C:,j,0]  = field_ky[self.ixLCFS_C:,j, 1]
        field_kex[self.ixLCFS_C:,j,-1] = field_ky[self.ixLCFS_C:,j,-2]

    #.Interpolate onto a finer mesh along z.
    field_kintPos = np.zeros((self.kyDimsC[0],self.kyDimsC[1],self.nzI), dtype=np.cdouble)
    # separate real and imaginary part
    field_kintPos_real = np.zeros((self.kyDimsC[0],self.kyDimsC[1],self.nzI), dtype=np.double)
    field_kintPos_imag = np.zeros((self.kyDimsC[0],self.kyDimsC[1],self.nzI), dtype=np.double)
    # interpolate real and imaginary part separately
    for i in range(self.kyDimsC[0]):
        for j in range(self.kyDimsC[1]):
            field_kintPos_real[i,j,:] = pchip_interpolate(self.zGridEx, np.real(field_kex[i,j,:]), self.zgridI)
            field_kintPos_imag[i,j,:] = pchip_interpolate(self.zGridEx, np.imag(field_kex[i,j,:]), self.zgridI)
    # combine real and imaginary part
    field_kintPos = field_kintPos_real + 1j*field_kintPos_imag

    #.Append negative ky values.
    field_kint = np.zeros((self.kyDimsC[0],2*self.kyDimsC[1],self.nzI), dtype=np.cdouble)
    for i in range(self.kyDimsC[0]):
        for j in range(self.kyDimsC[1]):
            field_kint[i,j,:]  = field_kintPos[i,j,:]
            field_kint[i,-j,:] = np.conj(field_kintPos[i,j,:])

    #.Convert (x,y,z) data to (R,Z):
    field_RZ = np.zeros([self.dimsC[0],self.nzI])
    for i in range(self.dimsC[0]):
        for k in range(self.nzI):
            field_RZ[i,k] = np.real(np.sum(self.xyz2RZ[i,:,k]*field_kint[i,:,k]))
            
    return field_RZ

  def plot(self, fieldName, timeFrame, outFilename='', colorMap = '', inset=True, fluctuation='',
           xlim=[],ylim=[],clim=[],climInset=[], colorScale='linear', logScaleFloor = 1e-3, favg = None,
           shading='auto'):
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
        climInset: Color limits for the inset. (optional)
        colorScale: Color scale. (default: 'linear')
    '''
    if isinstance(fluctuation, bool): fluctuation = 'tavg' if fluctuation else ''
    if isinstance(timeFrame, list):
      avg_window = timeFrame
      timeFrame = timeFrame[-1]
    else:
      avg_window = [timeFrame]
    
    with Frame(self.sim, name=fieldName, tf=timeFrame, load=True) as field_frame:
      time = field_frame.time
      vsymbol = field_frame.vsymbol
      vunits = field_frame.vunits
      toproject = field_frame.values

    if len(fluctuation) > 0:
      if favg is not None:
        toproject -= favg
      else:
        serie = TimeSerie(simulation=self.sim, name=fieldName, time_frames=avg_window, load=True)
        if 'tavg' in fluctuation:
          average = serie.get_time_average()
          vsymbol = r'$\delta_t$'+vsymbol
        elif 'yavg' in fluctuation:
          average = serie.get_y_average()
          vsymbol = r'$\delta_y$'+vsymbol
        toproject -= average
        if 'relative' in fluctuation:
          toproject = 100.0 * toproject / average
          vunits = r'\%'
      colorMap = colorMap if colorMap else 'bwr'
    else:
      colorMap = colorMap if colorMap else self.sim.fields_info[fieldName+'colormap']

    field_RZ = self.project_field(toproject)
    
    vlims = [np.min(field_RZ), np.max(field_RZ)]
    vlims_SOL = [np.min(field_RZ[self.ixLCFS_C:,:]), np.max(field_RZ[self.ixLCFS_C:,:])]
    
    lcfColor = 'white' # default color for LCFS
    if colorMap == 'inferno': 
        vlims[0] = np.max([0,vlims[0]])
        vlims_SOL[0] = np.max([0,vlims_SOL[0]])
        lcfColor = 'white'
    elif colorMap == 'bwr':
        vmax = np.max(np.abs(vlims))
        vlims = [-vmax, vmax]
        vmax_SOL = np.max(np.abs(vlims_SOL))
        vlims_SOL = [-vmax_SOL, vmax_SOL]
        lcfColor = 'gray'

    if clim:
      fldMin = clim[0]
      fldMax = clim[1]
    else:
      fldMin = vlims[0]
      fldMax = vlims[1]

    if climInset:
      minSOL = climInset[0]
      maxSOL = climInset[1]
    else:
      minSOL = vlims_SOL[0]
      maxSOL = vlims_SOL[1]

    #.Create the figure.
    ax1aPos   = [ [0.10, 0.08, 0.76, 0.88] ]
    cax1aPos  = [0.88, 0.08, 0.02, 0.88]
    fig1a     = plt.figure(figsize=self.figSize)
    ax1a      = list()
    for i in range(len(ax1aPos)):
        ax1a.append(fig1a.add_axes(ax1aPos[i]))
    cbar_ax1a = fig1a.add_axes(cax1aPos)
    
    hpl1a = list()
    pcm1 = ax1a[0].pcolormesh(self.RIntN, self.ZIntN, field_RZ, shading=shading,cmap=colorMap,
                              vmin=fldMin,vmax=fldMax)
    hpl1a.append(pcm1)

    #fig1a.suptitle
    ax1a[0].set_title('t = %.2f'%(time)+' '+self.sim.normalization.dict['tunits'],fontsize=titleFontSize) 
    ax1a[0].set_xlabel(r'$R$ (m)',fontsize=xyLabelFontSize, labelpad=-2)
    #setTickFontSize(ax1a[0],tickFontSize)
    ax1a[0].set_ylabel(r'$Z$ (m)',fontsize=xyLabelFontSize, labelpad=-10)
    cbar = plt.colorbar(hpl1a[0],ax=ax1a,cax=cbar_ax1a)
    cbar.ax.tick_params(labelsize=10)#tickFontSize)
    cbar.set_label(vsymbol+r'$(R,\varphi=0,Z)$'+'['+vunits+']', 
                    rotation=270, labelpad=18, fontsize=colorBarLabelFontSize)
    hmag = cbar.ax.yaxis.get_offset_text().set_size(tickFontSize)

    #.Plot lcfs
    ax1a[0].plot(self.Rlcfs,self.Zlcfs,linewidth=1.5,linestyle='--',color=lcfColor,alpha=.8)

    #.Plot the limiter
    xWidth = np.min(self.Rlcfs) - np.min(self.RIntN)
    xCorner = np.min(self.RIntN)
    yWidth = 0.01
    yCorner = self.geom.Z_axis - 0.5*yWidth
    ax1a[0].add_patch(Rectangle((xCorner,yCorner),xWidth,yWidth,color='gray'))

    if inset:
      self.inset.add_inset(fig1a, ax1a[0], self.RIntN, self.ZIntN, field_RZ, colorMap,
                           colorScale, minSOL, maxSOL, climInset, logScaleFloor, shading,
                           LCFS=[self.Rlcfs,self.Zlcfs,lcfColor], limiter=[yWidth])      

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

  def movie(self, fieldName, timeFrames, moviePrefix='', colorMap =None, inset=True,
          xlim=[],ylim=[],clim=[],climInset=[], colorScale='linear', logScaleFloor = 1e-3,
          pilLoop=0, pilOptimize=False, pilDuration=100, fluctuation=False):
      # Create a temporary folder to store the movie frames
      movDirTmp = 'movie_frames_tmp'
      os.makedirs(movDirTmp, exist_ok=True)   

      if fluctuation:
        with TimeSerie(simulation=self.sim, name=fieldName, time_frames=timeFrames, load=True) \
          as field_frames:
            favg = field_frames.get_time_average()
      else:
        favg = None
        
      clim = clim if clim else []
      climInset = climInset if climInset else []
      
      frameFileList = []
      total_frames = len(timeFrames)
      for i, tf in enumerate(timeFrames, 1):  # Start the index at 1  
          frameFileName = f'movie_frames_tmp/frame_{tf}.png'
          frameFileList.append(f'movie_frames_tmp/frame_{tf}.png')

          self.plot(fieldName=fieldName, timeFrame=tf, outFilename=frameFileName,
                          colorMap = colorMap, inset=inset,
                          colorScale=colorScale, logScaleFloor=logScaleFloor,
                          xlim=xlim, ylim=ylim, clim=clim, climInset=climInset,
                          fluctuation=fluctuation, favg=favg)
          cutname = ['RZ'+str(self.nzInterp)]

          # Update progress
          progress = f"Processing frames: {i}/{total_frames}... "
          sys.stdout.write("\r" + progress)
          sys.stdout.flush()

      sys.stdout.write("\n")

      # Naming
      movieName = fieldName+'_RZ'
      if fluctuation: movieName = 'd' + movieName
      if colorScale == 'log': movieName = 'log'+movieName
      movieName = moviePrefix + movieName
      movieName+='_xlim_%2.2d_%2.2d'%(xlim[0],xlim[1]) if xlim else ''
      movieName+='_ylim_%2.2d_%2.2d'%(ylim[0],ylim[1]) if ylim else ''

      # Compiling the movie images
      fig_tools.compile_movie(frameFileList, movieName, rmFrames=True,
                              pilLoop=pilLoop, pilOptimize=pilOptimize, pilDuration=pilDuration)
      
class Inset:
  """
  Class to add an inset to a plot.
  """
  def __init__(self, zoom=1.5, 
               zoom_loc='lower left', 
               inset_rel_pos=(0.35,0.35), 
               vmin=0, vmax=1,
               width="10%", 
               height="100%", 
               loc='lower left', 
               borderpad=0,
               xlim=(1.07, 1.16),
               ylim=(0.04, 0.24),
               nbinsx=7, nbinsy=2,
               format="{x:.2f}",
               shading='auto',
               anchor_colorbar = (1.05, 0., 1, 1),
               markloc = [1, 4]):
    self.zoom = zoom
    self.zoom_loc = zoom_loc
    self.lower_corner_rel_pos = inset_rel_pos
    self.vmin = vmin
    self.vmax = vmax
    self.width = width
    self.height = height
    self.loc = loc
    self.borderpad = borderpad
    self.xlim = xlim
    self.ylim = ylim
    self.nbinsx = nbinsx
    self.nbinsy = nbinsy
    self.format = format
    self.shading = shading
    self.anchor_colorbar = anchor_colorbar
    self.markloc = markloc
      
  def add_inset(self, fig, ax, R, Z, field_RZ, colorMap, colorScale, 
                minSOL, maxSOL, climInset, logScaleFloor, shading, LCFS=[], limiter=[]):
    # sub region of the original image
    axins = zoomed_inset_axes(ax, self.zoom, loc=self.zoom_loc, 
                              bbox_to_anchor=self.lower_corner_rel_pos,bbox_transform=ax.transAxes)
    img_in = axins.pcolormesh(R, Z, field_RZ,
                              cmap=colorMap, shading=shading,vmin=minSOL,vmax=maxSOL)
      
    cax = inset_axes(axins,
                    width=self.width,
                    height=self.height,
                    loc=self.loc,
                    bbox_to_anchor=self.anchor_colorbar,
                    bbox_transform=axins.transAxes,
                    borderpad=self.borderpad)
    fig.colorbar(img_in,cax=cax)
    axins.set_xlim(self.xlim)
    axins.set_ylim(self.ylim)
    if colorScale == 'log':
      colornorm = colors.LogNorm(vmax=maxSOL, vmin=logScaleFloor*maxSOL) if minSOL > 0 \
          else colors.SymLogNorm(vmax=maxSOL, vmin=minSOL, linscale=1.0, linthresh=logScaleFloor*maxSOL)
      img_in.set_norm(colornorm)
    if climInset: img_in.set_clim(climInset)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.yaxis.get_major_locator().set_params(nbins=self.nbinsy)
    axins.xaxis.get_major_locator().set_params(nbins=self.nbinsx)
    axins.xaxis.set_major_formatter(ticker.StrMethodFormatter(self.format))
    
    if len(LCFS) > 0:
      axins.plot(LCFS[0],LCFS[1],linewidth=1.5,linestyle='--',color=LCFS[2],alpha=.8)

    if len(limiter) > 0:
      xWidth = np.min(LCFS[0]) - np.min(R)
      xCorner = np.min(R)
      yWidth = limiter[0]
      yCorner = 0.5*(np.min(Z)+np.max(Z)) - 0.5*yWidth
      axins.add_patch(Rectangle((xCorner,yCorner),xWidth,yWidth,color='gray'))

    
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=self.markloc[0], loc2=self.markloc[1], fc="none", ec="0.5")