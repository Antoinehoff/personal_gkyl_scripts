import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pyvista as pv

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
    self.phiTor = 0
    self.alpha_rz_phi0 = None
    self.dimsI = 0
    self.meshC = None
    self.gridCheck = False
    self.zExt = True
    self.TSBC = True
    
  def setup(self, simulation, timeFrame=0, nzInterp=16, phiTor=0, Rlim = [], rholim = [],
            intMethod='trapz32',figSize = (8,9), zExt=True, gridCheck=False, TSBC=True):

    # Store simulation and a link to geometry objects
    self.sim = simulation
    self.geom = simulation.geom_param
    self.nzInterp = nzInterp
    self.figSize = figSize

    if self.sim.polprojInset is not None:
      self.inset = self.sim.polprojInset
    else:
      self.inset = Inset()
    self.phiTor = phiTor
    self.gridCheck = gridCheck
    self.TSBC = TSBC
    self.zExt = True if TSBC else zExt
    
    # Load a frame to get the grid
    if len(simulation.available_frames['field']) > 0:
      fieldName = 'phi'
    else:
      fieldName = 'ni'
      
    field_frame = Frame(self.sim, name=fieldName, tf=timeFrame, load=True)
    self.gridsN = field_frame.xNodal # nodal grids
    self.ndim = len(self.gridsN) # Dimensionality

    #.Centered mesh creation
    meshC = [[] for i in range(self.ndim)] 
    for i in range(self.ndim):
        nNodes  = len(self.gridsN[i])
        meshC[i] = np.zeros(nNodes-1)
        meshC[i] = np.multiply(0.5,self.gridsN[i][0:nNodes-1]+self.gridsN[i][1:nNodes])
    self.dimsC = [np.size(meshC[i]) for i in range(self.ndim)]
    self.meshC = meshC
    del meshC
    
    self.LyC = self.meshC[1][-1] - self.meshC[1][0] # length in the y direction
    # Do we need to rescale the y length to fill the integer toroidal mode number ? not sure...
    Ntor0 = 2*np.pi * (self.geom.r0 / self.geom.q0) / self.LyC
    Ntor = int(np.round(Ntor0))
    self.LyC = 2*np.pi * (self.geom.r0 / self.geom.q0) / Ntor
    self.meshC[1] = self.meshC[1] * (self.LyC / (self.meshC[1][-1] - self.meshC[1][0]))

    #.Should we shift the z grid?
    self.meshC[2] = self.meshC[2] #- self.meshC[2][0] # shift the z grid to start at 0

    #.Precompute grids and arrays needed in transforming/plotting data
    field = np.squeeze(field_frame.values)
    field_ky = np.fft.rfft(field, axis=1, norm="forward")
    self.kyDimsC = field_ky.shape

    #.Extend along z by in each direction by applying twist-shift BCs in the 
    #.closed-flux region, and just copying the last values (along z) in the SOL.
    # Number of points for the z interpolation (BIG)
    self.nzI = nzInterp*self.dimsC[2]
  
    zGrid = self.meshC[2]
    z1, zN, dz = zGrid[0], zGrid[-1], zGrid[1] - zGrid[0]
    if self.zExt:
      #. This handles the conection between +pi and -pi regions
      self.zGridEx = np.concatenate( ([z1-dz/2], zGrid, [zN+dz/2]) )
    else:
      self.zGridEx = zGrid
      
    #.Interpolate onto a finer mesh along z.
    self.zgridI = np.linspace(self.zGridEx[0],self.zGridEx[-1],self.nzI)
    
    if Rlim or rholim:
      #.Select the radial region of interest
      if Rlim:
        Rx = self.geom.R_x(self.meshC[0])
        dR = Rx[1] - Rx[0]
        if not isinstance(Rlim, list): Rlim = [Rlim - dR/2, Rlim + dR/2]
        Rlim = [max(R, Rx[ 0]) for R in Rlim]
        Rlim = [min(R, Rx[-1]) for R in Rlim]
        self.ix0 = np.argmin(np.abs(Rx - Rlim[0]))
        self.ix1 = np.argmin(np.abs(Rx - Rlim[1]))
      elif rholim:
        rhox = self.geom.r_x(self.meshC[0])/self.geom.a_mid
        drho = rhox[1] - rhox[0]
        if not isinstance(rholim, list): rholim = [rholim - drho/2, rholim + drho/2]
        elif isinstance(rholim[0],int):
          rholim[0] = rhox[rholim[0]]
          if len(rholim) > 1:
            rholim[1] = rhox[rholim[1]]
        rholim = [max(rho, rhox[ 0]) for rho in rholim]
        rholim = [min(rho, rhox[-1]) for rho in rholim]
        self.ix0 = np.argmin(np.abs(rhox - rholim[0]))
        self.ix1 = np.argmin(np.abs(rhox - rholim[1]))

      if self.ix1 - self.ix0 < 2:
        if self.ix0 == 0:
          self.ix1 = self.ix0 + 2
        if self.ix1 == len(self.meshC[0])-1:
          self.ix0 = self.ix1 - 2
        else:
          self.ix1 = self.ix0 + 2

      self.meshC[0] = self.meshC[0][self.ix0:self.ix1]
            
      self.dimsC[0] = len(self.meshC[0])
      kyDimC = list(self.kyDimsC)
      kyDimC[0] = len(self.meshC[0])
      self.kyDimsC = tuple(kyDimC)
    else:
      self.ix0 = 0
      self.ix1 = self.dimsC[0]
              
    #.Radial index of the last closed flux surface on the centered mesh
    if self.geom.x_LCFS > self.meshC[0][0] and self.geom.x_LCFS <= self.meshC[0][-1]:
      self.ixLCFS_C = np.argmin(np.abs(self.meshC[0] - self.geom.x_LCFS))
    else:
      self.ixLCFS_C = None
      
    #.Calculate R,Z for LCFS plotting
    rLCFS = self.geom.r_x(self.geom.x_LCFS)
    self.Rlcfs = self.geom.R_rt(rLCFS,self.zgridI)
    self.Zlcfs = self.geom.Z_rt(rLCFS,self.zgridI)
    
    self.compute_alpha(method=intMethod)
    
    self.compute_xyz2RZ(phiTor=self.phiTor)

    self.compute_nodal_coordinates()
    
  def compute_alpha(self, method='trapz32'):
    #.Compute alpha(r,z,phi=0) which is independent of y.
    self.alpha_rz_phi0 = np.zeros([self.dimsC[0],self.nzI])
    for ix in range(self.dimsC[0]): # we do it point by point because we integrate over r for each point
      dPsidr = self.geom.dPsidr(self.geom.r_x(self.meshC[0][ix]),method=method)
      for iz in range(self.nzI):
          self.alpha_rz_phi0[ix,iz]  = self.geom.alpha0(self.geom.r_x(self.meshC[0][ix]),self.zgridI[iz], 0.0, method=method)/dPsidr

  def compute_xyz2RZ(self,phiTor=0.0):
    phiTor += np.pi # To match the obmp with varphi=0
    # this can be a very big array
    self.xyz2RZ = np.zeros([self.dimsC[0],2*self.kyDimsC[1],self.nzI], dtype=np.cdouble)
    n0 = 2*np.pi * self.geom.Cy/ self.LyC
    for k in range(self.kyDimsC[1]):
        for iz in range(self.nzI):
            #.Positive ky's.
            ### Not sure at all about this phase factor
            self.xyz2RZ[:,+k,iz]  = np.exp(1j*k*(n0*(self.alpha_rz_phi0[:,iz]) + phiTor))
            #.Negative ky's.
            self.xyz2RZ[:,-k,iz] = np.conj(self.xyz2RZ[:,+k,iz])
            
  def compute_nodal_coordinates(self):
    #.Compute R(x,z) and Z(x,z)
    xxI, zzI = math_tools.custom_meshgrid(self.meshC[0],self.zgridI)
    self.dimsI = np.shape(xxI) # interpolation plane dimensions (R,Z)
    rrI = self.geom.r_x(xxI)
    Rint = self.geom.R_rt(rrI,zzI)
    Zint = self.geom.Z_rt(rrI,zzI)

    self.RIntN, self.ZIntN = np.zeros((self.dimsI[0]+1,self.dimsI[1]+1)), np.zeros((self.dimsI[0]+1,self.dimsI[1]+1))
    for j in range(self.dimsI[1]):
        for i in range(self.dimsI[0]):
            self.RIntN[i,j] = Rint[i,j]-0.5*(Rint[1,j]-Rint[0,j])
        self.RIntN[self.dimsI[0],j] = Rint[-1,j]+0.5*(Rint[-1,j]-Rint[-2,j])
        self.RIntN[:,self.dimsI[1]] = self.RIntN[:,-2]

    for i in range(self.dimsI[0]):
        for j in range(self.dimsI[1]):
            self.ZIntN[i,j] = Zint[i,j]-0.5*(Zint[i,1]-Zint[i,0])
        self.ZIntN[i,self.dimsI[1]] = Zint[i,-1]+0.5*(Zint[i,-1]-Zint[i,-2])
        self.ZIntN[self.dimsI[0],:] = self.ZIntN[-2,:]
        
  def toroidal_rotate(self, dphi=0.0):
    '''
    Rotate by dphi in the toroidal direction.
    This is done by multiplying the projection by exp(1j*k*dphi) for each k.
    '''
    self.phiTor += dphi
    for k in range(self.kyDimsC[1]):
      for iz in range(self.nzI):
          self.xyz2RZ[:,+k,iz] *= np.exp(1j*k* dphi)
          self.xyz2RZ[:,-k,iz] = np.conj(self.xyz2RZ[:,+k,iz])
    
  def project_field(self, field, evalDGfunc=None):
    
    if self.gridCheck:
      field = np.zeros(self.dimsC)
      yGrid = self.meshC[1]
      Ly = yGrid[-1] - yGrid[0]
      sigma = Ly/16
      mu = 0*Ly/4
      for ix in range(len(self.meshC[0])):
        for iy in range(len(self.meshC[1])):
          for iz in range(len(self.meshC[2])):
            field[ix, iy, iz] = np.exp(-0.5 * ((yGrid[iy] - mu) / sigma) ** 2)
      
    if self.zExt and not self.TSBC:
      proj_zExt_lo = np.zeros((self.dimsC[0],self.dimsC[1]), dtype=np.double)
      proj_zExt_up = np.zeros((self.dimsC[0],self.dimsC[1]), dtype=np.double)

      proj_zExt_lo = evalDGfunc(grid = [self.meshC[0], self.meshC[1], -np.pi])
      proj_zExt_up = evalDGfunc(grid = [self.meshC[0], self.meshC[1], np.pi])
          
      field_ex = np.zeros(self.dimsC + np.array([0,0,2]))
      field_ex[:,:,1:-1] = field[self.ix0:self.ix1,:,:]
      field_ex[:,:,0] = proj_zExt_lo
      field_ex[:,:, -1] = proj_zExt_up
      field = field_ex
      
    #.Approach: FFT along y, then follow a procedure similar to that in pseudospectral
    #.codes (e.g. GENE, see Xavier Lapillonne's PhD thesis 2010, section 3.2.2, page 55).
    field_ky = np.fft.rfft(field, axis=1, norm="forward")
    field_ky = field_ky[self.ix0:self.ix1,:,:] # select the radial region of interest
    
    if self.TSBC:
      #.Apply twist-shift BCs in the closed-flux region.
      if self.ixLCFS_C is None: icore_end = self.dimsC[0]
      else: icore_end = self.ixLCFS_C
      xGridCore = self.meshC[0][:icore_end] # x grid on in the core region
      torModNum = 2.*np.pi * (self.geom.r0 / self.geom.q0) / self.LyC # torroidal mode number (n_0 in Lapillone thesis 2009)
      bcPhaseShift = 2.0*np.pi * torModNum*self.geom.qprofile(self.geom.r_x(xGridCore))
      n0 = 2*np.pi * self.geom.Cy/ self.LyC
      field_kex = np.zeros(self.kyDimsC+np.array([0,0,2]), dtype=np.cdouble)
      field_kex[:,:,1:-1] = field_ky
      lo, up = 0, -1
      for ik in range(self.kyDimsC[1]):
        f_lo = field_ky[:icore_end,ik,lo]
        f_up = field_ky[:icore_end,ik,up]
        ts_lu = np.exp(-1j*ik*bcPhaseShift)
        ts_ul = np.exp(+1j*ik*bcPhaseShift)
        field_kex[:icore_end,ik,up]  = 0.5*(f_up + ts_ul * f_lo)
        field_kex[:icore_end,ik,lo]  = 0.5*(f_lo + ts_lu * f_up)
        field_kex[icore_end:,ik,lo]  = field_ky[icore_end:,ik,lo]
        field_kex[icore_end:,ik,up]  = field_ky[icore_end:,ik,up]
    else:
      field_kex = field_ky
  
    #.Interpolate onto a finer mesh along z.
    field_kintPos = np.zeros((self.kyDimsC[0],self.kyDimsC[1],self.nzI), dtype=np.cdouble)
    # separate real and imaginary part
    field_kintPos_real = np.zeros((self.kyDimsC[0],self.kyDimsC[1],self.nzI), dtype=np.double)
    field_kintPos_imag = np.zeros((self.kyDimsC[0],self.kyDimsC[1],self.nzI), dtype=np.double)
    # interpolate real and imaginary part separately
    for ix in range(self.kyDimsC[0]):
        for ik in range(self.kyDimsC[1]):
            field_kintPos_real[ix,ik,:] = pchip_interpolate(self.zGridEx, np.real(field_kex[ix,ik,:]), self.zgridI)
            field_kintPos_imag[ix,ik,:] = pchip_interpolate(self.zGridEx, np.imag(field_kex[ix,ik,:]), self.zgridI)
    # combine real and imaginary part
    field_kintPos = field_kintPos_real + 1j*field_kintPos_imag

    #.Append negative ky values.
    field_kint = np.zeros((self.kyDimsC[0],2*self.kyDimsC[1],self.nzI), dtype=np.cdouble)
    for ix in range(self.kyDimsC[0]):
        for ik in range(self.kyDimsC[1]):
            field_kint[ix,+ik,:] = field_kintPos[ix,ik,:]
            field_kint[ix,-ik,:] = np.conj(field_kintPos[ix,ik,:])

    #.Convert (x,y,z) data to (R,Z):
    field_RZ = np.zeros([self.dimsC[0],self.nzI])
    for ix in range(self.dimsC[0]):
        for iz in range(self.nzI):
            field_RZ[ix,iz] = np.real(np.sum(self.xyz2RZ[ix,:,iz]*field_kint[ix,:,iz]))
            
    return field_RZ
  
  def get_projection(self, fieldName, timeFrame):
    field_frame = Frame(self.sim, name=fieldName, tf=timeFrame, load=True)
    if self.zExt and not self.TSBC :
      evalDGfunc = field_frame.eval_DG_proj
    else:
      evalDGfunc = None
    field_RZ = self.project_field(field_frame.values, evalDGfunc=evalDGfunc)
    return field_RZ, self.RIntN, self.ZIntN

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
      frame_info = Frame(self.sim, name=fieldName, tf=timeFrame, load=False)

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

    field_RZ = self.project_field(toproject, frame_info)

    vlims = [np.min(field_RZ), np.max(field_RZ)]
    if self.ixLCFS_C is not None:
      vlims_SOL = [np.min(field_RZ[self.ixLCFS_C:,:]), np.max(field_RZ[self.ixLCFS_C:,:])]
    else:
      vlims_SOL = vlims
    
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

  def movie(self, fieldName, timeFrames=[], moviePrefix='', colorMap =None, inset=True,
          xlim=[],ylim=[],clim=[],climInset=[], colorScale='linear', logScaleFloor = 1e-3,
          pilLoop=0, pilOptimize=False, pilDuration=100, fluctuation=False, timeFrame=[],
          rmFrames=True):
    
      # Naming
      movieName = fieldName+'_RZ'
      if fluctuation: movieName = 'd' + movieName
      if colorScale == 'log': movieName = 'log'+movieName
      movieName = moviePrefix + movieName
      movieName+='_xlim_%2.2d_%2.2d'%(xlim[0],xlim[1]) if xlim else ''
      movieName+='_ylim_%2.2d_%2.2d'%(ylim[0],ylim[1]) if ylim else ''
      
      # Create a temporary folder to store the movie frames (random name)
      movDirTmp = movieName+'_frames_tmp_%4d'%np.random.randint(9999)
      os.makedirs(movDirTmp, exist_ok=True)   
      
      timeFrames = timeFrame if not timeFrames else timeFrames

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
          frameFileName = f'{movDirTmp}/frame_{tf}.png'
          frameFileList.append(f'{movDirTmp}/frame_{tf}.png')

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

      # Compiling the movie images
      fig_tools.compile_movie(frameFileList, movieName, rmFrames=rmFrames,
                              pilLoop=pilLoop, pilOptimize=pilOptimize, pilDuration=pilDuration)
      
  def reset_inset(self):
    self.inset = Inset()
          
class Inset:
  """
  Class to add an inset to a plot.
  """
  def __init__(self, zoom=1.5, 
               zoomLoc='lower left', 
               insetRelPos=(0.35,0.35), 
               vmin=0, vmax=1,
               width="10%", 
               height="100%", 
               loc='lower left', 
               borderPad=0,
               xlim=(1.07, 1.16),
               ylim=(0.04, 0.24),
               nbinsx=7, nbinsy=2,
               format="{x:.2f}",
               shading='auto',
               anchorColorbar = (1.05, 0., 1, 1),
               markLoc = [1, 4]):
    self.zoom = zoom
    self.zoomLoc = zoomLoc
    self.lowerCornerRelPos = insetRelPos
    self.vmin = vmin
    self.vmax = vmax
    self.width = width
    self.height = height
    self.loc = loc
    self.borderPad = borderPad
    self.xlim = xlim
    self.ylim = ylim
    self.nbinsx = nbinsx
    self.nbinsy = nbinsy
    self.format = format
    self.shading = shading
    self.anchorColorbar = anchorColorbar
    self.markLoc = markLoc
      
  def add_inset(self, fig, ax, R, Z, fieldRZ, colorMap, colorScale, 
                minSOL, maxSOL, climInset, logScaleFloor, shading, LCFS=[], limiter=[]):
    # sub region of the original image
    axins = zoomed_inset_axes(ax, self.zoom, loc=self.zoomLoc, 
                              bbox_to_anchor=self.lowerCornerRelPos,bbox_transform=ax.transAxes)
    img_in = axins.pcolormesh(R, Z, fieldRZ,
                              cmap=colorMap, shading=shading,vmin=minSOL,vmax=maxSOL)
      
    cax = inset_axes(axins,
                    width=self.width,
                    height=self.height,
                    loc=self.loc,
                    bbox_to_anchor=self.anchorColorbar,
                    bbox_transform=axins.transAxes,
                    borderpad=self.borderPad)
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
    mark_inset(ax, axins, loc1=self.markLoc[0], loc2=self.markLoc[1], fc="none", ec="0.5")


class FluxSurfProjection:
  def __init__(self):
    self.sim = None
    self.smooth = True
    self.overSampFact = 2
    self.ix0 = 0
    self.ix1 = 0
    self.Nint = 0
    self.npol = 0
    self.ntor = 0
    self.phi_fs = None
    self.theta_fs = None
    self.phi_rep = None
    self.theta_rep = None
    self.field_rep = None
    self.y_int = None
    self.z_int = None
    
  def setup(self, simulation, timeFrame=0, Nint=64, rho=0.9, phi=[0,2*np.pi], smooth=True):
    
    self.sim = simulation
    self.smooth = smooth
    self.phiLim = phi if len(phi) == 2 else [0, 2 * np.pi]
    
    # Load a frame to get the grid
    if len(simulation.available_frames['field']) > 0:
      fieldName = 'phi'
    else:
      fieldName = 'ni'
      
    frame = Frame(self.sim, name=fieldName, tf=timeFrame, load=True)

     # Compute rcut and find the index
    rx = self.sim.geom_param.r_x(frame.cgrids[0])
    rhox = rx / self.sim.geom_param.a_mid
    if isinstance(rho, int): rho = rhox[rho]
    rcut = rho * self.sim.geom_param.a_mid
    self.ix0 = np.argmin(np.abs(rx - rcut))
    self.r0 = self.sim.geom_param.r_x(frame.cgrids[0][self.ix0])
    self.rho = rho
  
    self.Nint = Nint
    phi_fs = np.linspace(phi[0], phi[1], self.Nint)
    theta_fs = np.linspace(-np.pi, np.pi, self.Nint)
    self.phi_fs, self.theta_fs = np.meshgrid(phi_fs, theta_fs, indexing='ij')
        
  def project_field(self, field, Nintz=24):
    
    phi = np.linspace(self.phiLim[0], self.phiLim[1], self.Nint)
    polproj = PoloidalProjection()
    polproj.setup(self.sim,nzInterp=Nintz, rholim=self.rho)

    field_fs = np.zeros((self.Nint, polproj.nzI))
    
    dphi = (self.phiLim[1] - self.phiLim[0]) / self.Nint
    for i in range(len(phi)):
      polproj.toroidal_rotate(dphi=dphi)
      f_RZ = polproj.project_field(field)
      field_fs[i, :] = f_RZ[0, :]
      
    theta = np.linspace(-np.pi, np.pi, polproj.nzI)
    self.phi_fs, self.theta_fs = np.meshgrid(phi, theta, indexing='ij')
    
    if self.smooth:
      field_fs = math_tools.smooth2D(field_fs)
    
    return field_fs
    
  def get_projection(self, fieldName, timeFrame):
    field_frame = Frame(self.sim, name=fieldName, tf=timeFrame, load=True)
    field_fs = self.project_field(field_frame.values)
    return field_fs, self.phi_fs, self.theta_fs
  
  def plot(self, fieldName, timeFrame, outFilename='', fluctuation='',
           figout=[], xlim=[], ylim=[], clim=[], colorMap=None):

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
      timetitle = field_frame.timetitle
      frame_info = Frame(self.sim, name=fieldName, tf=timeFrame, load=False)

    if len(fluctuation) > 0:
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

    field_fs = self.project_field(toproject)
    
    fig, ax = plt.subplots(figsize=(fig_tools.default_figsz[0], fig_tools.default_figsz[1]))
    colorMap = colorMap if colorMap else self.sim.data_param.field_info_dict[fieldName+'colormap']
    if colorMap == 'bwr':
        vmax = np.max(np.abs(field_fs))
        clim = [-vmax, vmax]
        if np.min(field_fs) > 0:
            clim = []
            colorMap = 'inferno'
    pcm = ax.pcolormesh(self.phi_fs/np.pi, self.theta_fs/np.pi, field_fs, shading='auto')
    cbar = plt.colorbar(pcm, ax=ax)
    clabel = fig_tools.label(vsymbol,vunits)

    fig_tools.finalize_plot(ax, fig, pcm=pcm, xlabel=r'$\varphi/\pi$', ylabel=r'$\theta/\pi$', title=timetitle,
                            figout=figout, xlim=xlim, ylim=ylim, clim=clim, clabel=clabel, cbar=cbar, cmap=colorMap)

class TorusProjection:
  """
  Class to combine the poloidal and flux surface projections and plot field on the full torus.
  """
  def __init__(self):
    self.polprojs = []
    self.fsprojs = []
    self.pvphishift = np.pi/2 # required to have the right orientation in pyvista
    
  def setup(self, simulation, timeFrame=0, Nint_polproj=16, Nint_fsproj=32, phiLim=[0, np.pi], rhoLim=[0.8,1.5], ixlim=[0,-1],
            intMethod='trapz32', figSize = (8,9), zExt=True, gridCheck=False, TSBC=True):
    self.sim = simulation
    self.phiLim = phiLim if isinstance(phiLim, list) else [phiLim]
    self.rhoLim = rhoLim if isinstance(rhoLim, list) else [rhoLim]
    
    #. Poloidal projection setup
    for i in range(len(self.phiLim)):
      self.polprojs.append(PoloidalProjection())
      self.polprojs[i].setup(simulation, timeFrame=timeFrame, nzInterp=Nint_polproj, phiTor=phiLim[i], rholim=rhoLim, 
                             intMethod=intMethod, figSize=figSize, zExt=zExt, gridCheck=gridCheck, TSBC=TSBC)
    
    #. Flux surface projection setup
    for i in range(len(self.rhoLim)):
      self.fsprojs.append(FluxSurfProjection())
      self.fsprojs[i].setup(simulation, timeFrame=timeFrame, Nint=Nint_fsproj, rho=self.rhoLim[i], phi=phiLim)
      
  def get_data(self, fieldName, timeFrame, fluctuation):

    if isinstance(timeFrame, list):
      avg_window = timeFrame
      timeFrame = timeFrame[-1]
    else:
      avg_window = [timeFrame]
    
    with Frame(self.sim, name=fieldName, tf=timeFrame, load=True) as field_frame:
      toproject = field_frame.values

    if len(fluctuation) > 0:
      serie = TimeSerie(simulation=self.sim, name=fieldName, time_frames=avg_window, load=True)
      if 'tavg' in fluctuation:
        average = serie.get_time_average()
      elif 'yavg' in fluctuation:
        average = serie.get_y_average()
      toproject -= average
      if 'relative' in fluctuation:
        toproject = 100.0 * toproject / average

    field_fs = []
    field_RZ = []
    for i in range(len(self.rhoLim)):
      field_fs.append(self.fsprojs[i].project_field(toproject))
    
    for i in range(len(self.phiLim)):
      field_RZ.append(self.polprojs[i].project_field(toproject))
      
    return field_fs, field_RZ
  
  def data_to_pvmesh(self, X, Y, Z, field=None, indexing='ij', fieldName='field'):
      nx, ny = X.shape
      points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
      pvmesh = pv.StructuredGrid()
      pvmesh.points = points
      if indexing == 'ij':
        pvmesh.dimensions = (ny, nx, 1)
      else:
        pvmesh.dimensions = (nx, ny, 1)
      if field is not None:
        pvmesh[fieldName] = field.ravel()
      return pvmesh
    
  def init_pvmeshes(self, fieldName, timeFrame, fluctuation):
    
    field_fs, field_RZ = self.get_data(fieldName, timeFrame, fluctuation)
    
    pvmeshes = []
    phishift = self.pvphishift # required to have the right orientation
    for i in range(len(field_fs)):
      rcut = self.fsprojs[i].r0
      # Parametric equations for torus
      Rtor = self.sim.geom_param.R_rt(rcut,self.fsprojs[i].theta_fs)
      Xtor = Rtor * np.cos(self.fsprojs[i].phi_fs + phishift)
      Ytor = Rtor * np.sin(self.fsprojs[i].phi_fs + phishift)
      Ztor = self.sim.geom_param.Z_rt(rcut,self.fsprojs[i].theta_fs)
      
      pvmesh = self.data_to_pvmesh(Xtor, Ytor, Ztor, field_fs[i], indexing='ij', fieldName=fieldName)
      pvmeshes.append(pvmesh)
    
    for i in range(len(field_RZ)):
      # Parametric equations for the poloidal cross-section
      Xpol = np.cos(self.phiLim[i] + phishift) * self.polprojs[i].RIntN
      Ypol = np.sin(self.phiLim[i] + phishift) * self.polprojs[i].RIntN
      Zpol = self.polprojs[i].ZIntN
      
      pvmesh = self.data_to_pvmesh(Xpol, Ypol, Zpol, field_RZ[i], indexing='ij',  fieldName=fieldName)
      pvmeshes.append(pvmesh)
    
    return pvmeshes
    
  def draw_vessel(self, plotter, smooth_shading=True, opacity=0.2):

      # Draw the limiter
      RWidth = np.min(self.polprojs[0].Rlcfs) - np.min(self.polprojs[0].RIntN)
      R0 = np.min(self.polprojs[0].RIntN) * 0.99
      R1 = R0 + RWidth
      ZWidth = 0.015
      Z0 = self.polprojs[0].geom.Z_axis - 0.5*ZWidth
      Z1 = Z0 + ZWidth
      phi = np.linspace(0, 2*np.pi, 100)
      R, PHI = np.meshgrid([R0, R0, R1, R1, R0], phi, indexing='ij')
      Z, PHI = np.meshgrid([Z0, Z1, Z1, Z0, Z0], phi, indexing='ij')
      # R,Z draw the vessel contour at one angle phi, now define the toroidal surface
      Xtor = R * np.cos(PHI)
      Ytor = R * np.sin(PHI)
      Ztor = Z
      pvmesh = self.data_to_pvmesh(Xtor, Ytor, Ztor, indexing='ij')
      plotter.add_mesh(pvmesh, color='gray', opacity=1.0, show_scalar_bar=False, smooth_shading=smooth_shading)
   
      # # Draw the vessel
      Rvess = self.sim.geom_param.vesselData['R']
      Zvess = self.sim.geom_param.vesselData['Z']
      # we ensure tht the vessel encloses the plasma
      Rplasma_min = np.min(self.polprojs[0].RIntN)
      Rplasma_max = np.max(self.polprojs[0].RIntN)
      Rvessmin = np.min(Rvess)
      Rvessmax = np.max(Rvess)
      
      for i in range(len(Rvess)):
        if abs(Rvess[i] - Rplasma_min) < 0.05*Rplasma_min:
          Rvess[i] = min(Rvess[i], Rplasma_min)
        if abs(Rvess[i] - Rplasma_max) < 0.05*Rplasma_max:
          Rvess[i] = max(Rvess[i], Rplasma_max)
      
      phi = self.fsprojs[0].phi_fs + self.pvphishift
      R, PHI = np.meshgrid(Rvess, phi, indexing='ij')
      Z, PHI = np.meshgrid(Zvess, phi, indexing='ij')
      # R,Z draw the vessel contour at one angle phi, now define the toroidal surface
      Xtor = R * np.cos(PHI)
      Ytor = R * np.sin(PHI)
      Ztor = Z
      pvmesh = self.data_to_pvmesh(Xtor, Ytor, Ztor, indexing='ij')
      plotter.add_mesh(pvmesh, color='gray', opacity=opacity, show_scalar_bar=False, smooth_shading=smooth_shading)
      
      return plotter
    
  def plot(self, fieldName, timeFrame, filePrefix='', colorMap = '', fluctuation='', logScale = False,
           clim=None, colorbar=False, vessel=False, smooth_shading=False, lighting=False, jupyter_backend='none',
           vesselOpacity=0.2, viewVector = [1, 1, 0.2], camZoom = 2.0, imgSize=(800, 600), save_html=True):

    if isinstance(fluctuation, bool): fluctuation = 'yavg' if fluctuation else ''
    if isinstance(timeFrame, list): timeFrame = timeFrame[-1]
    if clim == []: clim = None
    colorMap = colorMap if colorMap else self.sim.fields_info[fieldName+'colormap']
    if fluctuation: colorMap = 'bwr'
    
    plotter = pv.Plotter(window_size=imgSize)
    
    pvmeshes = self.init_pvmeshes(fieldName, timeFrame, fluctuation=fluctuation)
    N_plas_mesh = len(pvmeshes)
    
    for i in range(N_plas_mesh):
      plotter.add_mesh(pvmeshes[i], scalars=fieldName, show_scalar_bar=colorbar, clim=clim, cmap=colorMap, 
                       opacity=1.0, smooth_shading=smooth_shading, lighting=lighting, log_scale=logScale,
                       label=fieldName)
    
    if vessel and self.sim.geom_param.vesselData is not None:
      plotter = self.draw_vessel(plotter, smooth_shading=smooth_shading, opacity=vesselOpacity)
      
    plotter.view_vector(vector=viewVector)  
    plotter.camera.Zoom(camZoom)
    
    plotter.show(screenshot=filePrefix+'torproj_'+fieldName+'.png', jupyter_backend=jupyter_backend)
    if save_html:
      plotter.export_html(filePrefix+'torproj_'+fieldName+'.html')

  def movie(self, fieldName, timeFrames, filePrefix='', colorMap = '', fluctuation='',
           clim=[], logScale=False, colorbar=False, vessel=False, smooth_shading=False, lighting=False,
           vesselOpacity=0.2, viewVector = [1, 1, 0.2], camZoom = 2.0, imgSize=(800, 600), fps=14):
    if smooth_shading: print('Warning: smooth_shading may create flickering in the movie. Idk why :/')
 
    if isinstance(fluctuation, bool): fluctuation = 'yavg' if fluctuation else ''
    if clim == []: clim = None
    colorMap = colorMap if colorMap else self.sim.fields_info[fieldName+'colormap']
    
    outFilename = filePrefix+'torproj_movie_'+fieldName+'.gif'
          
    plotter = pv.Plotter(window_size=imgSize)
    plotter.open_gif(outFilename, fps=fps)

    n = 0
    print_progress(n, len(timeFrames))
    
    # Create initial frame
    timeFrame = timeFrames[0]

    pvmeshes = self.init_pvmeshes(fieldName, timeFrame, fluctuation=fluctuation)
    N_plas_mesh = len(pvmeshes)
    
    for i in range(N_plas_mesh):
      plotter.add_mesh(pvmeshes[i], scalars=fieldName, show_scalar_bar=colorbar, clim=clim, cmap=colorMap, opacity=1.0,
                      smooth_shading=smooth_shading, lighting=lighting, log_scale=logScale)
    del pvmeshes
    
    if vessel and self.sim.geom_param.vesselData is not None:
      plotter = self.draw_vessel(plotter, smooth_shading=smooth_shading, opacity=vesselOpacity)
      
    plotter.view_vector(vector=viewVector)  
    plotter.camera.Zoom(camZoom)
    
    # plotter.render()
    plotter.write_frame()

    n += 1
    print_progress(n, len(timeFrames))
    
    for timeFrame in timeFrames[1:]:
      
      # Update the meshes with new data
      field_fs, field_RZ = self.get_data(fieldName, timeFrame, fluctuation)
      for i in range(len(field_fs)):
        plotter.meshes[i][fieldName] = field_fs[i].ravel()
      for i in range(len(field_RZ)):
        plotter.meshes[i+len(field_fs)][fieldName] = field_RZ[i].ravel()

      # plotter.render()
      plotter.write_frame()
      
      n += 1
      print_progress(n, len(timeFrames))
    
    sys.stdout.write("\n")
    
    plotter.close()
    print(f"Movie saved as {outFilename}")
    
    
def print_progress( n, total_frames):
  progress = f"Processed frames: {n}/{total_frames}... "
  sys.stdout.write("\r" + progress)
  sys.stdout.flush()