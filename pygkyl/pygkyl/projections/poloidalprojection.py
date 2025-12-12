import numpy as np
import matplotlib.pyplot as plt
import os, sys
import postgkyl as pg
from copy import deepcopy

from scipy.interpolate import pchip_interpolate
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Rectangle

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib import ticker
from matplotlib import colors
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from ..classes import Frame, TimeSerie, Simulation
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
    self.ixLCFS_C = None
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
    self.insets = []
    self.phiTor = 0
    self.alpha_rz_phi0 = None
    self.dimsI = 0
    self.meshC = None
    self.gridCheck = False
    self.zExt = True
    self.TSBC = True
    self.dpi = 150
    
  def setup(self, simulation: Simulation, timeFrame=0, nzInterp=16, phiTor=0, Rlim = [], rholim = [],
            intMethod='trapz32',figSize = (8,9), zExt=True, gridCheck=False, TSBC=True, dpi=150,
            nodefilename = ''):

    # Store simulation and a link to geometry objects
    self.sim = simulation
    self.geom = simulation.geom_param
    self.nzInterp = nzInterp
    self.figSize = figSize
    self.dpi = dpi
    self.timeFrame0 = timeFrame
    self.nodefilename = nodefilename

    if self.sim.polprojInsets is not None:
      self.insets = deepcopy(self.sim.polprojInsets)
    else:
      self.insets = [Inset()]
    self.phiTor = phiTor
    self.gridCheck = gridCheck
    self.TSBC = TSBC
    self.zExt = True if TSBC else zExt
    
    # Load a frame to get the grid
    if len(simulation.available_frames['field']) > 0:
      fieldName = 'phi'
    else:
      fieldName = 'ni'
      
    field_frame = Frame(self.sim, fieldname=fieldName, tf=timeFrame, load=True)
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

    radius = self.geom.r0
    if self.ndim == 3:
      self.LyC = self.meshC[1][-1] - self.meshC[1][0] # length in the y direction
      # Here we scale y to match the integer fraction condition in the toroidal direction
      Ntor0 = 2*np.pi * radius / self.geom.q0 / self.LyC # this may not be an integer
      Ntor = max(1,int(np.ceil(Ntor0)))
      self.LyC = 2*np.pi * radius / (self.geom.q0 * Ntor)
      self.meshC[1] = self.meshC[1] * (self.LyC / (self.meshC[1][-1] - self.meshC[1][0]))
    else:
      self.LyC = 2.*np.pi*radius/self.geom.q0
      
    # Minimal toroidal mode number (must be used in the toroidal rotation, n_0 in Lapillone thesis 2009)
    self.n0 = 2*np.pi * self.geom.Cy/ self.LyC
    
    #.Precompute grids and arrays needed in transforming/plotting data
    field = np.squeeze(field_frame.values)
    field_ky = np.fft.rfft(field, axis=1, norm="forward")
    self.kyDimsC = field_ky.shape

    #.Extend along z by in each direction by applying twist-shift BCs in the 
    #.closed-flux region, and just copying the last values (along z) in the SOL.
    # Number of points for the z interpolation (BIG)
    self.nzI = nzInterp*self.dimsC[-1]
  
    zGrid = self.meshC[-1]
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
    phi0 = 0.0
    #.Compute alpha(r,z,phi=0) which is independent of y.
    self.alpha_rz_phi0 = np.zeros([self.dimsC[0],self.nzI])
    for ix in range(self.dimsC[0]): # we do it point by point because we integrate over r for each point
      dPsidr = self.geom.dPsidr(self.geom.r_x(self.meshC[0][ix]),method=method)
      for iz in range(self.nzI):
          self.alpha_rz_phi0[ix,iz]  = \
            self.geom.alpha0(self.geom.r_x(self.meshC[0][ix]),self.zgridI[iz], phi0, method=method)\
              /dPsidr

  def compute_xyz2RZ(self,phiTor=0.0):
    phiTor += np.pi # To match the obmp with varphi=0
    # this can be a very big array
    self.xyz2RZ = np.zeros([self.dimsC[0],2*self.kyDimsC[1],self.nzI], dtype=np.cdouble)
    for k in range(self.kyDimsC[1]):
        for iz in range(self.nzI):
            shift = self.n0*(self.alpha_rz_phi0[:,iz]) + phiTor
            self.xyz2RZ[:,+k,iz]  = np.exp(1j*k*shift)
            #.Negative ky's.
            self.xyz2RZ[:,-k,iz] = np.conj(self.xyz2RZ[:,+k,iz])
            
  def compute_nodal_coordinates(self):
    #.Compute R(x,z) and Z(x,z)
    xxI, zzI = math_tools.custom_meshgrid(self.meshC[0],self.zgridI)
    self.dimsI = np.shape(xxI) # interpolation plane dimensions (R,Z)  

    # Get the (R,Z) grid (Rint,Zint) according to the interpolated z-grid
    if self.sim.geom_param.geom_type == 'Miller':
      rrI = self.geom.r_x(xxI) # Call to analytic geometry functions (Miller geometry)
      Rint = self.geom.R_rt(rrI,zzI) # Call to analytic geometry functions (Miller geometry)
      Zint = self.geom.Z_rt(rrI,zzI) # Call to analytic geometry functions (Miller geometry)
      del rrI
    
    elif self.sim.geom_param.geom_type in ['efit', 'Millernodal']:
      if len(self.nodefilename) > 0 : 
        nodefile = self.nodefilename
        if not os.path.isfile(nodefile): 
          ValueError("File name for nodes {nodefile} is not found.")
      else:
        simName = self.sim.data_param.fileprefix
        nodefile = simName+"-nodes_intZ.gkyl"
        if not os.path.isfile(nodefile): nodefile =  simName+"-nodes.gkyl"
      nodalData = pg.GData(nodefile)
      nodalVals = nodalData.get_values()
      alpha_idx = 0
      if self.sim.geom_param.geom_type == 'efit':
        R = nodalVals[:, alpha_idx, :, 0]
        Z = nodalVals[:, alpha_idx, :, 1]
        Phi = nodalVals[:, alpha_idx, :, 2] 
      elif self.sim.geom_param.geom_type == 'Millernodal':
        X = nodalVals[:, alpha_idx, :, 0]
        Y = nodalVals[:, alpha_idx, :, 1]
        Z = nodalVals[:, alpha_idx, :, 2] + self.sim.geom_param.Z_axis
        R = np.sqrt(X**2 + Y**2)  # R = sqrt(x^2 + y^2)
      nodalGridTemp = nodalData.get_grid()   # contains one more element than number of nodes.
      nodalGrid = []
      for d in range(0,len(nodalGridTemp)):
          nodalGrid.append( np.linspace(nodalGridTemp[d][0], nodalGridTemp[d][-1], len(nodalGridTemp[d])-1) )

      RInterpolator = RegularGridInterpolator((nodalGrid[0], nodalGrid[2]), R)
      ZInterpolator = RegularGridInterpolator((nodalGrid[0], nodalGrid[2]), Z)
      PhiInterpolator = RegularGridInterpolator((nodalGrid[0], nodalGrid[2]), Phi)

      Rint = RInterpolator((xxI, zzI))
      Zint = ZInterpolator((xxI, zzI))
      Phiint = PhiInterpolator((xxI, zzI))

      if self.sim.geom_param.geom_type == 'efit':
        self.alpha_rz_phi0 = -self.gridsN[1][alpha_idx] - Phiint # Overwrite the results of compute_alpha
        phiTor = np.pi
        for k in range(self.kyDimsC[1]): # Overwrite the results of compute_xyz2RZ
          for iz in range(self.nzI):
            shift = -2*np.pi*self.alpha_rz_phi0[:,iz]/self.LyC + phiTor
            self.xyz2RZ[:,+k,iz]  = np.exp(1j*k*shift)
            #.Negative ky's.
            self.xyz2RZ[:,-k,iz] = np.conj(self.xyz2RZ[:,+k,iz])

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
    This is done by multiplying the projection by exp(1j*k*dphi) for each k,
    which introduces a phase shift in the Fourier space.
    '''
    dphi *= self.n0 # Take into account the minimal toroidal mode number
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
      
    # select the radial region of interest
    field = field[self.ix0:self.ix1,:,:]
      
    #.Approach: FFT along y, then follow a procedure similar to that in pseudospectral
    #.codes (e.g. GENE, see Xavier Lapillonne's PhD thesis 2010, section 3.2.2, page 55).
    field_ky = np.fft.rfft(field, axis=1, norm="forward")
    # field_ky = field_ky[self.ix0:self.ix1,:,:] # select the radial region of interest
    
    if self.TSBC:
      #.Apply twist-shift BCs in the closed-flux region.
      if self.ixLCFS_C is None: icore_end = self.dimsC[0] # SOL only
      else: icore_end = self.ixLCFS_C
      xGridCore = self.meshC[0][:icore_end] # x grid on in the core region
      bcPhaseShift = 2.0*np.pi * self.n0 * self.geom.qprofile_R(self.geom.R_x(xGridCore))
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
    field_frame = Frame(self.sim, fieldname=fieldName, tf=timeFrame, load=True)
    if self.zExt and not self.TSBC :
      evalDGfunc = field_frame.eval_DG_proj
    else:
      evalDGfunc = None
    field_RZ = self.project_field(field_frame.values, evalDGfunc=evalDGfunc)
    return field_RZ, self.RIntN, self.ZIntN
  
  def plot(self, fieldName, timeFrame, outFilename='', colorMap = '', fluctuation='',
           xlim=[],ylim=[],clim=[],climInset=[], colorScale='linear', logScaleFloor = 1e-3, favg = None,
           shading='auto', average='',show_LCFS=True, show_limiter=True, show_inset=True, show_vessel=False,
           cmap_period = 1):
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
    colorMap = fig_tools.check_colormap(colorMap)
    if isinstance(fluctuation, bool): fluctuation = 'tavg' if fluctuation else ''
    if isinstance(timeFrame, list):
      avg_window = timeFrame.copy()
      timeFrame = timeFrame[-1]
    else:
      avg_window = [timeFrame]

    with Frame(self.sim, fieldname=fieldName, tf=timeFrame, load=True) as field_frame:
      time = field_frame.time
      vsymbol = field_frame.vsymbol
      vunits = field_frame.vunits
      toproject = field_frame.values
      frame_info = Frame(self.sim, fieldname=fieldName, tf=timeFrame, load=False)

    if (len(fluctuation) > 0) or (len(average) > 0):
      serie = TimeSerie(simulation=self.sim, fieldname=fieldName, time_frames=avg_window, load=True)
      if fluctuation:
        serie.fluctuations(fluctuationType=fluctuation,avg_array=favg)
        colorMap = colorMap if colorMap else 'bwr'
      elif average:
        serie.average(averageType=average)
        colorMap = colorMap if colorMap else self.sim.normalization.dict[fieldName+'colormap']
      toproject = serie.frames[-1].values
      vsymbol = serie.vsymbol
      vunits = serie.vunits
    else:
      colorMap = colorMap if colorMap else self.sim.normalization.dict[fieldName+'colormap']

    field_RZ = self.project_field(toproject, frame_info.eval_DG_proj)

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
    fig1a     = plt.figure(figsize=self.figSize, dpi=self.dpi)
    ax1a      = list()
    for i in range(len(ax1aPos)):
        ax1a.append(fig1a.add_axes(ax1aPos[i]))
    cbar_ax1a = fig1a.add_axes(cax1aPos)
    
    hpl1a = list()
    pcm1 = ax1a[0].pcolormesh(self.RIntN, self.ZIntN, field_RZ, shading=shading,cmap=colorMap,
                              vmin=fldMin,vmax=fldMax)
    hpl1a.append(pcm1)
    
    # Handle periodic colormap
    if cmap_period > 1:
        # Create a periodic colormap by repeating the original colormap
        original_cmap = cm.get_cmap(colorMap)
        clrs = original_cmap(np.linspace(0, 1, 512))
        # Create repeated colors with reversed alternate periods for continuity
        repeated_colors = []
        for i in range(cmap_period):
            if i % 2 == 1:
                repeated_colors.append(clrs[::-1])
            else:
                repeated_colors.append(clrs)  # Reverse for continuity
        repeated_colors = np.vstack(repeated_colors)
        # Create new colormap from repeated colors
        periodic_cmap = mcolors.ListedColormap(repeated_colors)
        pcm1.set_cmap(periodic_cmap)

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
    if show_LCFS:
      # SAZ - Making LCFS line larger
      #ax1a[0].plot(self.Rlcfs,self.Zlcfs,linewidth=1.5,linestyle='--',color=lcfColor,alpha=.8)
      ax1a[0].plot(self.Rlcfs,self.Zlcfs,linewidth=3,linestyle='--',color=lcfColor,alpha=1.0)
      LCFSinset = [self.Rlcfs,self.Zlcfs,lcfColor]
    else:
      LCFSinset = []
      
    #.Plot the limiter
    if show_limiter:
      xWidth = np.min(self.Rlcfs) - np.min(self.RIntN)
      xCorner = np.min(self.RIntN)
      yWidth = 0.01
      yCorner = self.geom.Z_axis - 0.5*yWidth
      ax1a[0].add_patch(Rectangle((xCorner,yCorner),xWidth,yWidth,color='gray'))
      limiter = [yWidth]
    else:
      limiter = []
      
    #.Plot the vessel contours
    if show_vessel:
      if self.sim.geom_param.vessel_data is not None:
        Rvess = self.sim.geom_param.vessel_data['R']
        Zvess = self.sim.geom_param.vessel_data['Z']
        ax1a[0].plot(Rvess,Zvess,linewidth=1.0,linestyle='-',color='black',alpha=.8)

    if show_inset:
      for inset in self.insets:
        inset.add_inset(fig1a, ax1a[0], self.RIntN, self.ZIntN, field_RZ, colorMap,
                        colorScale, minSOL, maxSOL, climInset, logScaleFloor, shading,
                        LCFS=LCFSinset, limiter=limiter)      

    ax1a[0].set_aspect('equal',adjustable='datalim')

    if xlim: ax1a[0].set_xlim(xlim)
    if ylim: ax1a[0].set_ylim(ylim)
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

  def movie(self, fieldName, timeFrames=[], moviePrefix='', colorMap='',
          xlim=[],ylim=[],clim=[],climInset=[], colorScale='linear', logScaleFloor = 1e-3,
          pilLoop=0, pilOptimize=False, pilDuration=100, fluctuation='', timeFrame=[],
          rmFrames=True, show_LCFS=True, show_limiter=True, show_inset=True):
      colorMap = fig_tools.check_colormap(colorMap)    
      # Naming
      movieName = fieldName+'_RZ'
      if len(fluctuation)>0: movieName = 'd' + movieName
      if colorScale == 'log': movieName = 'log'+movieName
      movieName = moviePrefix + movieName
      movieName+='_xlim_%2.2d_%2.2d'%(xlim[0],xlim[1]) if xlim else ''
      movieName+='_ylim_%2.2d_%2.2d'%(ylim[0],ylim[1]) if ylim else ''
      
      # Create a temporary folder to store the movie frames (random name)
      movDirTmp = movieName+'_frames_tmp_%4d'%np.random.randint(9999)
      os.makedirs(movDirTmp, exist_ok=True)   
      
      timeFrames = timeFrame if not timeFrames else timeFrames

      if 'tavg' in fluctuation:
        with TimeSerie(simulation=self.sim, fieldname=fieldName, time_frames=timeFrames, load=True) \
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
                          colorMap = colorMap, colorScale=colorScale, logScaleFloor=logScaleFloor,
                          xlim=xlim, ylim=ylim, clim=clim, climInset=climInset,
                          show_LCFS=show_LCFS, show_limiter=show_limiter, show_inset=show_inset,
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
      
  def reset_insets(self):
    if self.sim.polprojInsets is not None:
      self.insets = deepcopy(self.sim.polprojInsets)
    else:
      self.insets = []
      
  def set_inset(self, index=0, **kwargs):
    self.insets[index].set(**kwargs)
      
  def add_inset(self, **kwargs):
    self.insets.append(Inset(**kwargs))
          
class Inset:
  
  """
  Class to add an inset to a plot.
  """
  def __init__(self, zoom=1.5, 
               zoomLoc='lower left', 
               lowerCornerRelPos=(0.35,0.35), 
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
    self.lowerCornerRelPos = lowerCornerRelPos
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
    
  def set(self, **kwargs):
    for key, value in kwargs.items():
      if hasattr(self, key):
        setattr(self, key, value)
      
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
