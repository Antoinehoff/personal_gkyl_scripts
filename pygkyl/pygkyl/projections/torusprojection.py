import numpy as np
import sys
import pyvista as pv

from ..classes import Frame, TimeSerie
from .fluxsurfprojection import FluxSurfProjection
from .poloidalprojection import PoloidalProjection

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
           vesselOpacity=0.2, viewVector = [1, 1, 0.2], camZoom = 2.0, imgSize=(800, 600), save_html=False):

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
    
    if fluctuation: fieldName = 'd' + fieldName
    if save_html:
      plotter.export_html(filePrefix+'torproj_'+fieldName+'.html')
    plotter.show(screenshot=filePrefix+'torproj_'+fieldName+'.png', jupyter_backend=jupyter_backend)

  def movie(self, fieldName, timeFrames, filePrefix='', colorMap = '', fluctuation='',
           clim=[], logScale=False, colorbar=False, vessel=False, smooth_shading=False, lighting=False,
           vesselOpacity=0.2, viewVector = [1, 1, 0.2], camZoom = 2.0, imgSize=(800, 600), fps=14):
    if smooth_shading: print('Warning: smooth_shading may create flickering in the movie. Idk why :/')
 
    if isinstance(fluctuation, bool): fluctuation = 'yavg' if fluctuation else ''
    if clim == []: clim = None
    colorMap = colorMap if colorMap else self.sim.fields_info[fieldName+'colormap']
    if fluctuation: 
      colorMap = 'bwr'
      outFilename = filePrefix+'torproj_movie_d'+fieldName+'.gif'
    else: 
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