import numpy as np
import sys
import pyvista as pv
pv.global_theme.colorbar_orientation = 'vertical'

from ..classes import Frame, TimeSerie, Simulation
from .fluxsurfprojection import FluxSurfProjection
from .poloidalprojection import PoloidalProjection
from PIL import Image

colors = ['red', 'blue', 'green', 'yellow']

class TorusProjection:
  """
  Class to combine the poloidal and flux surface projections and plot field on the full torus.
  """
  txt_texts = []
  txt_positions = []
  txt_sizes = []
  txt_names = []
  imgSize = (800, 600)
  off_screen = False
  show_colorbar = True
  colorbar_args = {
    'title': None,
    'position_x': 0.025,     # Right side position (0-1)
    'position_y': 0.25,      # Top position (0-1) 
    'width': 0.05,
    'height': 0.5,
    'title_font_size': 15,
    'label_font_size': 14,
    'n_labels': 5,
    'fmt': '%.1e'           # Scientific notation format
  }
  show_vessel = True
  vessel_opacity = 0.2
  background_color = 'white'
  additional_text = None
  logo_path = None
  logo_position = (0.0, 0.01)
  logo_size = (0.2,0.2)
  logo_opacity = 0.6
  
  sim = None
  phiLim = [0, 3*np.pi/4]
  rhoLim = [4, -2]
  timeFrame0 = 0
  tref = 1e-3 # reference time in s (default 1 ms)
  t0 = 0.0 # time offset in units of tref (ms by default)
  
  def __init__(self):
    self.polprojs = []
    self.fsprojs = []
    self.pvphishift = np.pi/2 # required to have the right orientation in pyvista
    
  def setup(self, simulation : Simulation, timeFrame=0, Nint_polproj=16, Nint_fsproj=32, phiLim=[0, np.pi], rhoLim=[0.8,1.5], t0=0.0,
            intMethod='trapz32', figSize = (8,9), zExt=True, gridCheck=False, TSBC=True, imgSize=(800,600), tref=1e-3):
    self.sim = simulation
    self.phiLim = phiLim if isinstance(phiLim, list) else [phiLim]
    self.rhoLim = rhoLim if isinstance(rhoLim, list) else [rhoLim]
    self.t0 = t0
    self.timeFrame0 = timeFrame
    self.text_color = 'white' if self.background_color == 'black' else 'black'
    self.imgSize = imgSize
    self.tref = tref
    
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
    with Frame(self.sim, fieldname=fieldName, tf=timeFrame, load=True) as field_frame:
      toproject = field_frame.values
      time = field_frame.time * self.sim.normalization.dict['tscale'] / self.tref
    if len(fluctuation) > 0:
      serie = TimeSerie(simulation=self.sim, fieldname=fieldName, time_frames=avg_window, load=True)
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
    return field_fs, field_RZ, time
  
  def data_to_pvmesh(self, X, Y, Z, field=None, indexing='ij', fieldlabel='field'):
      nx, ny = X.shape
      points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
      pvmesh = pv.StructuredGrid()
      pvmesh.points = points
      if indexing == 'ij':
        pvmesh.dimensions = (ny, nx, 1)
      else:
        pvmesh.dimensions = (nx, ny, 1)
      if field is not None:
        pvmesh[fieldlabel] = field.ravel()
      return pvmesh
    
  def init_pvmeshes(self, fieldName, timeFrame, fluctuation):
    
    if fieldName in ['test']:
      field_fs = [np.ones_like(fsproj.theta_fs) for fsproj in self.fsprojs]
      field_RZ = [np.ones_like(polproj.RIntN) for polproj in self.polprojs]
      # Dummy values for test field
      scale = 0.0
      for i in range(len(field_fs)):
        field_fs[i] = field_fs[i] * scale
        scale = scale + 1.0
      for i in range(len(field_RZ)):
        field_RZ[i] = field_RZ[i] * scale
        scale = scale + 1.0
      time = 10.651 # dummy time for test field
    else:
      field_fs, field_RZ, time = self.get_data(fieldName, timeFrame, fluctuation)
    
    pvmeshes = []
    phishift = self.pvphishift # required to have the right orientation
    for i in range(len(field_fs)):
      rcut = self.fsprojs[i].r0
      
      # Parametric equations for torus
      Rtor = self.sim.geom_param.R_rt(rcut,self.fsprojs[i].theta_fs)
      Ztor = self.sim.geom_param.Z_rt(rcut,self.fsprojs[i].theta_fs)
      
      # we increase slightly the size of the poloidal cross-section to avoid gaps with flux surface projection
      factor = 0.01
      Rmin = np.min(Rtor)
      Rmax = np.max(Rtor)
      Delta = factor*Rmin 
      if self.fsprojs[i].r0 == max([fs.r0 for fs in self.fsprojs]):
        Delta = -Delta
      alpha = 1 + 2*Delta / (Rmax - Rmin)
      shift = (1 - alpha) * Rmin - Delta
      Rtor = alpha * Rtor + shift
      
      Zmin = np.min(Ztor)
      Zmax = np.max(Ztor)
      Delta = factor*Zmin
      if self.fsprojs[i].r0 == min([fs.r0 for fs in self.fsprojs]):
        Delta = -Delta
      alpha = 1 + 2*Delta / (Zmax - Zmin)
      shift = (1 - alpha) * Zmin - Delta
      Ztor = alpha * Ztor + shift
      
      # Cartesian coordinates
      Xtor = Rtor * np.cos(self.fsprojs[i].phi_fs + phishift)
      Ytor = Rtor * np.sin(self.fsprojs[i].phi_fs + phishift)
      
      fieldlabel = get_label(fieldName, fluctuation)
      pvmesh = self.data_to_pvmesh(Xtor, Ytor, Ztor, field_fs[i], indexing='ij', fieldlabel=fieldlabel)
      pvmeshes.append(pvmesh)
    
    for i in range(len(field_RZ)):
      # Parametric equations for the poloidal cross-section
      Xpol = np.cos(self.phiLim[i] + phishift) * self.polprojs[i].RIntN
      Ypol = np.sin(self.phiLim[i] + phishift) * self.polprojs[i].RIntN
      Zpol = self.polprojs[i].ZIntN
      
      pvmesh = self.data_to_pvmesh(Xpol, Ypol, Zpol, field_RZ[i], indexing='ij',  fieldlabel=fieldlabel)
      pvmeshes.append(pvmesh)
    
    return pvmeshes, time
    
  def draw_vessel(self, plotter, smooth_shading=True, opacity=0.2):

      # Draw the limiter
      RWidth = np.min(self.polprojs[0].Rlcfs) - np.min(self.polprojs[0].RIntN)
      R0 = np.min(self.polprojs[0].RIntN) * 0.95
      R1 = R0 + RWidth + 0.05 * np.min(self.polprojs[0].RIntN)
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
   
      # Draw the vessel
      Rvess = self.sim.geom_param.vesselData['R']
      Zvess = self.sim.geom_param.vesselData['Z']

      # scale slightly the vessel to avoid intersection with the plasma
      Rplas_min = self.polprojs[0].RIntN.min()
      Rplas_max = self.polprojs[0].RIntN.max()
      Rvess_min = Rvess.min()
      Rvess_max = Rvess.max()
      vpgap_min = Rvess_min - Rplas_min
      vpgap_max = Rvess_max - Rplas_max
      # check if there is clipping
      if vpgap_min > 0 or vpgap_max < 0:
        if vpgap_min > -vpgap_max:
          Delta = vpgap_min * 1.05
        else:
          Delta = -vpgap_max * 1.05
        alpha = 1 + 2*Delta / (Rvess_max - Rvess_min)
        shift = (1 - alpha) * Rvess_min - Delta
        Rvess = alpha * Rvess + shift
      
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
    
  def setup_frame(self, plotter: pv.Plotter, fieldName, timeFrame, fluctuation, logScale, colorMap, clim):
    fieldlabel = get_label(fieldName, fluctuation)
    pvmeshes, time = self.init_pvmeshes(fieldName, timeFrame, fluctuation=fluctuation)
    N_plas_mesh = len(pvmeshes)
    self.colorbar_args['title'] = fieldlabel
    for i in range(N_plas_mesh):
      if fieldName in ['test']:
        plotter.add_mesh(pvmeshes[i], scalars=fieldlabel, smooth_shading=False, lighting=False, cmap=colorMap,
                         log_scale=logScale, show_scalar_bar=self.show_colorbar, clim=clim, 
                         scalar_bar_args=self.colorbar_args)
      else:
        plotter.add_mesh(pvmeshes[i], scalars=fieldlabel, show_scalar_bar=self.show_colorbar, clim=clim, cmap=colorMap, 
                        opacity=1.0, smooth_shading=False, lighting=False, log_scale=logScale,
                         scalar_bar_args=self.colorbar_args)
    
    if self.show_vessel and self.sim.geom_param.vesselData is not None:
      plotter = self.draw_vessel(plotter, smooth_shading=False, opacity=self.vessel_opacity)
    
    plotter.set_background(self.background_color)

    # write time and bottom text
    if self.additional_text:
      self.add_text(self.additional_text['text'], self.additional_text['position'], 
                    self.additional_text['font_size'], self.additional_text['name'])
    else:
      self.add_text(self.sim.dischargeID, position='upper_left', 
                    font_size=10, name="dischargeID")
    self.add_text(f"t={(time + self.t0):5.3f} ms", position='lower_left', 
                  font_size=10, name="time_label")
    plotter = self.write_texts(plotter)
    
    if self.logo_path:
      plotter.add_logo_widget(self.logo_path, position=self.logo_position, 
                              size=self.logo_size, opacity=self.logo_opacity)

    return plotter
    
    
  def plot(self, fieldName, timeFrame, filePrefix='', colorMap = None, fluctuation='', 
           logScale = False, clim=None, jupyter_backend='none', save_html=False, cameraSettings=None):
    if isinstance(fluctuation, bool): fluctuation = 'yavg' if fluctuation else ''
    if isinstance(timeFrame, list): timeFrame = timeFrame[-1]
    if clim == []: clim = None
    if fieldName != 'test': colorMap = colorMap if colorMap else self.sim.fields_info[fieldName+'colormap']
    else: colorMap = ['red', 'blue', 'green', 'yellow']  # for test field, use a list of colors
    if fluctuation: colorMap = 'bwr'
    if cameraSettings == None: cameraSettings = self.sim.geom_param.camera_global

    plotter = pv.Plotter(window_size=self.imgSize, off_screen=self.off_screen)
    plotter = self.setup_frame(plotter, fieldName, timeFrame, fluctuation, logScale, colorMap, clim)
    
    cam = Camera(self.sim.geom_param, cameraSettings)
    plotter = cam.update_plotter(plotter)
    
    if fluctuation: fieldName = 'd' + fieldName
    if save_html:
      plotter.export_html(filePrefix+'torproj_'+fieldName+'.html')
      print(f"HTML saved as {filePrefix}torproj_{fieldName}.html")
    plotter.show(screenshot=filePrefix+'torproj_'+fieldName+'.png', jupyter_backend=jupyter_backend)
    print(f"Image saved as {filePrefix}torproj_{fieldName}.png")
    self.del_texts()  # Clear texts after plotting

  def movie(self, fieldName, timeFrames, filePrefix='', colorMap = 'inferno', fluctuation='',
           clim=[], logScale=False, fps=14, cameraPath=[], movie_type='gif'):
    if isinstance(fluctuation, bool): fluctuation = 'yavg' if fluctuation else ''
    if clim == []: clim = None
    if fieldName != 'test': colorMap = colorMap if colorMap else self.sim.fields_info[fieldName+'colormap']
    if fluctuation: 
      colorMap = 'bwr'
      outFilename = filePrefix+'torproj_movie_d'+fieldName
    else: 
      outFilename = filePrefix+'torproj_movie_'+fieldName
    fieldlabel = get_label(fieldName, fluctuation)
    
    plotter = pv.Plotter(window_size=self.imgSize, off_screen=True)
    if movie_type in ['gif','.gif']:
      outFilename += '.gif'
      plotter.open_gif(outFilename, fps=fps)
    else:
      outFilename += '.mp4'
      plotter.open_movie(outFilename, framerate=fps)

    cam = Camera(stops=cameraPath, geom=self.sim.geom_param, nframes=len(timeFrames))

    n = 0
    print_progress(n, len(timeFrames))
    timeFrame = timeFrames[0]
    
    plotter = self.setup_frame(plotter, fieldName, timeFrame, fluctuation, logScale, colorMap, clim)    
    
    plotter = cam.update_plotter(plotter)

    plotter.write_frame()
    
    self.clear_texts(plotter)  # Clear for next frame

    n += 1
    print_progress(n, len(timeFrames))
    
    for timeFrame in timeFrames[1:]:
      if fieldName not in ['test']:
        # Update the meshes with new data
        field_fs, field_RZ, time = self.get_data(fieldName, timeFrame, fluctuation)
        for i in range(len(field_fs)):
          plotter.meshes[i][fieldlabel] = field_fs[i].ravel()
        for i in range(len(field_RZ)):
          plotter.meshes[i+len(field_fs)][fieldlabel] = field_RZ[i].ravel()
      else:
        time = 10650 + n + self.t0 # dummy time for test field (in mus)
      cam.update_camera(n)
      plotter = cam.update_plotter(plotter, zoom=False)
      self.update_text("time_label", f"t={time + self.t0:5.3f} ms")
      plotter = self.write_texts(plotter)
      plotter.write_frame()
      self.clear_texts(plotter)
      n += 1
      print_progress(n, len(timeFrames))
    
    sys.stdout.write("\n")
    
    plotter.close()
    print(f"Movie saved as {outFilename}")
    self.del_texts()  # Clear texts after movie creation
    
  def add_text(self,text, position, font_size, name):
    self.txt_texts.append(text)
    self.txt_positions.append(position)
    self.txt_sizes.append(font_size)
    self.txt_names.append(name)
    
  def update_text(self, name, text):
    index = self.txt_names.index(name)
    self.txt_texts[index] = text

  def write_texts(self, plotter):
    for i in range(len(self.txt_texts)):
      plotter.add_text(self.txt_texts[i], position=self.txt_positions[i], 
                       font_size=self.txt_sizes[i], name=self.txt_names[i],
                       color=self.text_color)
    return plotter
    
  def clear_texts(self, plotter):
    for name in self.txt_names:
      plotter.remove_actor(name)
      
  def del_texts(self):
    self.txt_texts = []
    self.txt_positions = []
    self.txt_sizes = []
    self.txt_names = []
    
def print_progress( n, total_frames):
  progress = f"Processed frames: {n}/{total_frames}... "
  sys.stdout.write("\r" + progress)
  sys.stdout.flush()

class Camera:
  
  def __init__(self, geom, settings=[], stops=[], nframes=1):
    stops = stops if isinstance(stops, list) else [stops]
    
    if not settings:
      settings = {
        'position':(2.3, 2.3, 0.75),
        'looking_at':(0, 0, 0),
          'zoom': 1.0
      }
      
    self.update_camera = self.update_static  
       
    if stops:
      settings = stops[0]
      self.Ncp = len(stops)
      if self.Ncp > 1:
        self.update_camera = self.update_moving
        self.nframes = nframes
        self.checkpoints = [ i*nframes//(self.Ncp-1) for i in range(self.Ncp) ]
        self.icp = 0
        self.dpos = []
        self.dlook = []
        self.dzoom = []
        for i in range(self.Ncp-1):
          Nint = (self.checkpoints[i+1] - self.checkpoints[i])
          self.dpos.append([0, 0, 0])
          self.dlook.append([0, 0, 0])
          for k in range(3):
            self.dpos[i][k] = (stops[i+1]['position'][k] - stops[i]['position'][k]) / Nint * geom.R_LCFSmid
            self.dlook[i][k] = (stops[i+1]['looking_at'][k] - stops[i]['looking_at'][k]) / Nint * geom.R_LCFSmid
          self.dzoom.append((stops[i+1]['zoom'] - stops[i]['zoom']) / Nint)          
    
    self.position = settings['position']
    self.looking_at = settings['looking_at']
    self.view_up = (0,0,1)
    self.zoom = settings['zoom']
    # scale the lengths with the major radius
    self.position = [loc * geom.R_LCFSmid for loc in self.position]
    self.looking_at = [loc * geom.R_LCFSmid for loc in self.looking_at]
    
  def update_static(self, iframe):
    pass

  def update_moving(self, iframe):
    # check if we passed a checkpoint
    if iframe > self.checkpoints[self.icp+1]:
      self.icp = self.icp + 1
    
    # update the camera position
    for k in range(3):
      self.position[k] += self.dpos[self.icp][k]
      self.looking_at[k] += self.dlook[self.icp][k]
    self.zoom += self.dzoom[self.icp]

  def update_plotter(self, plotter, zoom=True):
    plotter.camera_position = [self.position, self.looking_at, self.view_up]
    if zoom: plotter.camera.Zoom(self.zoom)
    return plotter
  
  
def get_label(fieldName, fluctuation):
  if 'relative' in fluctuation:
    if fieldName == 'ni':
      return 'ni - <ni> [%]'
    elif fieldName == 'ne':
      return 'ne - <ne> [%]'
    elif fieldName == 'Te':
      return 'Te - <Te> [%]'
    elif fieldName == 'Ti':
      return 'Ti - <Ti> [%]'
    elif fieldName == 'phi':
      return 'phi - <phi> [%]'
    elif fieldName == 'pi':
      return 'pi - <pi> [%]'
    elif fieldName == 'pe':
      return 'pe - <pe> [%]'
    elif fieldName == 'test':
      return 'test - <test> [%]'
    else:
      print(f"No label implemented for fluct. {fieldName} yet.")
      return fieldName
  else:
    if fieldName == 'ni':
      return 'ni [m-3]'
    elif fieldName == 'ne':
      return 'ne [m-3]'
    elif fieldName == 'Te':
      return 'Te [eV]'
    elif fieldName == 'Ti':
      return 'Ti [eV]'
    elif fieldName == 'phi':
      return 'phi [V]'
    elif fieldName == 'pi':
      return 'pi [Pa]'
    elif fieldName == 'pe':
      return 'pe [Pa]'
    elif fieldName == 'test':
      return 'test [a.u.]'
    else:
      print(f"No label implemented for {fieldName} yet.")
      return fieldName