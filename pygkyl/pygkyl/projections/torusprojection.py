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
  Class to handle torus projections using PyVista for 3D rendering.

  Responsibilities:
    - Build 3D PyVista structured grids for flux-surface (toroidal) and poloidal cuts.
    - Fetch, preprocess, and (optionally) fluctuate physical field data.
    - Render still images or time-dependent movies with camera control.
    - Overlay vessel geometry, colorbars, text labels, and logos.

  Main Workflow:
    1. Call setup(...) once to initialize projection helper objects.
    2. Use plot(...) for a single frame or movie(...) for animations.

  Attributes:
    polprojs (list[PoloidalProjection]): Poloidal projection helpers (one per phi in phiLim).
    fsprojs (list[FluxSurfProjection]): Flux-surface projection helpers (one per rho in rhoLim).
    pvphishift (float): Phase shift for correct orientation in PyVista.
    text_color (str): Auto-selected contrasting color for text (based on background).
    txt_texts (list[str]): Text strings queued for rendering.
    txt_positions (list[str|tuple]): Positions (PyVista keywords or normalized coords).
    txt_sizes (list[int]): Font sizes for each queued text.
    txt_names (list[str]): Unique names/keys for the text actors (used for updates/removal).
    imgSize (tuple[int,int]): Render window size in pixels.
    off_screen (bool): Enable off-screen rendering (for batch/movie generation).
    show_colorbar (bool): Toggle scalar bar display.
    colorbar_args (dict): PyVista scalar bar customization arguments.
    show_vessel (bool): Toggle vessel geometry rendering.
    vessel_opacity (float): Base opacity for vessel surfaces.
    vessel_opacity_inner (float): Reduced opacity for inner vessel regions.
    vessel_lighting (bool): Enable lighting on vessel mesh.
    vessel_smooth_shading (bool): Use smooth shading on vessel mesh.
    vessel_pbr (bool): Enable Physically Based Rendering for vessel.
    vessel_metallic (float): PBR metallic coefficient.
    vessel_roughness (float): PBR roughness coefficient.
    vessel_split_sharp_edges (bool): Split sharp edges to improve shading.
    vessel_ntor (int): Toroidal resolution for vessel surface construction.
    vessel_rgb (list[int]): Base RGB color of vessel (0–255 per channel).
    mesh_lighting (bool): Enable lighting for plasma field meshes.
    mesh_smooth_shading (bool): Smooth shading for plasma meshes.
    mesh_specular (float): Specular intensity for plasma meshes.
    mesh_specular_power (float): Specular exponent for plasma meshes.
    background_color (str): Scene background color.
    additional_text (dict|None): Extra text overlay: {'text': str, 'position': str|tuple, 'name': str}.
    font_size (int): Base font size for text overlays.
    logo_path (str|None): Path to image used as a logo overlay.
    logo_position (tuple[float,float]): Normalized (x,y) logo anchor position.
    logo_size (tuple[float,float]): Normalized (w,h) logo size.
    logo_opacity (float): Logo transparency (0–1).
    sim (Simulation|None): Active simulation object (set in setup()).
    phiLim (list[float]): Toroidal angles (rad) for poloidal sections.
    rhoLim (list[float]): Radial values for flux-surface extractions.
    timeFrame0 (int): Initial time frame index stored at setup.
    tref (float): Reference physical time scale (seconds) for label normalization.
    t0 (float): Time offset added to displayed time (in units of tref).

  Key Methods:
    setup(simulation, ...):
      Initialize projection helper objects and rendering parameters.

    plot(fieldName, timeFrame, ...):
      Produce a single image (and optional HTML export).

    movie(fieldName, timeFrames, ...):
      Generate animated GIF or MP4 with optional camera path interpolation.

  Notes:
    - For fluctuation strings, accepted substrings: 'tavg', 'yavg' + optional 'relative'.
    - For test rendering, fieldName='test' builds synthetic scalar fields.
    - Camera motion is handled by Camera class via user-defined keyframes (stops).
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
    'fmt': '%.1e',           # Scientific notation format
    'shadow': True,
    'vertical': True,
  }
  
  # Default vessel rendering parameters
  show_vessel = True
  vessel_opacity = 1.0
  vessel_opacity_inner = 0.3
  vessel_lighting = True
  vessel_smooth_shading = True
  vessel_pbr = True
  vessel_metallic = 0.8
  vessel_roughness = 0.3
  vessel_split_sharp_edges = True
  vessel_ntor = 256
  vessel_rgb = [128, 128, 128]

  # Default mesh rendering parameters
  mesh_lighting = False
  mesh_smooth_shading = False
  mesh_specular = 1.0
  mesh_specular_power = 128
  
  background_color = 'white'
  additional_text = None
  font_size = 12
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
  time_pos = 'lower_left'
  
  def __init__(self):
    self.polprojs = []
    self.fsprojs = []
    self.pvphishift = np.pi/2 # required to have the right orientation in pyvista
    
  def setup(self, simulation : Simulation, timeFrame=0, Nint_polproj=16, Nint_fsproj=32, phiLim=[0, np.pi], rhoLim=[0.8,1.5], t0=0.0,
            intMethod='trapz32', figSize = (8,9), zExt=True, gridCheck=False, TSBC=True, imgSize=(800,600), tref=1e-3, font_size=12):
    """
    Configure toroidal, poloidal, and flux-surface projection objects.

    Parameters:
      simulation (Simulation): Simulation object with geometry, normalization, and field metadata.
      timeFrame (int): Initial time frame index to load. Default 0.
      Nint_polproj (int): Number of interpolation points along poloidal (R,Z) direction for poloidal cuts. Higher -> smoother. Default 16.
      Nint_fsproj (int): Number of interpolation points along theta (poloidal angle) for flux-surface projections. Default 32.
      phiLim (list[float] | list): List of toroidal angles (rad) at which poloidal cross-sections are taken. If single value, converted to list. Default [0, np.pi].
      rhoLim (list[float] | list): List of normalized radii (or minor radius values) defining flux surfaces to project. If single value, converted to list. Default [0.8, 1.5].
      t0 (float): Time offset (in units of tref) added to displayed time labels. Default 0.0.
      intMethod (str): Integration / interpolation method key passed to Poloidal/Flux surface projection objects (e.g., 'trapz32'). Default 'trapz32'.
      figSize (tuple[int,int]): Matplotlib-like figure size used internally by projection helpers (not the 3D window). Default (8, 9).
      zExt (bool): If True, extend/interpolate poloidal grid in Z for smoother surfaces. Default True.
      gridCheck (bool): If True, perform diagnostic checks/prints on grids. Default False.
      TSBC (bool): If True, apply time series boundary condition handling in projections (implementation dependent). Default True.
      imgSize (tuple[int,int]): 3D rendering window size in pixels (width, height). Default (800, 600).
      tref (float): Reference physical time scale (seconds) used to normalize simulation time to milliseconds in labels. Default 1e-3 (1 ms).
      font_size (int): Base font size for overlaid text annotations. Default 12.

    Side Effects:
      Initializes self.polprojs and self.fsprojs lists.
      Sets timing, geometry, and rendering related attributes.
      Prepares colorbar sizing relative to font size.

    Returns:
      None
    """
    self.sim = simulation
    self.phiLim = phiLim if isinstance(phiLim, list) else [phiLim]
    self.rhoLim = rhoLim if isinstance(rhoLim, list) else [rhoLim]
    self.t0 = t0
    self.timeFrame0 = timeFrame
    self.text_color = 'white' if self.background_color == 'black' else 'black'
    self.imgSize = imgSize
    self.tref = tref
    self.font_size = font_size
    self.colorbar_args['label_font_size'] = round(1.4*font_size)
    self.colorbar_args['title_font_size'] = round(1.5*font_size)
    
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
    
  def draw_vessel(self, plotter : pv.Plotter):

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
      plotter.add_mesh(pvmesh, color='gray', opacity=1.0, show_scalar_bar=False)
   
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
      ## Interpolate the phi values on a larger grid to improve the vessel surface quality
      phi = np.linspace(phi[0], phi[-1], self.vessel_ntor)
      R, PHI = np.meshgrid(Rvess, phi, indexing='ij')
      Z, PHI = np.meshgrid(Zvess, phi, indexing='ij')
      # R,Z draw the vessel contour at one angle phi, now define the toroidal surface
      Xtor = R * np.cos(PHI)
      Ytor = R * np.sin(PHI)
      Ztor = Z
      pvmesh = self.data_to_pvmesh(Xtor, Ytor, Ztor, indexing='ij')
      
      # Set a lower opacity value for the interior parts of the vessel
      opacity_values = np.ones_like(Xtor) * self.vessel_opacity
      min_R = np.min(R)
      opacity_values[R < 1.05*min_R] = self.vessel_opacity_inner # Or whatever lower opacity you want
      gray_rgb = np.array(self.vessel_rgb)
      rgba_colors = np.tile(gray_rgb, (pvmesh.n_points, 1))
      alpha_channel = (opacity_values.ravel() * 255).astype(np.uint8)
      rgba_colors = np.c_[rgba_colors, alpha_channel]
      pvmesh['rgba_colors'] = rgba_colors

      plotter.add_mesh(pvmesh, scalars='rgba_colors', rgba=True,
                       show_scalar_bar=False, 
                       lighting=self.vessel_lighting, smooth_shading=self.vessel_smooth_shading, split_sharp_edges=self.vessel_split_sharp_edges, 
                       pbr=self.vessel_pbr, metallic=self.vessel_metallic, roughness=self.vessel_roughness)
      
      return plotter
    
  def setup_frame(self, plotter: pv.Plotter, fieldName, timeFrame, fluctuation, logScale, colorMap, clim):
    fieldlabel = get_label(fieldName, fluctuation)
    pvmeshes, time = self.init_pvmeshes(fieldName, timeFrame, fluctuation=fluctuation)
    N_plas_mesh = len(pvmeshes)
    self.colorbar_args['title'] = fieldlabel
    for i in range(N_plas_mesh):
      if fieldName in ['test']:
        plotter.add_mesh(pvmeshes[i], scalars=fieldlabel, smooth_shading=self.mesh_smooth_shading, 
                         lighting=self.mesh_lighting, cmap=colorMap,
                         log_scale=logScale, show_scalar_bar=self.show_colorbar, clim=clim, 
                         scalar_bar_args=self.colorbar_args,
                         specular=self.mesh_specular, specular_power=self.mesh_specular_power)
      else:
        plotter.add_mesh(pvmeshes[i], scalars=fieldlabel, show_scalar_bar=self.show_colorbar, clim=clim, cmap=colorMap, 
                         smooth_shading=self.mesh_smooth_shading, lighting=self.mesh_lighting, log_scale=logScale,
                         scalar_bar_args=self.colorbar_args,
                         specular=self.mesh_specular, specular_power=self.mesh_specular_power)
    if self.show_vessel and self.sim.geom_param.vesselData is not None:
      plotter = self.draw_vessel(plotter)
    
    plotter.set_background(self.background_color)

    # write time and bottom text
    if self.additional_text:
      self.add_text(self.additional_text['text'], self.additional_text['position'], 
                    self.font_size, self.additional_text['name'])
    else:
      self.add_text(self.sim.dischargeID, position='upper_left', 
                    font_size=self.font_size, name="dischargeID")
    if self.additional_text['position'] == 'lower_left':
      time_pos = 'lower_right'
    else:
      time_pos = self.time_pos
    self.add_text(f"t={(time + self.t0):5.3f} ms", position=time_pos, 
                  font_size=self.font_size, name="time_label")
    plotter = self.write_texts(plotter)
    
    if self.logo_path:
      plotter.add_logo_widget(self.logo_path, position=self.logo_position, 
                              size=self.logo_size, opacity=self.logo_opacity)

    return plotter
    
    
  def plot(self, fieldName, timeFrame, filePrefix='', colorMap = None, fluctuation='', 
           logScale = False, clim=None, jupyter_backend='none', save_html=False, cameraSettings=None):
    """
    Render a single torus projection frame and save a PNG (and optional HTML).

    Parameters:
      fieldName (str): Name of the field to plot. Special value 'test' builds synthetic data.
      timeFrame (int | list[int]): Time frame index or list (for averaging window when using fluctuations).
      filePrefix (str, optional): Prefix added to output filenames. Default ''.
      colorMap (str | list | None, optional): Colormap name. If None, pulled from sim.fields_info. For 'test', a list of colors is used.
      fluctuation (str | bool, optional): 
      '' (default) for raw field.
      'tavg', 'tavgrelative' for (relative) time-average fluctuation.
      'yavg', 'yavgrelative' for (relative) y-average fluctuation.
      True is interpreted as 'yavg'.
      logScale (bool, optional): Apply logarithmic color scaling. Default False.
      clim (tuple[float, float] | None, optional): Color limits. None -> auto. Empty list [] also treated as auto.
      jupyter_backend (str, optional): PyVista Jupyter backend ('none','static','pythreejs','client'). Default 'none'.
      save_html (bool, optional): If True, also exports an interactive HTML file.
      cameraSettings (dict | None, optional): Dict with keys: 'position', 'looking_at', 'zoom'. If None, uses simulation default.

    Side Effects:
      Writes PNG to: {filePrefix}torproj_<[d]fieldName>.png
      Optionally writes HTML to: {filePrefix}torproj_<[d]fieldName>.html
      Modifies internal text overlay lists (cleared at end).

    Notes:
      If fluctuation specified, output filename is prefixed with 'd' (e.g., dTe).
      Relative fluctuations are expressed in percent.
      When timeFrame is a list, the last element is plotted; the full list is used for averaging.

    Returns:
      None
    """
    if isinstance(fluctuation, bool): fluctuation = 'yavg' if fluctuation else ''
    if isinstance(timeFrame, list): timeFrame = timeFrame[-1]
    if clim == []: clim = None
    if fieldName != 'test': colorMap = colorMap if colorMap else self.sim.fields_info[fieldName+'colormap']
    else: colorMap = ['red', 'blue', 'green', 'yellow']  # for test field, use a list of colors
    if fluctuation: colorMap = 'bwr'
    if cameraSettings == None: cameraSettings = self.sim.geom_param.camera_global

    plotter = pv.Plotter(window_size=self.imgSize, off_screen=self.off_screen)
    plotter.enable_anti_aliasing()
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
    """
    Create an animated torus projection (GIF or MP4).

    Parameters:
      fieldName (str): Name of field to render. Special 'test' generates synthetic data.
      timeFrames (list[int]): Ordered list of time frame indices to include (each becomes a frame).
      filePrefix (str, optional): Filename prefix for outputs. Default ''.
      colorMap (str, optional): Colormap name. If None/empty, pulled from sim.fields_info. Ignored if fluctuation set (forced to 'bwr'). Default 'inferno'.
      fluctuation (str | bool, optional):
      '' (default) for raw field.
      'tavg', 'tavgrelative' -> subtract (and optionally normalize to %) time average over provided window.
      'yavg', 'yavgrelative' -> subtract (and optionally normalize to %) y-average.
      True is interpreted as 'yavg'.
      clim (list | tuple | None, optional): [vmin, vmax] color limits. [] or None -> automatic per PyVista.
      logScale (bool, optional): Use logarithmic coloring. Default False.
      fps (int, optional): Frames per second in output movie. Default 14.
      cameraPath (list[dict], optional): Keyframe camera settings. Each dict keys:
      'position': (x,y,z) in units of major-radius (scaled internally),
      'looking_at': (x,y,z),
      'zoom': float.
      If >=2 provided, camera interpolates linearly between them across frames.
      movie_type (str, optional): 'gif' or 'mp4' (accepts 'gif' or '.gif'; everything else -> mp4). Default 'gif'.

    Behavior:
      - Builds first frame as in plot(); subsequent frames update scalar arrays in existing meshes for speed.
      - If fluctuation specified, output filename is prefixed with 'd'.
      - Time label auto-updated each frame (ms units scaled by tref and offset t0).
      - Camera motion handled by Camera class (static if cameraPath empty).

    Output:
      Writes animated file:
      {filePrefix}torproj_movie_[d]fieldName.(gif|mp4)

    Side Effects:
      Uses off-screen PyVista plotter.
      Prints incremental progress to stdout.

    Returns:
      None
    """
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
    plotter.enable_anti_aliasing()

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

  def write_texts(self, plotter : pv.Plotter):
    for i in range(len(self.txt_texts)):
      plotter.add_text(self.txt_texts[i], position=self.txt_positions[i], 
                       font_size=self.txt_sizes[i], name=self.txt_names[i],
                       color=self.text_color, shadow=True)
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