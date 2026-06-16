import sys
import os
import glob
import tempfile
import multiprocessing as mp

import numpy as np
import pyvista as pv
pv.global_theme.colorbar_orientation = 'vertical'

from ..classes import Frame, TimeSerie, Simulation
from ..tools import fig_tools
from .fluxsurfprojection import FluxSurfProjection
from .poloidalprojection import PoloidalProjection

# ---------------------------------------------------------------------------
# Module-level field label tables
# ---------------------------------------------------------------------------
_FIELD_LABELS_RAW = {
  'ni':   'ni [m-3]',
  'ne':   'ne [m-3]',
  'Te':   'Te [eV]',
  'Ti':   'Ti [eV]',
  'phi':  'phi [V]',
  'pi':   'pi [Pa]',
  'pe':   'pe [Pa]',
  'test': 'test [a.u.]',
}
_FIELD_LABELS_RELATIVE = {
  'ni':   'ni - <ni> [%]',
  'ne':   'ne - <ne> [%]',
  'Te':   'Te - <Te> [%]',
  'Ti':   'Ti - <Ti> [%]',
  'phi':  'phi - <phi> [%]',
  'pi':   'pi - <pi> [%]',
  'pe':   'pe - <pe> [%]',
  'test': 'test - <test> [%]',
}

def get_label(fieldName, fluctuation):
  """Return a human-readable colorbar label for *fieldName* / *fluctuation*."""
  is_relative = 'relative' in fluctuation
  table = _FIELD_LABELS_RELATIVE if is_relative else _FIELD_LABELS_RAW
  if fieldName in table:
    return table[fieldName]
  print(f"No label implemented for {'fluct. ' if is_relative else ''}{fieldName} yet.")
  return fieldName


def _print_progress(n, total):
  sys.stdout.write(f"\rProcessed frames: {n}/{total}... ")
  sys.stdout.flush()


# ---------------------------------------------------------------------------
# Parallel movie generation helpers
# ---------------------------------------------------------------------------
_TORUS_FORK_STATE = {}


def _torus_movie_worker(args):
  """Render a single torus movie frame to a PNG file (runs in a forked worker)."""
  frame_idx, timeFrame, out_path = args
  torproj     = _TORUS_FORK_STATE['torproj']
  fieldName   = _TORUS_FORK_STATE['fieldName']
  fluctuation = _TORUS_FORK_STATE['fluctuation']
  logScale    = _TORUS_FORK_STATE['logScale']
  colorMap    = _TORUS_FORK_STATE['colorMap']
  clim        = _TORUS_FORK_STATE['clim']
  cam_stops   = _TORUS_FORK_STATE['cam_stops']
  n_frames    = _TORUS_FORK_STATE['n_frames']

  plotter = pv.Plotter(window_size=torproj.imgSize, off_screen=True)
  plotter.enable_anti_aliasing()
  plotter = torproj.setup_frame(plotter, fieldName, timeFrame, fluctuation, logScale, colorMap, clim)
  cam = Camera(torproj.sim.geom_param, stops=cam_stops, nframes=n_frames)
  cam.update_camera(frame_idx)
  plotter = cam.update_plotter(plotter)
  plotter.show(screenshot=out_path)
  torproj._texts.clear()
  return frame_idx


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

  Key Attributes:
    polprojs (list[PoloidalProjection]): Poloidal projection helpers (one per phi in phiLim).
    fsprojs (list[FluxSurfProjection]): Flux-surface projection helpers (one per rho in rhoLim).
    pvphishift (float): Phase shift for correct orientation in PyVista.
    text_color (str): Auto-selected contrasting color for text (based on background).
    imgSize (tuple[int,int]): Render window size in pixels.
    off_screen (bool): Enable off-screen rendering (for batch/movie generation).
    show_colorbar (bool): Toggle scalar bar display.
    colorbar_args (dict): PyVista scalar bar customization arguments.
    show_vessel (bool): Toggle vessel geometry rendering.
    show_limiter (bool): Toggle limiter geometry rendering.
    adjust_vessel (bool): Scale vessel R to match plasma extent.
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
    time_label (bool): Show time label overlay.
    time_pos (str): Position keyword for the time label.

  Notes:
    - For fluctuation strings, accepted substrings: 'tavg', 'yavg' + optional 'relative'.
    - For test rendering, fieldName='test' builds synthetic scalar fields.
    - Camera motion is handled by Camera class via user-defined keyframes (stops).
  """

  # ------------------------------------------------------------------ #
  #  Class-level defaults (safe immutable types; dicts live in __init__) #
  # ------------------------------------------------------------------ #

  # Rendering window
  imgSize         = (800, 600)
  off_screen      = False

  # Colorbar
  show_colorbar   = True

  # Vessel rendering
  show_vessel           = True
  adjust_vessel         = True
  show_limiter          = True
  vessel_opacity        = 1.0
  vessel_opacity_inner  = 0.3
  vessel_lighting       = True
  vessel_smooth_shading = True
  vessel_pbr            = True
  vessel_metallic       = 0.8
  vessel_roughness      = 0.3
  vessel_split_sharp_edges = True
  vessel_ntor           = 256
  vessel_rgb            = [128, 128, 128]

  # Plasma mesh rendering
  mesh_lighting       = False
  mesh_smooth_shading = False
  mesh_specular       = 1.0
  mesh_specular_power = 128

  # Scene / overlays
  background_color = 'white'
  additional_text  = None
  font_size        = 12
  logo_path        = None
  logo_position    = (0.0, 0.01)
  logo_size        = (0.2, 0.2)
  logo_opacity     = 0.6
  time_label       = True
  time_pos         = 'lower_left'

  # Simulation / geometry
  sim        = None
  phiLim     = [0, 3*np.pi/4]
  rhoLim     = [4, -2]
  timeFrame0 = 0
  tref       = 1e-3   # reference time in s (default 1 ms)
  t0         = 0.0    # time offset in units of tref

  # ------------------------------------------------------------------ #
  def __init__(self):
    self.polprojs  = []
    self.fsprojs   = []
    self.pvphishift = np.pi/2   # phase shift for correct orientation in PyVista

    # Per-instance copy so mutations in setup() don't bleed across instances
    self.colorbar_args = {
      'title':           None,
      'position_x':      0.025,
      'position_y':      0.25,
      'width':           0.05,
      'height':          0.5,
      'title_font_size': 15,
      'label_font_size': 14,
      'n_labels':        5,
      'fmt':             '%.1e',
      'shadow':          True,
      'vertical':        True,
    }

    # Ordered dict: name -> {'text': str, 'position': str|tuple, 'size': int}
    self._texts = {}


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
    phishift = self.pvphishift
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

      # Draw the vessel
      Rvess = self.sim.geom_param.vessel_data['R']
      Zvess = self.sim.geom_param.vessel_data['Z']
      
      # scale slightly the vessel to avoid intersection with the plasma
      Rplas_min = self.polprojs[0].RIntN.min() * np.cos(np.pi / self.fsprojs[0].ntor) * 0.98
      Rplas_max = self.polprojs[0].RIntN.max() * 1.02
      Rvess_min = Rvess.min()
      Rvess_max = Rvess.max()
      vgap_min = Rplas_min - Rvess_min
      vgap_max = Rvess_max - Rplas_max
      # check if there is clipping
      clip_min = vgap_min < 0
      clip_max = vgap_max < 0
      # if vgap_min > 0 or vgap_max < 0:
      #   if vgap_min > -vgap_max:
      #     Delta = vgap_min * 1.05
      #   else:
      #     Delta = -vgap_max * 1.05
      #   alpha = 1 + 2*Delta / (Rvess_max - Rvess_min)
      #   shift = (1 - alpha) * Rvess_min - Delta
      #   Rvess = alpha * Rvess + shift
      if self.adjust_vessel:
        alpha = (Rplas_max - Rplas_min) / (Rvess_max - Rvess_min)
        beta = Rplas_min - alpha * Rvess_min
        Rvess = alpha * Rvess + beta  
      
      phi = self.fsprojs[0].phi_fs + self.pvphishift
      # Interpolate the phi values on a larger grid to improve the vessel surface quality
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
      min_R_vess = np.min(R)
      opacity_values[R < 1.1*min_R_vess] = self.vessel_opacity_inner # Or whatever lower opacity you want
      gray_rgb = np.array(self.vessel_rgb)
      rgba_colors = np.tile(gray_rgb, (pvmesh.n_points, 1))
      alpha_channel = (opacity_values.ravel() * 255).astype(np.uint8)
      rgba_colors = np.c_[rgba_colors, alpha_channel]
      pvmesh['rgba_colors'] = rgba_colors

      plotter.add_mesh(pvmesh, scalars='rgba_colors', rgba=True,
                       show_scalar_bar=False, 
                       lighting=self.vessel_lighting, smooth_shading=self.vessel_smooth_shading, split_sharp_edges=self.vessel_split_sharp_edges, 
                       pbr=self.vessel_pbr, metallic=self.vessel_metallic, roughness=self.vessel_roughness)
      
      if self.show_limiter:
        # Draw the limiter
        R0 = Rvess.min()
        R1 = np.min(self.polprojs[0].Rlcfs)
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
    
      return plotter
    
  def _resolve_render_params(self, fieldName, fluctuation, colorMap, clim):
    """Normalise fluctuation/colorMap/clim args shared by plot() and movie()."""
    if isinstance(fluctuation, bool):
      fluctuation = 'yavg' if fluctuation else ''
    if clim == [] or clim is None:
      clim = None
    if fieldName == 'test':
      colorMap = colorMap or ['red', 'blue', 'green', 'yellow']
    else:
      colorMap = colorMap or self.sim.fields_info[fieldName + 'colormap']
    if fluctuation:
      colorMap = 'bwr'
    out_name = ('d' + fieldName) if fluctuation else fieldName
    return fluctuation, colorMap, clim, out_name

  def setup_frame(self, plotter: pv.Plotter, fieldName, timeFrame, fluctuation, logScale, colorMap, clim):
    fieldlabel = get_label(fieldName, fluctuation)
    pvmeshes, time = self.init_pvmeshes(fieldName, timeFrame, fluctuation=fluctuation)
    self.colorbar_args['title'] = fieldlabel
    for pvmesh in pvmeshes:
      plotter.add_mesh(pvmesh, scalars=fieldlabel, cmap=colorMap,
                       show_scalar_bar=self.show_colorbar, clim=clim, log_scale=logScale,
                       smooth_shading=self.mesh_smooth_shading, lighting=self.mesh_lighting,
                       scalar_bar_args=self.colorbar_args,
                       specular=self.mesh_specular, specular_power=self.mesh_specular_power)
    if self.show_vessel and self.sim.geom_param.vessel_data is not None:
      plotter = self.draw_vessel(plotter)
    plotter.set_background(self.background_color)

    # Queue text overlays
    time_pos = self.time_pos
    if self.additional_text:
      self._queue_text(self.additional_text['text'], self.additional_text['position'],
                       self.font_size, self.additional_text['name'])
      if self.additional_text['position'] == 'lower_left':
        time_pos = 'lower_right'
    else:
      self._queue_text(self.sim.dischargeID, 'upper_left', self.font_size, 'dischargeID')
    if self.time_label:
      self._queue_text(f"t={(time + self.t0):5.3f} ms", time_pos, self.font_size, 'time_label')
    plotter = self._flush_texts(plotter)

    if self.logo_path:
      plotter.add_logo_widget(self.logo_path, position=self.logo_position,
                              size=self.logo_size, opacity=self.logo_opacity)
    return plotter
    
    
  def plot(self, fieldName, timeFrame, filePrefix='', colorMap=None, fluctuation='',
           logScale=False, clim=None, jupyter_backend='none', save_html=False, cameraSettings=None):
    """
    Render a single torus projection frame and save a PNG (and optional HTML).

    Parameters:
      fieldName (str): Name of the field to plot. Special value 'test' builds synthetic data.
      timeFrame (int | list[int]): Time frame index or list (for averaging window when using fluctuations).
      filePrefix (str): Prefix added to output filenames. Default ''.
      colorMap (str | list | None): Colormap name. None -> pulled from sim.fields_info.
      fluctuation (str | bool): '' raw; 'tavg'/'yavg' [+'relative'] for fluctuations; True -> 'yavg'.
      logScale (bool): Apply logarithmic color scaling. Default False.
      clim (tuple | None): Color limits [vmin, vmax]. None or [] -> auto.
      jupyter_backend (str): PyVista Jupyter backend. Default 'none'.
      save_html (bool): Also export an interactive HTML file.
      cameraSettings (dict | None): Keys 'position', 'looking_at', 'zoom'. None -> simulation default.

    Returns:
      None
    """
    if isinstance(timeFrame, list):
      timeFrame = timeFrame[-1]
    fluctuation, colorMap, clim, out_name = self._resolve_render_params(fieldName, fluctuation, colorMap, clim)
    if cameraSettings is None:
      cameraSettings = self.sim.geom_param.camera_global

    plotter = pv.Plotter(window_size=self.imgSize, off_screen=self.off_screen)
    plotter.enable_anti_aliasing()
    plotter = self.setup_frame(plotter, fieldName, timeFrame, fluctuation, logScale, colorMap, clim)
    cam = Camera(self.sim.geom_param, cameraSettings)
    plotter = cam.update_plotter(plotter)

    out_path = filePrefix + 'torproj_' + out_name
    if save_html:
      plotter.export_html(out_path + '.html')
      print(f"HTML saved as {out_path}.html")
    plotter.show(screenshot=out_path + '.png', jupyter_backend=jupyter_backend)
    print(f"Image saved as {out_path}.png")
    self._texts.clear()

  def movie(self, fieldName, timeFrames, filePrefix='', colorMap='inferno', fluctuation='',
            clim=[], logScale=False, fps=14, cameraPath=[], movie_type='gif', n_workers=1):
    """
    Create an animated torus projection (GIF or MP4).

    Parameters:
      fieldName (str): Name of field to render. Special 'test' generates synthetic data.
      timeFrames (list[int]): Ordered list of time frame indices to include.
      filePrefix (str): Filename prefix for outputs. Default ''.
      colorMap (str): Colormap name. Forced to 'bwr' when fluctuation is set. Default 'inferno'.
      fluctuation (str | bool): '' raw; 'tavg'/'yavg' [+'relative'] for fluctuations; True -> 'yavg'.
      clim (list | None): [vmin, vmax] color limits. [] or None -> auto.
      logScale (bool): Use logarithmic coloring. Default False.
      fps (int): Frames per second in output movie. Default 14.
      cameraPath (list[dict]): Keyframe camera dicts ('position', 'looking_at', 'zoom').
        Two or more keyframes interpolate linearly over all frames.
      movie_type (str): 'gif' or 'mp4'. Default 'gif'.
      n_workers (int): Number of parallel worker processes for frame generation.
        n_workers=1 uses the sequential PyVista native writer (default).
        n_workers>1 forks independent processes, each rendering its frame off-screen,
        then compiles the PNGs into the output file. Requires off-screen rendering
        (EGL/OSMesa); recommended on headless systems (e.g. NERSC).

    Returns:
      None
    """
    fluctuation, colorMap, clim, out_name = self._resolve_render_params(fieldName, fluctuation, colorMap, clim)
    fieldlabel = get_label(fieldName, fluctuation)
    ext = '.gif' if movie_type in ('gif', '.gif') else '.mp4'
    out_path = filePrefix + 'torproj_movie_' + out_name + ext

    # ------------------------------------------------------------------
    # Parallel path: fork workers, each renders one frame to PNG, then
    # compile with fig_tools. Fork is safe here because no pv.Plotter
    # has been created yet in this process.
    # ------------------------------------------------------------------
    if n_workers > 1:
      _TORUS_FORK_STATE.update({
        'torproj':     self,
        'fieldName':   fieldName,
        'fluctuation': fluctuation,
        'logScale':    logScale,
        'colorMap':    colorMap,
        'clim':        clim,
        'cam_stops':   cameraPath,
        'n_frames':    len(timeFrames),
      })
      tmpdir = tempfile.mkdtemp()
      args = [
        (i, tf, os.path.join(tmpdir, f'frame_{i:06d}.png'))
        for i, tf in enumerate(timeFrames)
      ]
      completed = 0
      _print_progress(completed, len(timeFrames))
      ctx = mp.get_context('fork')
      with ctx.Pool(processes=n_workers) as pool:
        for _ in pool.imap_unordered(_torus_movie_worker, args):
          completed += 1
          _print_progress(completed, len(timeFrames))
      sys.stdout.write("\n")
      frame_files = sorted(glob.glob(os.path.join(tmpdir, 'frame_*.png')))
      fig_tools.compile_movie(frame_files, out_path, extension=ext.lstrip('.'), rmFrames=True)
      print(f"Movie saved as {out_path}")
      return

    # ------------------------------------------------------------------
    # Sequential path (n_workers=1): native PyVista writer
    # ------------------------------------------------------------------
    plotter = pv.Plotter(window_size=self.imgSize, off_screen=True)
    plotter.enable_anti_aliasing()
    if ext == '.gif':
      plotter.open_gif(out_path, fps=fps)
    else:
      plotter.open_movie(out_path, framerate=fps)

    cam = Camera(stops=cameraPath, geom=self.sim.geom_param, nframes=len(timeFrames))
    n = 0
    _print_progress(n, len(timeFrames))

    plotter = self.setup_frame(plotter, fieldName, timeFrames[0], fluctuation, logScale, colorMap, clim)
    plotter = cam.update_plotter(plotter)
    plotter.write_frame()
    self._clear_texts(plotter)
    n += 1
    _print_progress(n, len(timeFrames))

    for timeFrame in timeFrames[1:]:
      if fieldName != 'test':
        field_fs, field_RZ, time = self.get_data(fieldName, timeFrame, fluctuation)
        for i, fs in enumerate(field_fs):
          plotter.meshes[i][fieldlabel] = fs.ravel()
        for i, rz in enumerate(field_RZ):
          plotter.meshes[i + len(field_fs)][fieldlabel] = rz.ravel()
      else:
        time = 10650 + n   # dummy time for test field
      cam.update_camera(n)
      plotter = cam.update_plotter(plotter, zoom=False)
      self._texts['time_label']['text'] = f"t={time + self.t0:5.3f} ms"
      plotter = self._flush_texts(plotter)
      plotter.write_frame()
      self._clear_texts(plotter)
      n += 1
      _print_progress(n, len(timeFrames))

    sys.stdout.write("\n")
    plotter.close()
    print(f"Movie saved as {out_path}")
    self._texts.clear()

  # ------------------------------------------------------------------ #
  #  Internal text overlay helpers                                       #
  # ------------------------------------------------------------------ #

  def _queue_text(self, text, position, font_size, name):
    """Add or update a named text entry in the queue."""
    self._texts[name] = {'text': text, 'position': position, 'size': font_size}

  def _flush_texts(self, plotter: pv.Plotter):
    """Write all queued text entries to *plotter*."""
    for name, entry in self._texts.items():
      plotter.add_text(str(entry['text']), position=entry['position'],
                       font_size=entry['size'], name=name,
                       color=self.text_color, shadow=True)
    return plotter

  def _clear_texts(self, plotter: pv.Plotter):
    """Remove all text actors from *plotter* (keeps the queue for the next frame)."""
    for name in self._texts:
      plotter.remove_actor(name)

class Camera:
  """Manages PyVista camera state, supporting static and linearly-interpolated paths."""

  _DEFAULT_SETTINGS = {
    'position':   (2.3, 2.3, 0.75),
    'looking_at': (0, 0, 0),
    'zoom':       1.0,
  }

  def __init__(self, geom, settings=None, stops=None, nframes=1):
    """
    Parameters:
      geom: Geometry object with R_LCFSmid for length scaling.
      settings (dict | None): Single camera dict ('position', 'looking_at', 'zoom').
        Used only when stops is empty. Falls back to _DEFAULT_SETTINGS.
      stops (list[dict] | None): Keyframe camera dicts. Two or more keyframes
        activate linear interpolation across nframes.
      nframes (int): Total number of movie frames (used for path interpolation).
    """
    stops = (stops or []) if not isinstance(stops, list) else (stops or [])

    if stops:
      settings = stops[0]
      if len(stops) > 1:
        self._init_moving(stops, geom, nframes)
      else:
        self.update_camera = self._update_static
    else:
      settings = settings or self._DEFAULT_SETTINGS
      self.update_camera = self._update_static

    scale = geom.R_LCFSmid
    self.position   = [loc * scale for loc in settings['position']]
    self.looking_at = [loc * scale for loc in settings['looking_at']]
    self.view_up    = (0, 0, 1)
    self.zoom       = settings['zoom']

  def _init_moving(self, stops, geom, nframes):
    """Pre-compute per-frame position/look/zoom deltas for a multi-stop path."""
    self.update_camera = self._update_moving
    Ncp = len(stops)
    self.checkpoints = [i * nframes // (Ncp - 1) for i in range(Ncp)]
    self.icp  = 0
    scale     = geom.R_LCFSmid
    self.dpos  = []
    self.dlook = []
    self.dzoom = []
    for i in range(Ncp - 1):
      Nint = self.checkpoints[i + 1] - self.checkpoints[i]
      self.dpos.append([
        (stops[i+1]['position'][k]   - stops[i]['position'][k])   / Nint * scale
        for k in range(3)
      ])
      self.dlook.append([
        (stops[i+1]['looking_at'][k] - stops[i]['looking_at'][k]) / Nint * scale
        for k in range(3)
      ])
      self.dzoom.append((stops[i+1]['zoom'] - stops[i]['zoom']) / Nint)

  def _update_static(self, iframe):
    pass

  def _update_moving(self, iframe):
    if iframe > self.checkpoints[self.icp + 1]:
      self.icp += 1
    for k in range(3):
      self.position[k]   += self.dpos[self.icp][k]
      self.looking_at[k] += self.dlook[self.icp][k]
    self.zoom += self.dzoom[self.icp]

  def update_plotter(self, plotter, zoom=True):
    plotter.camera_position = [self.position, self.looking_at, self.view_up]
    if zoom:
      plotter.camera.Zoom(self.zoom)
    return plotter