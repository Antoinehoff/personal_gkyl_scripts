import numpy as np
import matplotlib.pyplot as plt

from ..classes import Frame, TimeSerie
from ..tools import fig_tools, math_tools
from .poloidalprojection import PoloidalProjection

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
    self.y_int = None
    self.z_int = None
    
  def setup(self, simulation, timeFrame=0, Nint=64, rho=0.9, phi=[0,2*np.pi], smooth=True):
    
    self.sim = simulation
    self.smooth = smooth
    self.phiLim = phi if len(phi) == 2 else [0, 2 * np.pi]
    self.timeFrame0 = timeFrame
    
    # Load a frame to get the grid
    if len(simulation.available_frames['field']) > 0:
      fieldName = 'phi'
    else:
      fieldName = 'ni'
      
    frame = Frame(self.sim, fieldname=fieldName, tf=timeFrame, load=True)

     # Compute rcut and find the index
    rx = self.sim.geom_param.r_x(frame.cgrids[0])
    rhox = rx / self.sim.geom_param.a_mid
    if isinstance(rho, int): rho = rhox[rho]
    rcut = rho * self.sim.geom_param.a_mid
    self.ix0 = np.argmin(np.abs(rx - rcut))
    self.r0 = self.sim.geom_param.r_x(frame.cgrids[0][self.ix0])
    self.rho = rho
  
    self.Nint = Nint
    self.ntor = Nint
    self.npol = Nint
    phi_fs = np.linspace(phi[0], phi[1], self.Nint)
    theta_fs = np.linspace(-np.pi, np.pi, self.Nint)
    self.phi_fs, self.theta_fs = np.meshgrid(phi_fs, theta_fs, indexing='ij')
        
  def project_field(self, field, Nintz=24):
    
    phi = np.linspace(self.phiLim[0], self.phiLim[1], self.Nint)
    polproj = PoloidalProjection()
    polproj.setup(self.sim,nzInterp=Nintz, rholim=self.rho, timeFrame=self.timeFrame0)

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
    field_frame = Frame(self.sim, fieldname=fieldName, tf=timeFrame, load=True)
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
    
    with Frame(self.sim, fieldname=fieldName, tf=timeFrame, load=True) as field_frame:
      time = field_frame.time
      vsymbol = field_frame.vsymbol 
      vunits = field_frame.vunits
      toproject = field_frame.values
      timetitle = field_frame.timetitle
      frame_info = Frame(self.sim, fieldname=fieldName, tf=timeFrame, load=False)

    if len(fluctuation) > 0:
      serie = TimeSerie(simulation=self.sim, fieldname=fieldName, time_frames=avg_window, load=True)
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
