import numpy as np
import matplotlib.pyplot as plt
import os

# Configure plotting
plt.rcParams["figure.figsize"] = (6,4)

# Custom libraries and routines
import pygkyl

home_dir = os.path.expanduser("~")
repo_dir = home_dir+'/personal_gkyl_scripts/'

simdir = 'sim_data_dir_example/3x2v_example/gk_tcv_posD_iwl_3x2v_electron_heating/'
fileprefix = 'gk_tcv_posD_iwl_3x2v_D02'

simulation = pygkyl.simulation_configs.import_config('tcv_nt', simdir, fileprefix)

simulation.normalization.set('t','mus') # time in micro-seconds
simulation.normalization.set('x','minor radius') # radial coordinate normalized by the minor radius (rho=r/a)
simulation.normalization.set('y','Larmor radius') # binormal in term of reference sound Larmor radius
simulation.normalization.set('z','pi') # parallel angle devided by pi
simulation.normalization.set('fluid velocities','thermal velocity') # fluid velocity moments are normalized by the thermal velocity
simulation.normalization.set('temperatures','eV') # temperatures in electron Volt
simulation.normalization.set('pressures','Pa') # pressures in Pascal
simulation.normalization.set('energies','MJ') # energies in mega Joules

sim_frames = simulation.available_frames['ion_BiMaxwellianMoments'] # you can check the available frames for each data type like ion_M0, ion_BiMaxwellian, etc.)
print("%g time frames available (%g to %g)"%(len(sim_frames),sim_frames[0],sim_frames[-1]))


Nint_polproj = 32
Nint_fsproj = 24
torproj = pygkyl.TorusProjection()
torproj.setup(simulation, Nint_polproj=Nint_polproj, Nint_fsproj=Nint_fsproj, 
              phiLim = [0, 3*np.pi/2], rhoLim = [2,-2])


camera_settings = simulation.geom_param.camera_global
torproj.plot(fieldName='Ti', timeFrame=sim_frames[-1], fluctuation='yavg_relative',clim=[-20,20], logScale=False, 
             vessel=True, filePrefix='', imgSize=(800,600), jupyter_backend='none', colorbar=True,
             save_html=True, lighting=False, 
            #  viewVector=[1,-1,1],
             cameraSettings=camera_settings)


c0 = simulation.geom_param.camera_global
c1 = simulation.geom_param.camera_zoom_lower

camera_path = [c0, c0, c1, c1]
time_frames = sim_frames[:]
torproj.movie(fieldName='Te', timeFrames=time_frames, fluctuation='yavg_relative', filePrefix='fullsim_',
              cameraPath=camera_path, logScale=False, clim=[-20,20])