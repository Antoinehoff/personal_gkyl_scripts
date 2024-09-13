import os
import re
import numpy as np
import postgkyl as pg
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#-----------------Define functions-------------------#
# function to extract filePrefix from lua file(s)
def find_prefix(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                #result.append(os.path.join(root, name))
                prefix = re.sub('.lua','',name)
                result.append(prefix)
    return result

# Function reads gkyl data of 2D axisymmetric fields and produces 1D array
# at outer midplane (omp)
def func_data_omp(field2d, comp):
    field2dInterp = pg.data.GInterpModal(field2d, 1, 'ms')
    interpGrid, field2dValues = field2dInterp.interpolate(comp)
    # get cell center coordinates since interpolate returns edge values
    CCC = []
    for j in range(0,len(interpGrid)):
        CCC.append((interpGrid[j][1:] + interpGrid[j][:-1])/2)
    x_vals = CCC[0]
    z_vals = CCC[1]
    z_slice = len(z_vals)//2
    field1dValues = field2dValues[:,z_slice,0]
    return x_vals,field1dValues

def func_time_ave(listIn):
    arrayOut = np.array(listIn)
    arrayOut = np.mean(arrayOut,axis=0)
    return arrayOut

def func_calc_norm_fluc(data2d, dataAve, dataNorm, Nt, Ny, Nx):
    data2dTot = np.reshape(data2d, (Nt*Ny,Nx))
    dataAve2d = np.array([dataAve,]*(Nt*Ny))
    delt = data2dTot - dataAve2d

    sigma = np.sqrt(np.mean(delt**2,axis=0)) # rms of density fluctuations
    delt_norm = sigma/dataNorm
    return delt, delt_norm

def find_available_frames(folder_path,dataname):
    # Regular expression pattern to match files with the format "*dataname_X.gkyl"
    pattern = re.compile(r"%s_([0-9]+)\.gkyl$"%dataname)

    # List to store the frame numbers
    frames = []
    if len(os.listdir(folder_path)) == 0:
        print("No file found in %s"%folderpath)
    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        # Use regular expression to find matching filenames
        match = pattern.search(filename)
        if match:
            # Extract the frame number and add it to the list
            frame_number = int(match.group(1))
            frames.append(frame_number)

    # Sort the frame numbers for easier interpretation
    frames.sort()
    frames = list(set(frames))
    return frames

def get_1xt_slice(fileprefix, dataname, cutdirection, ccoords, tf, comp=0):
    """
    Extracts a slice of data from a 1D simulation at specific y and z coordinates for a given time frame.
    Parameters:
    - fileprefix: Prefix of the data file
    - dataname: Name of the data field to extract
    - cutdirection: string to stipulate the cut 'x','y','z'
    - xcoords: spatial coordinates of the two cutting dimensions [c1,c2]
    - tf: Time frame index
    Returns:
    - X: 1D slice coordinate
    - Y: Data values at the slice
    - t: Simulation time corresponding to the time frame
    - cs: effective 2x cut
    """
    # Construct the filename for the given time frame
    fname = "%s-%s_%d.gkyl" % (fileprefix, dataname, tf)
    # Load the data from the file
    data = pg.data.GData(fname)
    # Interpolate the data using modal interpolation
    dg = pg.data.GInterpModal(data,1,'ms')
    dg.interpolate(comp,overwrite=True)
    cs = [0,0]
    # Select the specific slice
    if cutdirection == 'x':
        pg.data.select(data, z1=ccoords[0], z2=ccoords[1], overwrite=True)
        cs[0] = (data.ctx['lower'][1]+data.ctx['upper'][1])/2
        cs[1] = (data.ctx['lower'][2]+data.ctx['upper'][2])/2
        comp  = 0
    elif cutdirection == 'y':
        pg.data.select(data, z0=ccoords[0], z2=ccoords[1], overwrite=True) 
        cs[0] = (data.ctx['lower'][0]+data.ctx['upper'][0])/2
        cs[1] = (data.ctx['lower'][2]+data.ctx['upper'][2])/2     
        comp  = 1
    elif cutdirection == 'z':
        pg.data.select(data, z0=ccoords[0], z1=ccoords[1], overwrite=True)
        cs[0] = (data.ctx['lower'][0]+data.ctx['upper'][0])/2
        cs[1] = (data.ctx['lower'][1]+data.ctx['upper'][1])/2
        comp  = 2
        
    # Extract the simulation time and slice location from the data context
    ts = data.ctx['time']

    # Get the spatial grid and corresponding data values
    x = data.get_grid()
    f = data.get_values()
    # Extract the first component of X (1D grid)
    x = x[comp]
    # Discard the last point in the X array (assumed redundant for the slice)
    x = x[0:-1]
    # Flatten the data values array for 1D representation
    f = f.flatten()
    return x, f, cs, ts

def get_1xt_diagram(simdir, fileprefix, dataname, cutdirection, ccoords, comp=0):
    """
    Generate an X-T diagram using data from simulation slices.
    Parameters:
    - simdir: Directory containing simulation data
    - fileprefix: Prefix of the data files
    - dataname: Name of the data field to extract
    - yf: y-coordinate value for the slice
    - zf: z-coordinate value for the slice
    Returns:
    - XX: Meshgrid for the spatial dimension
    - YY: Meshgrid for the time dimension
    - ZZ: Data values for each (x, t) point
    """
    # Get available time frames
    tfs = find_available_frames(simdir + 'wk/', "field")
    # Get dimensions: Assume first time frame is representative for X dimension
    X, _, _, _ = get_1xt_slice(fileprefix,dataname,cutdirection,ccoords,tfs[0],comp)
    nx = len(X)
    nt = len(tfs)
    # Initialize ZZ to store the data (time vs space)
    ZZ = np.zeros((nt, nx))
    # store times
    t  = np.zeros((nt, 1))
    # Fill ZZ with data for each time frame
    for it, tf in enumerate(tfs):
        _, ZZ[it, :], cs, ts = get_1xt_slice(fileprefix,dataname,cutdirection,ccoords,tf,comp)
        t[it] = ts
    # Create meshgrids for X and T (time)
    XX, YY = np.meshgrid(X, t)
    return XX, YY, ZZ, cs

def get_2D_from_3D(fileprefix, dataname, cdirection, cc, tf, comp=0):
    """
    Extracts a slice of data from a simulation at specific cut coordinates cc
    Parameters:
    - fileprefix: Prefix of the data file
    - dataname: Name of the data field to extract
    - cutposition: string to stipulate the cut position 'x','y','z'
    - cc: spatial cut coordinates
    - tf: time frame
    - comp: component of the field (optional)
    Returns:
    - XX: meshgrid of cut1 grid values
    - YY: meshgrid of cut2 grid values
    - ZZ: field values
    - cs: effective cut coordinates
    - ts: cutting time
    """
    # Construct the filename for the given time frame
    fname = "%s-%s_%d.gkyl" % (fileprefix, dataname, tf)
    # Load the data from the file
    data = pg.data.GData(fname)
    # Interpolate the data using modal interpolation
    dg = pg.data.GInterpModal(data,1,'ms')
    dg.interpolate(comp,overwrite=True)
    
    # Select the specific slice
    if cdirection == 'z':
        pg.data.select(data ,z2=cc, overwrite=True)
        values = data.get_values(); values = values[:,:,0,0]
        cs     = (data.ctx['lower'][2]+data.ctx['upper'][2])/2
        comp1  = 0
        comp2  = 1 
    elif cdirection == 'y':
        pg.data.select(data ,z1=cc, overwrite=True)
        values = data.get_values(); values = values[:,0,:,0]
        cs     = (data.ctx['lower'][1]+data.ctx['upper'][1])/2
        comp1  = 0
        comp2  = 2 
    elif cdirection == 'x':
        pg.data.select(data ,z0=cc, overwrite=True)
        values = data.get_values(); values = values[0,:,:,0]
        cs     = (data.ctx['lower'][0]+data.ctx['upper'][0])/2
        comp1  = 1
        comp2  = 2   
        
    # Extract the simulation time and slice location from the data context
    ts = data.ctx['time']
    
    # Get the spatial grid and corresponding data values
    grid   = data.get_grid()
    # Extract the grids
    x = grid[comp1]
    y = grid[comp2]
    # Discard the last point in the X array (assumed redundant for the slice)
    x = x[0:-1]
    y = y[0:-1]
    # Make it a meshgrid
    YY, XX = np.meshgrid(y, x)
    # Flatten the data values array for 1D representation
    # values = values.flatten()
    return XX, YY, values, cs, ts

def make_2D_movie(fileprefix, dataname, cdirection, ccoord, tfs,
                  xscale=1., yscale=1., cscale=1.,
                  xlim=[], ylim=[], clim=[],
                  xlabel='', ylabel='', clabel='', title='', comp=0):
    for tf in tfs:
        XX, YY, ZZ, cs, ts = get_2D_from_3D(fileprefix, dataname, cdirection, ccoord, tf, comp);
        fig,ax = plt.subplots()
        pcm = ax.pcolormesh(XX/xscale,YY/yscale,ZZ/cscale,cmap='inferno');
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel);
        ax.set_title((title+", t=%2.2e (ms)")%(cs,ts*1000));
        cbar = fig.colorbar(pcm,label=clabel);
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_xlim(ylim)
        if clim:
            pcm.set_clim(clim)    
        fig.tight_layout()
        fig.savefig(f'gif_tmp/plot_{tf}.png')
        plt.close()


    moviename = 'movie_'+dataname+'_'+cdirection+'='+('%2.2f'%cs)
    if xlim:
        moviename+='_xlim_%2.2d_%2.2d'%(xlim[0],xlim[1])
    if ylim:
        moviename+='_ylim_%2.2d_%2.2d'%(ylim[0],ylim[1])
    if clim:
        moviename+='_clim_%2.2d_%2.2d'%(clim[0],clim[1])
    moviename += '.gif'
    # Load images
    images = [Image.open(f'gif_tmp/plot_{tf}.png') for tf in tfs]
    # Save as gif
    images[0].save(moviename, save_all=True, append_images=images[1:], duration=200, loop=1)
    print("movie "+moviename+" created.");

def plot_radial_profile_time_evolution(fileprefix,dataname,comp,spec,cdirection,ccoords,twindow,xlabel,ylabel,title):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)

    if dataname == 'field':
        fulldataname = dataname
    else:
        fulldataname = spec.name+'_'+dataname
        ylabel = ylabel%(spec.name[0])

    if dataname[1] == '2':
        yscale = 2./spec.m
    if dataname[1] == '1':
        yscale = spec.vth
    else:
        yscale = 1.

    # Collect all time values for normalization
    times = []
    for tf in twindow:
        _, _, _, t = get_1xt_slice(fileprefix,fulldataname,cdirection,ccoords,tf,comp)
        times.append(t*1000)
    norm = plt.Normalize(min(times), max(times))
    colormap = cm.viridis  # You can choose any colormap, e.g., 'plasma', 'inferno', etc.

    for tf in twindow:
        x,f,cs,ts = get_1xt_slice(fileprefix,fulldataname,cdirection,ccoords,tf,comp)
        if dataname[1] == '1':
            _,m0,_,_ = utils.get_1xt_slice(fileprefix,spec.name+'_'+'M0',cdirection,ccoords,tf,comp)
            f /= m0
        ax1.plot(x-x_LCFS,f/yscale,label=r'$t=%2.2e$'%ts,color=colormap(norm(ts*1000)))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title%(cs[0],cs[1]))
    # Add a colorbar to the figure
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm);sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1);cbar.set_label('Time (ms)')  # Label for the colorbar

class Species:
    def __init__(self, name, m, q, T0, n0):
        self.name = name # name
        self.m = m       # mass in kg
        self.q = q       # charge in C
        self.T0 = T0     # initial temperature in K
        self.n0 = n0     # initial density in m^-3
        self.vth = np.sqrt(T0 / m)  # thermal velocity vth = sqrt(T0/m)

    def get_omega_c(self, B):
        """Compute the cyclotron frequency: omega_c = qB/m"""
        omega_c = (self.q * B) / self.m
        return omega_c

    def get_larmor_radius(self, B):
        """Compute the Larmor radius: rho = vth / omega_c"""
        omega_c = self.get_omega_c(B)
        rho = self.vth / omega_c
        return rho
    
    def info(self):
        """Display species information and related parameters"""
        print(f"Species: {self.name}")
        print(f"Mass (m): {self.m:.3e} kg")
        print(f"Charge (q): {self.q:.3e} C")
        print(f"Initial Temperature (T0): {self.T0:.3e} K")
        print(f"Initial Density (n0): {self.n0:.3e} m^-3")
        print(f"Thermal Velocity (vth): {self.vth:.3e} m/s")
        
class Sim_data:
    def __init__(self, eps0, eV, mp, me, B_axis, AMU, x_LCFS,
                fileprefix,g0simdir,simname,simdir,expdatadir=''):
        """
        Initialize the simulation parameters.

        :param eps0: Vacuum permittivity (F/m)
        :param eV: Elementary charge (C)
        :param mp: Proton mass (kg)
        :param me: Electron mass (kg)
        :param B_axis: Magnetic field strength at the axis (T)
        :param AMU: Atomic mass unit (kg)
        :param x_LCFS: Last closed flux surface position (m)
        """
        self.eps0 = eps0  # Permittivity of free space
        self.eV = eV      # Elementary charge (eV)
        self.mp = mp      # Proton mass
        self.me = me      # Electron mass
        self.B_axis = B_axis  # Magnetic field strength
        self.AMU = AMU    # Atomic mass unit
        self.x_LCFS = x_LCFS  # Position of LCFS
        self.expdatadir = expdatadir
        self.g0simdir   = g0simdir
        self.simname    = simname
        self.simdir     = self.g0simdir+simdir+self.simname+'/'
        self.fileprefix = self.simdir+fileprefix
    def display_info(self):
        """
        Display the information of the simulation parameters.
        """
        print(f"Simulation Parameters:\n"
              f"  Vacuum Permittivity (eps0): {self.eps0} F/m\n"
              f"  Elementary Charge (eV): {self.eV} C\n"
              f"  Proton Mass (mp): {self.mp} kg\n"
              f"  Electron Mass (me): {self.me} kg\n"
              f"  Magnetic Field Strength (B_axis): {self.B_axis} T\n"
              f"  Atomic Mass Unit (AMU): {self.AMU} kg\n"
              f"  Last Closed Flux Surface (x_LCFS): {self.x_LCFS} m\n")

    def cyclotron_frequency(self, charge, mass):
        """
        Calculate the cyclotron frequency given a charge and mass.

        :param charge: Particle charge (C)
        :param mass: Particle mass (kg)
        :return: Cyclotron frequency (Hz)
        """
        omega_c = abs(charge * self.B_axis / mass)
        return omega_c

    def larmor_radius(self, thermal_velocity, charge, mass):
        """
        Calculate the Larmor radius given thermal velocity, charge, and mass.

        :param thermal_velocity: Thermal velocity (m/s)
        :param charge: Particle charge (C)
        :param mass: Particle mass (kg)
        :return: Larmor radius (m)
        """
        omega_c = self.cyclotron_frequency(charge, mass)
        rho = thermal_velocity / omega_c
        return rho