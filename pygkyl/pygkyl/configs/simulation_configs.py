import numpy as np
from ..classes import Simulation, Species, Source
from ..projections.poloidalprojection import Inset
from ..tools.gyacomo_interface import GyacomoInterface
from .vessel_data import tcv_vessel_data, d3d_vessel_data

def import_config(configName, simDir, filePrefix, x_LCFS = None, x_out = None, load_metric=True, add_source=True):
    if configName in ['TCV_PT', 'tcv_pt']:
        sim = get_tcv_pt_sim_config(simDir, filePrefix, x_LCFS, x_out)
    elif configName in ['TCV_NT', 'tcv_nt']:
        sim = get_tcv_nt_sim_config(simDir, filePrefix, x_LCFS, x_out)
    elif configName in ['D3D_NT', 'd3d_nt']:
        sim = get_d3d_nt_sim_config(simDir, filePrefix, x_LCFS, x_out)
    elif configName in ['gyacomo', 'GYACOMO', 'Gyacomo']:
        sim = get_gyacomo_sim_config(simDir, filePrefix)
        load_metric = False
        add_source = False
    else:
        display_available_configs()
        raise ValueError(f"Configuration {configName} is not supported.")
    
    if load_metric:
        sim.geom_param.load_metric(sim.data_param.fileprefix)

    if add_source:
        sim = add_source_baseline(sim)

    return sim

def display_available_configs():
    print("Available configurations: TCV_PT, TCV_NT")

def get_tcv_pt_sim_config(simdir, fileprefix, x_LCFS = None, x_out = None):
    '''
    This function returns a simulation object for a TCV PT clopen 3x2v simulation.
    '''
    R_axis = 0.8727315068
    if x_LCFS is None : x_LCFS = 0.04
    if x_out is None : x_out = 0.08
    simulation = Simulation(dimensionality='3x2v')
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )
    def qprofile_PT(r):
        R = r + R_axis
        a = [497.3420166252413, -1408.736172826569, 1331.4134861681464, -419.00692601227627]
        return a[0]*R**3 + a[1]*R**2 + a[2]*R + a[3]

    simulation.set_geom_param(
        B_axis      = 1.4,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,         # Magnetic axis major radius
        Z_axis      = 0.1414361745,         # Magnetic axis height
        R_LCFSmid   = 1.0968432365089495,   # Major radius of LCFS at the midplane
        a_shift     = 0.25,                 # Parameter in Shafranov shift
        kappa       = 1.45,                 # Elongation factor
        delta       = 0.35,                 # Triangularity factor
        qprofile    = qprofile_PT,                 # Safety factor
        x_LCFS      = x_LCFS,                 # position of the LCFS (= core domain width)
        x_out       = x_out                 # SOL domain width
    )
    # Define the species
    simulation.add_species(Species(name='ion',
                m=2.01410177811*simulation.phys_param.mp, # Ion mass
                q=simulation.phys_param.eV,               # Ion charge [C]
                T0=100*simulation.phys_param.eV, 
                n0=2.0e19))
    simulation.add_species(Species(name='elc',
                m=simulation.phys_param.me, 
                q=-simulation.phys_param.eV, # Electron charge [C]
                T0=100*simulation.phys_param.eV, 
                n0=2.0e19))

    simulation.set_data_param( simdir = simdir, fileprefix = fileprefix, species = simulation.species)
    
    # Add a custom poloidal projection inset to position the inset according to geometry.
    inset = Inset() # all default but the lower corner position
    inset.lowerCornerRelPos = [0.3,0.32]
    simulation.polprojInset = inset
    
    # Add vessel data filename
    simulation.vesselData = tcv_vessel_data

    return simulation

def get_tcv_nt_sim_config(simdir,fileprefix, x_LCFS = None, x_out = None):
    '''
    This function returns a simulation object for a TCV NT clopen 3x2v simulation.
    '''
    R_axis = 0.8867856264
    if x_LCFS is None : x_LCFS = 0.04
    if x_out is None : x_out = 0.08
    
    simulation = Simulation(dimensionality='3x2v')
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )
    def qprofile_NT(r):
        R = r + R_axis
        a = [484.0615913225881, -1378.25993228584, 1309.3099150729233, -414.13270311478726]
        return a[0]*R**3 + a[1]*R**2 + a[2]*R + a[3]

    simulation.set_geom_param(
        B_axis      = 1.4,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,         # Magnetic axis major radius
        Z_axis      = 0.1414361745,         # Magnetic axis height
        R_LCFSmid   = 1.0870056099999,   # Major radius of LCFS at the midplane
        a_shift     = 0.5,                 # Parameter in Shafranov shift
        kappa       = 1.4,                 # Elongation factor
        delta       =-0.38,                 # Triangularity factor
        qprofile    = qprofile_NT,                 # Safety factor
        x_LCFS      = x_LCFS,                 # position of the LCFS (= core domain width)
        x_out       = x_out                  # SOL domain width
    )
    # Define the species
    simulation.add_species(Species(name='ion',
                m=2.01410177811*simulation.phys_param.mp, # Ion mass
                q=simulation.phys_param.eV,               # Ion charge [C]
                T0=100*simulation.phys_param.eV, 
                n0=2.0e19))
    simulation.add_species(Species(name='elc',
                m=simulation.phys_param.me, 
                q=-simulation.phys_param.eV, # Electron charge [C]
                T0=100*simulation.phys_param.eV, 
                n0=2.0e19))

    simulation.set_data_param( simdir = simdir, fileprefix = fileprefix, species = simulation.species)

    # Add a custom poloidal projection inset to position the inset according to geometry.
    inset = Inset() # all default but the lower corner position
    inset.lowerCornerRelPos = [0.35,0.3]
    simulation.inset = inset
    
    # Add vessel data filename
    simulation.geom_param.vesselData = tcv_vessel_data
    
    return simulation

def get_d3d_nt_sim_config(simdir,fileprefix, x_LCFS = None, x_out = None):
    '''
    This function returns a simulation object for a TCV NT clopen 3x2v simulation.
    '''
    R_axis = 1.7074685
    if x_LCFS is None : x_LCFS = 0.10
    if x_out is None : x_out = 0.05
    
    simulation = Simulation(dimensionality='3x2v')
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )

    def qprofile(r):
        R = r + R_axis
        a = [154.51071835546747,  -921.8584472748003, 1842.1077075366113, -1231.619813170522]
        return a[0]*R**3 + a[1]*R**2 + a[2]*R + a[3]

    simulation.set_geom_param(
        B_axis      = 2.0,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,        # Magnetic axis major radius
        Z_axis      = -0.0014645315,         # Magnetic axis height
        R_LCFSmid   = 2.17,   # Major radius of LCFS at the midplane
        a_shift     = 1.0,                 # Parameter in Shafranov shift
        kappa       = 1.35,                 # Elongation factor
        delta       = -0.4,                 # Triangularity factor
        qprofile    = qprofile,                 # Safety factor
        x_LCFS      = x_LCFS,                 # position of the LCFS (= core domain width)
        x_out       = x_out                  # SOL domain width
    )
    # Define the species
    simulation.add_species(Species(name='ion',
                m=2.01410177811*simulation.phys_param.mp, # Ion mass
                q=simulation.phys_param.eV,               # Ion charge [C]
                T0=300*simulation.phys_param.eV, 
                n0=2.0e19))
    simulation.add_species(Species(name='elc',
                m=simulation.phys_param.me, 
                q=-simulation.phys_param.eV, # Electron charge [C]
                T0=300*simulation.phys_param.eV, 
                n0=2.0e19))

    simulation.set_data_param( simdir = simdir, fileprefix = fileprefix, species = simulation.species)

    # Add a custom poloidal projection inset to position the inset according to geometry.
    inset = Inset() # all default but the lower corner position
    inset.lowerCornerRelPos = [0.4,0.3]
    inset.xlim = [2.12,2.25]
    inset.ylim = [-0.15,0.15]
    simulation.inset = inset
    
    # Add vessel data filename
    simulation.geom_param.vesselData = d3d_vessel_data

    return simulation

def add_source_baseline(simulation):
    n_srcOMP=2.4e23
    x_srcOMP=0.0
    Te_srcOMP=2 * simulation.species['elc'].T0
    Ti_srcOMP=2 * simulation.species['ion'].T0
    sigmax_srcOMP=0.03 * simulation.geom_param.Lx
    floor_src=1e-2
    def custom_density_src_profile(x,y,z):
        return n_srcOMP * (np.exp(-((x - x_srcOMP) ** 2) / (2.0 * sigmax_srcOMP ** 2)) + floor_src)
    def custom_temp_src_profile_elc(x, y = None, z = None):
        mask = x < (x_srcOMP + 3 * sigmax_srcOMP)
        fout = np.empty_like(x)
        fout[mask] = Te_srcOMP; fout[~mask] = Te_srcOMP * 3.0 / 8.0
        return fout  
    def custom_temp_src_profile_ion( x, y = None, z = None):
        mask = x < (x_srcOMP + 3 * sigmax_srcOMP)
        fout = np.empty_like(x)
        fout[mask] = Ti_srcOMP; fout[~mask] = Ti_srcOMP * 3.0 / 8.0
        return fout   
    OMPsource = Source(n_src=n_srcOMP,x_src=x_srcOMP,Te_src=Te_srcOMP,Ti_src=Ti_srcOMP,
                    sigma_src=sigmax_srcOMP,floor_src=floor_src,
                    density_src_profile=custom_density_src_profile,
                    temp_src_profile_elc=custom_temp_src_profile_elc,
                    temp_src_profile_ion=custom_temp_src_profile_ion)
    simulation.add_source('Core src',OMPsource)
    return simulation

def get_gyacomo_sim_config(simdir,fileprefix):
    '''
    This function returns a simulation object for analyzing a Gyacomo simulation.
    '''
    R_axis = 1.0
    simulation = Simulation(dimensionality='3x2v')
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )
    def qprofile_NT(r):
        return 1.0

    simulation.set_geom_param(
        B_axis      = 1.0,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,         # Magnetic axis major radius
        Z_axis      = 1.0,         # Magnetic axis height
        R_LCFSmid   = 1.0,   # Major radius of LCFS at the midplane
        a_shift     = 1.0,                 # Parameter in Shafranov shift
        kappa       = 1.0,                 # Elongation factor
        delta       = 0.0,                 # Triangularity factor
        qprofile    = qprofile_NT,                 # Safety factor
        x_LCFS      = 0.0,                 # position of the LCFS (= core domain width)
        x_out       = 0.0                  # SOL domain width
    )
    # Define the species
    simulation.add_species(Species(name='ion',
                m=2.01410177811*simulation.phys_param.mp, # Ion mass
                q=simulation.phys_param.eV,               # Ion charge [C]
                T0=100*simulation.phys_param.eV, 
                n0=2.0e19))
    simulation.add_species(Species(name='elc',
                m=simulation.phys_param.me, 
                q=-simulation.phys_param.eV, # Electron charge [C]
                T0=100*simulation.phys_param.eV, 
                n0=2.0e19))

    simulation.gyac = GyacomoInterface(simdir+fileprefix)
    simulation.data_param = simulation.gyac.adapt_data_param()
    
    # Add a custom poloidal projection inset to position the inset according to geometry.
    inset = Inset() # all default but the lower corner position
    inset.lowerCornerRelPos = [0.35,0.3]
    simulation.inset = inset
    simulation.code = 'gyacomo'
    
    # Add vessel data filename
    simulation.geom_param.vesselData =None

    return simulation
