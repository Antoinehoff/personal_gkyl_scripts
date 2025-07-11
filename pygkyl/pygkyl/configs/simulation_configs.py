import numpy as np
from ..classes import Simulation, Species, Source
from ..projections.poloidalprojection import Inset
from ..interfaces.gyacomointerface import GyacomoInterface
from .vessel_data import tcv_vessel_data, d3d_vessel_data, sparc_vessel_data, nstxu_vessel_data

def import_config(configName, simDir, filePrefix = '', x_LCFS = None, x_out = None, 
                  load_metric=True, add_source=True, dimensionality='3x2v'):
    if configName in ['TCV_PT', 'tcv_pt']:
        sim = get_tcv_pt_sim_config(simDir, filePrefix, x_LCFS, x_out, dimensionality)
    elif configName in ['TCV_NT', 'tcv_nt']:
        sim = get_tcv_nt_sim_config(simDir, filePrefix, x_LCFS, x_out, dimensionality)
    elif configName in ['D3D_PT', 'd3d_pt']:
        sim = get_d3d_pt_sim_config(simDir, filePrefix, x_LCFS, x_out)
    elif configName in ['D3D_NT', 'd3d_nt']:
        sim = get_d3d_nt_sim_config(simDir, filePrefix, x_LCFS, x_out, dimensionality)
    elif configName in ['SPARC', 'sparc']:
        sim = get_sparc_sim_config(simDir, filePrefix, x_LCFS, x_out, dimensionality)
    elif configName in ['NSTXU', 'nstxu']:
        sim = get_nstxu_sim_config(simDir, filePrefix, x_LCFS, x_out, dimensionality)
    elif configName in ['gyacomo', 'GYACOMO', 'Gyacomo']:
        sim = get_gyacomo_sim_config(simDir)
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

def get_tcv_pt_sim_config(simdir, fileprefix, x_LCFS = None, x_out = None, dimensionality='3x2v'):
    '''
    This function returns a simulation object for a TCV PT clopen 3x2v simulation.
    '''
    R_axis = 0.8727
    if x_LCFS is None : x_LCFS = 0.04
    if x_out is None : x_out = 0.08
    simulation = Simulation(dimensionality=dimensionality)
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )

    simulation.set_geom_param(
        B_axis      = 1.4,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,         # Magnetic axis major radius
        Z_axis      = 0.1414,         # Magnetic axis height
        R_LCFSmid   = 1.0969,   # Major radius of LCFS at the midplane
        a_shift     = 0.4080,                 # Parameter in Shafranov shift
        kappa       = 1.3951,                 # Elongation factor
        delta       = 0.2826,                 # Triangularity factor
        qfit        = [497.3420166252413, -1408.736172826569, 
                       1331.4134861681464, -419.00692601227627],
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
    
    # Add discharge ID
    simulation.dischargeID = 'TCV #65125'
    
    # Add vessel data filename
    simulation.geom_param.vesselData = tcv_vessel_data

    # Add view points for the toroidal projection
    simulation.geom_param.camera_global = {
        'position':(2.3, 2.3, 0.75),
        'looking_at':(0, 0, 0),
            'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_lower = {
        'position':(0.75, 0.75, 0.1),
        'looking_at':(0., 0.8, -0.03),
            'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_obmp = {
        'position':(0.5, 1.0, 0.1),
        'looking_at':(0.0, 1.0, 0.1),
            'zoom': 1.0
    }
    # Cameras for 2:1 formats
    simulation.geom_param.camera_global_2by1 = {
        'position':(2.3, 2.3, 0.75),
        'looking_at':(0, 0.7, 0),
            'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_2by1 = {   
        'position':(2.0, 0.78, 0.1),
        'looking_at':(0., 0.795, 0.05),
        'zoom': 1.0
    }
    return simulation

def get_tcv_nt_sim_config(simdir,fileprefix, x_LCFS = None, x_out = None, dimensionality='3x2v'):
    '''
    This function returns a simulation object for a TCV NT clopen 3x2v simulation.
    '''
    R_axis = 0.8868
    if x_LCFS is None : x_LCFS = 0.04
    if x_out is None : x_out = 0.08
    
    simulation = Simulation(dimensionality=dimensionality)
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )
    
    simulation.set_geom_param(
        B_axis      = 1.4,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,         # Magnetic axis major radius
        Z_axis      = 0.1389,         # Magnetic axis height
        R_LCFSmid   = 1.0875,   # Major radius of LCFS at the midplane
        a_shift     = 1.0,                 # Parameter in Shafranov shift
        kappa       = 1.3840,                 # Elongation factor
        delta       =-0.2592,                 # Triangularity factor
        qfit        = [484.0615913225881, -1378.25993228584, 
                       1309.3099150729233, -414.13270311478726],
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
    
    # Add discharge ID
    simulation.dischargeID = 'TCV #65130'
    
    # Add vessel data filename
    simulation.geom_param.vesselData = tcv_vessel_data
    
    # Add view points for the toroidal projection
    simulation.geom_param.camera_global = {
        'position':(2.3, 2.3, 0.75),
        'looking_at':(0, 0, 0),
            'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_lower = {
        'position':(0.75, 0.75, 0.1),
        'looking_at':(0., 0.8, -0.03),
            'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_obmp = {
        'position':(0.5, 1.0, 0.1),
        'looking_at':(0.0, 1.0, 0.1),
            'zoom': 1.0
    }
    # Cameras for 2:1 formats
    simulation.geom_param.camera_global_2by1 = {
        'position':(2.3, 2.3, 0.75),
        'looking_at':(0, 0.7, 0),
            'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_2by1 = {   
        'position':(2.0, 0.78, 0.1),
        'looking_at':(0., 0.795, 0.05),
        'zoom': 1.0
    }
    return simulation

def get_d3d_pt_sim_config(simdir,fileprefix, x_LCFS = None, x_out = None):
    '''
    This function returns a simulation object for a TCV NT clopen 3x2v simulation.
    '''
    R_axis = 1.6486461
    if x_LCFS is None : x_LCFS = 0.10
    if x_out is None : x_out = 0.05
    
    simulation = Simulation(dimensionality='3x2v')
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )

    simulation.set_geom_param(
        B_axis      = 2.0,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,        # Magnetic axis major radius
        Z_axis      = 0.013055028,         # Magnetic axis height
        R_LCFSmid   = 2.17,   # Major radius of LCFS at the midplane
        a_shift     = 0.5,                 # Parameter in Shafranov shift
        kappa       = 1.35,                 # Elongation factor
        delta       = 0.4,                 # Triangularity factor
        qfit        = [407.582626469394, -2468.613680167604, 
                       4992.660489790657, -3369.710290916853],
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
    
    # Add discharge ID
    simulation.dischargeID = 'DIII-D #171650'
        
    # Add vessel data filename
    simulation.geom_param.vesselData = d3d_vessel_data
    
    # Add view points for the toroidal projection
    simulation.geom_param.camera_global = {
    'position':(2.3, 2.3, 0.75),
    'looking_at':(0, 0, 0),
        'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_lower = {   
        'position':(0.83, 0.78, -0.1),
        'looking_at':(0., 0.74, -0.17),
        'zoom': 1.0
    }

    return simulation

def get_d3d_nt_sim_config(simdir,fileprefix, x_LCFS = None, x_out = None, dimensionality='3x2v'):
    '''
    This function returns a simulation object for a TCV NT clopen 3x2v simulation.
    '''
    R_axis = 1.7074685
    if x_LCFS is None : x_LCFS = 0.10
    if x_out is None : x_out = 0.05
    
    simulation = Simulation(dimensionality=dimensionality)
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )

    simulation.set_geom_param(
        B_axis      = 2.0,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,        # Magnetic axis major radius
        Z_axis      = -0.0014645315,         # Magnetic axis height
        R_LCFSmid   = 2.17,   # Major radius of LCFS at the midplane
        a_shift     = 1.0,                 # Parameter in Shafranov shift
        kappa       = 1.35,                 # Elongation factor
        delta       = -0.4,                 # Triangularity factor
        qfit        = [154.51071835546747, -921.8584472748003, 
                       1842.1077075366113, -1231.619813170522],
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
    
    # Add discharge ID
    simulation.dischargeID = 'DIII-D #171646'
        
    # Add vessel data filename
    simulation.geom_param.vesselData = d3d_vessel_data
    
    # Add view points for the toroidal projection
    simulation.geom_param.camera_global = {
    'position':(2.3, 2.3, 0.75),
    'looking_at':(0, 0, 0),
        'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_lower = {   
        'position':(0.83, 0.78, -0.1),
        'looking_at':(0., 0.74, -0.17),
        'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_obmp = {
        'position':(0.4, 0.9, 0.0),
        'looking_at':(0.0, 0.98, 0.0),
            'zoom': 1.0
    }
    # Cameras for 1:2 formats
    simulation.geom_param.camera_global_1by2 = {
    'position':(2.3, 2.3, 0.75),
    'looking_at':(0.0, 0.8, 0),
        'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_1by2 = {   
        'position':(2.0, 0.78, 0.1),
        'looking_at':(0., 0.74, 0.05),
        'zoom': 1.0
    }
    return simulation


def get_nstxu_sim_config(simdir, fileprefix, x_LCFS = None, x_out = None, dimensionality='3x2v'):
    '''
    This function returns a simulation object for a TCV PT clopen 3x2v simulation.
    '''
    R_axis = 1.0
    if x_LCFS is None : x_LCFS = 0.04
    if x_out is None : x_out = 0.08
    simulation = Simulation(dimensionality=dimensionality)
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )
    
    simulation.set_geom_param(
        B_axis      = 1.0,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,         # Magnetic axis major radius
        Z_axis      = 0.0,         # Magnetic axis height
        R_LCFSmid   = 1.4903225806451617,   # Major radius of LCFS at the midplane
        a_shift     = 0.1,                 # Parameter in Shafranov shift
        kappa       = 2.5,                 # Elongation factor
        delta       = 0.4,                 # Triangularity factor
        qfit        = [154.51071835546747, -921.8584472748003, 
                       1842.1077075366113, -1231.619813170522],
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
    
    # Add discharge ID
    simulation.dischargeID = 'NSTX-U'
    
    # Add vessel data filename
    simulation.geom_param.vesselData = nstxu_vessel_data

    # Add view points for the toroidal projection
    simulation.geom_param.camera_global = {
        'position':(2.3, 2.3, 0.75),
        'looking_at':(0, 0, 0),
            'zoom': 0.75
    }
    simulation.geom_param.camera_zoom_lower = {
        'position':(0.75, 0.75, 0.1),
        'looking_at':(0., 0.8, -0.03),
            'zoom': 1.0
    }
    
    simulation.geom_param.camera_zoom_obmp = {
        'position':(0.75, 0.75, 0.1),
        'looking_at':(0.0, 1.0, -0.03),
            'zoom': 2.0
    }
    
    return simulation


def get_sparc_sim_config(simdir, fileprefix, x_LCFS = None, x_out = None, dimensionality='3x2v'):
    '''
    This function returns a simulation object for a TCV PT clopen 3x2v simulation.
    '''
    R_axis = 1.8885793871866297
    if x_LCFS is None : x_LCFS = 0.04
    if x_out is None : x_out = 0.08
    simulation = Simulation(dimensionality=dimensionality)
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )

    simulation.set_geom_param(
        B_axis      = 8.0,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,         # Magnetic axis major radius
        Z_axis      = -0.004184100418409997,         # Magnetic axis height
        R_LCFSmid   = 2.4066852367688023*0.99,   # Major radius of LCFS at the midplane
        a_shift     = 0.1,                 # Parameter in Shafranov shift
        kappa       = 1.65,                 # Elongation factor
        delta       = 0.4,                 # Triangularity factor
        qfit        = [3.5],
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
    
    # Add discharge ID
    simulation.dischargeID = 'SPARC'
    
    # Add vessel data filename
    simulation.geom_param.vesselData = sparc_vessel_data

    # Add view points for the toroidal projection
    simulation.geom_param.camera_global = {
        'position':(2.3, 2.3, 0.75),
        'looking_at':(0, 0, 0),
            'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_lower = {
        'position':(0.75, 0.75, 0.1),
        'looking_at':(0., 0.8, -0.03),
            'zoom': 1.0
    }
    
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

def get_gyacomo_sim_config(path):
    '''
    This function returns a simulation object for analyzing a Gyacomo simulation.
    '''
    R_axis = 1.7074685
    amid = 0.64
    R_LCFSmid = R_axis + amid
    r0 = 0.5*amid
    Lx = 0.05
    simulation = Simulation(dimensionality='3x2v', code='gyacomo')

    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )
    def qprofile(R):
        r = R - R_axis
        q0 = 1.4
        s0 = 0.8
        return q0 * (1 + s0 * (r - r0) / r0)

    simulation.set_geom_param(
        B_axis      = 2.5,           # Magnetic field at magnetic axis [T]
        R_axis      = R_axis,        # Magnetic axis major radius
        Z_axis      = 0.0,         # Magnetic axis height
        R_LCFSmid   = R_LCFSmid,   # Major radius of LCFS at the midplane
        a_shift     = 0.0,                 # Parameter in Shafranov shift
        kappa       = 1.0,                 # Elongation factor
        delta       = 0.0,                 # Triangularity factor
        qprofile_R  = qprofile,                 # Safety factor
        x_LCFS      = Lx,                 # position of the LCFS (= core domain width)
        x_out       = 0.0                  # SOL domain width
    )
    
    # Define the species
    # Temperature and density are taken from Greenfield et al. 1997, Nucl. Fusion 37 1215
    simulation.add_species(Species(name='ion',
                m=simulation.phys_param.mp, # Ion mass (proton), Deutrerium is 2.01410177811
                q=simulation.phys_param.eV,
                T0=1500*simulation.phys_param.eV, 
                n0=4e19))
    simulation.add_species(Species(name='elc',
                m=simulation.phys_param.me, 
                q=-simulation.phys_param.eV,
                T0=1500*simulation.phys_param.eV, 
                n0=4e19))

    simulation.gyac = GyacomoInterface(path,simulation)
    
    # Set up the flux tube size within the cartesian domain.
    Lx = simulation.gyac.params['GRID']['Lx'] * simulation.gyac.l0
    simulation.geom_param.x_in = (amid - r0 + Lx/2.0)
    simulation.geom_param.x_LCFS = R_LCFSmid - (R_axis + r0)
    simulation.geom_param.x_out = -(amid - r0 - Lx/2.0)
    simulation.geom_param.update_geom_params()
    
    
    simulation.available_frames = simulation.gyac.available_frames
    simulation.data_param = simulation.gyac.adapt_data_param(simulation=simulation)
    simulation.normalization = simulation.gyac.adapt_normalization(simulation=simulation)
    
    # Add a custom poloidal projection inset to position the inset according to geometry.
    inset = Inset() # all default but the lower corner position
    inset.lowerCornerRelPos = [0.3,0.32]
    simulation.polprojInset = inset
    
    # Add discharge ID
    simulation.dischargeID = 'GYACOMO, Cyclone Base Case'
    
    # Add vessel data filename
    simulation.geom_param.vesselData = d3d_vessel_data

    # Add view points for the toroidal projection
    simulation.geom_param.camera_global = {
        'position':(2.3, 2.3, 0.75),
        'looking_at':(0, 0, 0),
            'zoom': 1.0
    }
    # Cameras for 1:2 formats
    simulation.geom_param.camera_zoom_1by2 = {   
        'position':(1.2, 1.2, 0.6),
        'looking_at':(0., 0.75, 0.1),
        'zoom': 1.0
    }

    return simulation
