import numpy as np
from ..classes import Simulation, Species, Source
from ..classes.poloidalprojection import Inset

def import_config(configName, simDir, filePrefix, x_LCFS = 0.04, x_out = 0.08, load_metric=True, add_source=True):
    if configName == 'TCV_PT':
        sim = get_TCV_PT_sim_config(simDir, filePrefix, x_LCFS, x_out)
    elif configName == 'TCV_NT':
        sim = get_TCV_NT_sim_config(simDir, filePrefix, x_LCFS, x_out)
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

def get_TCV_PT_sim_config(simdir, fileprefix, x_LCFS, x_out):
    '''
    This function returns a simulation object for a TCV PT clopen 3x2v simulation.
    '''
    simulation = Simulation(dimensionality='3x2v')
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )
    def qprofile_PT(R):
        a = [497.3420166252413, -1408.736172826569, 1331.4134861681464, -419.00692601227627]
        return a[0]*R**3 + a[1]*R**2 + a[2]*R + a[3]

    simulation.set_geom_param(
        B_axis      = 1.4,           # Magnetic field at magnetic axis [T]
        R_axis      = 0.8727315068,         # Magnetic axis major radius
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
    inset.lower_corner_rel_pos = [0.3,0.32]
    simulation.polprojInset = inset

    return simulation

def get_TCV_NT_sim_config(simdir,fileprefix, x_LCFS, x_out):
    '''
    This function returns a simulation object for a TCV NT clopen 3x2v simulation.
    '''
    simulation = Simulation(dimensionality='3x2v')
    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )
    def qprofile_NT(R):
        a = [484.0615913225881, -1378.25993228584, 1309.3099150729233, -414.13270311478726]
        return a[0]*R**3 + a[1]*R**2 + a[2]*R + a[3]

    simulation.set_geom_param(
        B_axis      = 1.4,           # Magnetic field at magnetic axis [T]
        R_axis      = 0.8867856264,         # Magnetic axis major radius
        Z_axis      = 0.1414361745,         # Magnetic axis height
        R_LCFSmid   = 1.0870056099999,   # Major radius of LCFS at the midplane
        a_shift     = 1.0,                 # Parameter in Shafranov shift
        kappa       = 1.4,                 # Elongation factor
        delta       =-0.39,                 # Triangularity factor
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
    inset.lower_corner_rel_pos = [0.35,0.3]
    simulation.inset = inset

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