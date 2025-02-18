from ..classes import Simulation, Species

def import_config(configName, simDir, filePrefix, x_LCFS = 0.04, x_out = 0.08, load_metric=True):
    if configName == 'TCV_PT':
        sim = get_TCV_PT_sim_config(simDir, filePrefix, x_LCFS, x_out)
    elif configName == 'TCV_NT':
        sim = get_TCV_NT_sim_config(simDir, filePrefix, x_LCFS, x_out)
    else:
        display_available_configs()
        raise ValueError(f"Configuration {configName} is not supported.")
    
    if load_metric:
        sim.geom_param.load_metric(sim.data_param.fileprefix)

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

    return simulation
