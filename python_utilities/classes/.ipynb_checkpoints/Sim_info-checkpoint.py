import numpy as np
import postgkyl as pg

class Sim_info:
    def __init__(self, eps0, eV, mp, me, B_axis, AMU, x_LCFS,
                fileprefix,g0simdir,simname,simdir,expdatadir='',wkdir=''):
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
        self.wkdir      = wkdir
        self.simname    = simname
        # main simulation directory (where submit.sh is)
        self.simdir     = self.g0simdir+simdir+self.simname+'/'
        # Prefix of the files, it's everything before e.g. -elc_0.gkyl
        self.fileprefix = self.wkdir+self.simdir+fileprefix
        # directory where the sim data are (usually simdir/wk)
        self.datadir    = self.simdir+self.wkdir
        # to store the grid definition
        self.grid       = None
    def initialize_grid(self):
        data = pg.data.GData(sim_params.datadir+'/xyz.gkyl')
        normgrids = data.get_grid()
        normx = normgrids[0]; normy = normgrids[1]; normz = normgrids[2];
        Nx = (normx.shape[0]-2)*2
        Ny = (normy.shape[0]-2)*2
        Nz = normz.shape[0]*4
        self.grid = Grid(Nx, Ny, Nz, Nvp=None, Nmu=None)  # Initialize the grid instance

    def info(self):
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
        if self.grid is not None:
            self.grid.info()

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