import numpy as np

class Species:
    """
    A class to represent a plasma species.

    Attributes:
    -----------
    name : str
        Name of the species.
    m : float
        Mass of the species in kilograms.
    q : float
        Charge of the species in coulombs.
    T0 : float
        Initial temperature of the species in kelvin.
    n0 : float
        Initial density of the species in per cubic meter.
    vt : float
        Thermal velocity of the species.
    omega_c : float or None
        Cyclotron frequency of the species.
    rho : float or None
        Larmor radius of the species.
    mu0 : float or None
        Magnetic moment of the species.
    """
    def __init__(self, name, m, q, T0, n0):
        self.name    = name   # Name of the species
        self.nshort  = name[0] # Short name (first letter of the name)
        self.m       = m      # Mass in kg
        self.q       = q      # Charge in C
        self.T0      = T0     # Initial temperature in K
        self.n0      = n0     # Initial density in m^-3
        self.vt      = np.sqrt(T0 / m)  # Thermal velocity vth = sqrt(T0/m)
        self.omega_c = None # Cyclotron frequency
        self.rho     = None # Larmor radius
        self.mu0     = None # Magnetic moment

    def set_gyromotion(self, B):
        """
        Calculate and set the gyromotion parameters based on the magnetic field B.

        Parameters:
        -----------
        B : float
            Magnetic field strength in tesla.
        """
        self.omega_c = (self.q * B) / self.m
        self.rho = self.vt / self.omega_c
        self.mu0 = self.T0 / B

    def info(self):
        """Display species information and related parameters"""
        print(f"Species: {self.name}")
        print(f"Mass (m): {self.m:.3e} kg")
        print(f"Charge (q): {self.q:.3e} C")
        print(f"Initial Temperature (T0): {self.T0:.3e} K")
        print(f"Initial Density (n0): {self.n0:.3e} m^-3")
        print(f"Thermal Velocity (vth): {self.vt:.3e} m/s")
