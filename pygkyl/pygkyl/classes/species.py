import numpy as np
from ..tools import phys_tools

# define a dictionary of color for different species
species_colors = {
    'elc': 'blue',
    'ion': 'red',
    'e-': 'blue',
    'D+': 'green',
    'T+': 'orange',
    'W10+': 'brown',
    'W45+': 'purple',
}

class Species:
    """Represents a plasma species with its physical properties."""
    def __init__(self, name, m, q, T0, n0, Bref = 0.0):
        """
        Initializes a Species object.

        Args:
            name (str): Name of the species (e.g., 'electron').
            m (float): Mass in kg.
            q (float): Charge in C.
            T0 (float): Reference temperature in J.
            n0 (float): Reference density in m^-3.
            Bref (float, optional): Reference magnetic field in Tesla. If > 0, gyromotion parameters are calculated. Defaults to 0.0.
        """
        self.name    = name   # Name of the species
        self.nshort  = name[0] # Short name (first letter of the name)
        self.m       = m      # Mass in kg
        self.q       = q      # Charge in C
        self.T0      = T0     # reference temperature in J
        self.n0      = n0     # reference density in m^-3
        self.vt      = phys_tools.thermal_vel(T0,m)  # Thermal velocity vth = sqrt(T0/m)
        self.omega_p = phys_tools.plasma_frequency(n0, q, m)  # Plasma frequency
        self.color   = species_colors.get(self.name, 'black')
        
        
        self.omega_c = None # Cyclotron frequency
        self.rho     = None # Larmor radius
        self.mu0     = None # Magnetic moment
        self.epsilon = None # Plasma permittivity
        self.gyrate  = False # Flag indicating if gyromotion parameters are set
        
        if Bref > 0.0 and self.q != 0.0:
            self.set_gyromotion(Bref)
        
    def set_gyromotion(self, B):
        """
        Calculate and set the gyromotion parameters based on the magnetic field B.

        Parameters:
        -----------
        B : float
            Magnetic field strength in tesla.
        """
        self.omega_c = phys_tools.gyrofrequency(self.q, self.m, B)
        self.rho = phys_tools.larmor_radius(self.q, self.m, self.T0, B)
        self.mu0 = self.T0 / B
        self.epsilon = phys_tools.plasma_permittivity(self.n0, self.q, self.rho, self.T0)
        self.gyrate = True

    def info(self):
        """Display species information and related parameters"""
        print(f"Species: {self.name}")
        print(f"Mass (m): {self.m:.3e} kg")
        print(f"Charge (q): {self.q:.3e} C")
        print(f"Initial Temperature (T0): {self.T0/phys_tools.eV:.3e} eV")
        print(f"Initial Density (n0): {self.n0:.3e} m^-3")
        print(f"Thermal Velocity (vth): {self.vt:.3e} m/s")
        if self.gyrate:
            print(f"Cyclotron Frequency (omega_c): {self.omega_c:.3e} rad/s")
            print(f"Larmor Radius (rho): {self.rho:.3e} m")
            print(f"Magnetic Moment (mu0): {self.mu0:.3e} J/T")
            print(f"Permittivity (epsilon): {self.epsilon:.3e}")
