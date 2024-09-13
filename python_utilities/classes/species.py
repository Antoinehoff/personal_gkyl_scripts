import numpy as np

class Species:
    def __init__(self, name, m, q, T0, n0):
        self.name = name # name
        self.m = m       # mass in kg
        self.q = q       # charge in C
        self.T0 = T0     # initial temperature in K
        self.n0 = n0     # initial density in m^-3
        self.vth = np.sqrt(T0 / m)  # thermal velocity vth = sqrt(T0/m)
        self.omega_c = None # Cyclotron frequency
        self.rho     = None # Larmor radius

    def set_gyromotion(self, B):
        self.omega_c = (self.q * B) / self.m
        self.rho = self.vth / self.omega_c

    
    def info(self):
        """Display species information and related parameters"""
        print(f"Species: {self.name}")
        print(f"Mass (m): {self.m:.3e} kg")
        print(f"Charge (q): {self.q:.3e} C")
        print(f"Initial Temperature (T0): {self.T0:.3e} K")
        print(f"Initial Density (n0): {self.n0:.3e} m^-3")
        print(f"Thermal Velocity (vth): {self.vth:.3e} m/s")
        