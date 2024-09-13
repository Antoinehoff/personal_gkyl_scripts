# PhysParam.py
class PhysParam:
    def __init__(self, eps0, eV, mp, me, B_axis):
        self.eps0 = eps0      # Permittivity of free space
        self.eV = eV          # Elementary charge (eV)
        self.mp = mp          # Proton mass
        self.me = me          # Electron mass
        self.B_axis = B_axis  # Magnetic field strength