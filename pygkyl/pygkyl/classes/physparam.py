# PhysParam.py
class PhysParam:
    """
    A class to represent physical parameters.

    Attributes:
    eps0 : float
        Permittivity of free space.
    eV : float
        Elementary charge (eV).
    mp : float
        Proton mass.
    me : float
        Electron mass.
    B_axis : float
        Magnetic field strength.
    """
    def __init__(self, eps0, eV, mp, me, B_axis):
        self.eps0 = eps0      # Permittivity of free space
        self.eV = eV          # Elementary charge (eV)
        self.mp = mp          # Proton mass
        self.me = me          # Electron mass
        self.B_axis = B_axis  # Magnetic field strength