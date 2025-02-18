# PhysParam.py
class PhysParam:
    """
    A class to store physical parameters.

    Attributes:
    eps0 : float
        Permittivity of free space (F/m).
    eV : float
        Elementary charge (eV).
    mp : float
        Proton mass (kg).
    me : float
        Electron mass (kg).
    """
    def __init__(self, eps0 = 8.854e-12, eV = 1.602e-19, mp = 1.673e-27, me = 9.109e-31):
        self.eps0 = eps0      # Permittivity of free space
        self.eV = eV          # Elementary charge (eV)
        self.mp = mp          # Proton mass
        self.me = me          # Electron mass

    def info(self):
        """
        Prints the physical parameters.
        """
        print(f"eps0 = {self.eps0}")
        print(f"eV = {self.eV}")
        print(f"mp = {self.mp}")
        print(f"me = {self.me}")