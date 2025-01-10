# NumParam.py
class NumParam:
    """
    numparam.py

    Manages numerical parameters like grid size and domain limits.

    Methods:
    - __init__: Initializes the NumParam object with the required parameters.
    - info: Displays the information of the numerical parameters.

    """
    def __init__(self, Nx=None, Ny=None, Nz=None, Nvp=None, Nmu=None):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nvp = Nvp
        self.Nmu = Nmu
        self.Lx  = None
        self.Ly  = None
        self.Lz  = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None

    def info(self):
        """
        Display the information of the numerical parameters.
        """
        print(f"Numerical Parameters:\n"
              f"  Nx: {self.Nx}\n"
              f"  Ny: {self.Ny}\n"
              f"  Nz: {self.Nz}\n"
              f"  Nvp: {self.Nvp}\n"
              f"  Nmu: {self.Nmu}\n"
              f"  Lx: {self.Lx}\n"
              f"  Ly: {self.Ly}\n"
              f"  Lz: {self.Lz}\n"
              f"  x_min: {self.x_min}\n"
              f"  x_max: {self.x_max}\n"
              f"  y_min: {self.y_min}\n"
              f"  y_max: {self.y_max}\n"
              f"  z_min: {self.z_min}\n"
              f"  z_max: {self.z_max}\n")