# NumParam.py
class NumParam:
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
              f"  Nmu: {self.Nmu}\n")