class Grid:
    def __init__(self, Nx, Ny, Nz, Nvp, Nmu):
        self.Nx = Nx    # Number of grid points in the x direction
        self.Ny = Ny    # Number of grid points in the y direction
        self.Nz = Nz    # Number of grid points in the z direction
        self.Nvp = Nvp  # Number of grid points in the vpar (parallel velocity) direction
        self.Nmu = Nmu  # Number of grid points in the mu direction

    def info(self):
        print(f"Grid Dimensions: Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}, Nvp={self.Nvp}, Nmu={self.Nmu}")