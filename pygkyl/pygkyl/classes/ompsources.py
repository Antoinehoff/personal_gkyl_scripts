import numpy as np
import matplotlib.pyplot as plt

class OMPsources:
    """
    Manages the setup and evaluation of OMP source profiles.

    Methods:
    - __init__: Initializes the OMPsources object with the required parameters.
    - gaussian_density: Gaussian profile for density.
    - exponential_density: Exponential profile for density.
    - default_temp_elc: Default temperature profile for electrons.
    - default_temp_ion: Default temperature profile for ions.
    - density_srcOMP: Evaluates the density source profile.
    - temp_elc_srcOMP: Evaluates the electron temperature source profile.
    - temp_ion_srcOMP: Evaluates the ion temperature source profile.
    - plot: Plots the source profiles.
    - info: Prints out the current configuration of the OMP source profiles.

    """ 
    def __init__(self, n_srcOMP, x_srcOMP, Te_srcOMP, Ti_srcOMP, sigma_srcOMP, floor_src, 
                 density_src_profile="default", temp_src_profile_elc="default", temp_src_profile_ion="default"):
        """
        Initializes the OMP source analysis with the required parameters.

        Parameters:
        n_srcOMP (float): Source density amplitude.
        x_srcOMP (float): Position for the source center.
        Te_srcOMP (float): Electron temperature source.
        Ti_srcOMP (float): Ion temperature source.
        sigma_srcOMP (float): Standard deviation of the source Gaussian.
        floor_src (float): Floor source value to avoid zero density.
        density_function (str or callable): Type of density function ('gaussian', 'exponential', or custom callable).
        """
        self.n_srcOMP     = n_srcOMP
        self.x_srcOMP     = x_srcOMP
        self.Te_srcOMP    = Te_srcOMP
        self.Ti_srcOMP    = Ti_srcOMP
        self.sigma_srcOMP = sigma_srcOMP
        self.floor_src    = floor_src

        # Assign the function type
        if callable(density_src_profile):
            self.density_function = density_src_profile
        elif density_src_profile in ["gaussian","default"]:
            self.density_function = self.gaussian_density
        elif density_src_profile == "exponential":
            self.density_function = self.exponential_density
        else:
            raise ValueError("Unsupported density function type. Use 'gaussian', 'exponential', or a callable.")

        if callable(temp_src_profile_elc):
            self.temp_elc_srcOMP = temp_src_profile_elc
        elif temp_src_profile_elc == "default":
            self.temp_elc_srcOMP = self.default_temp_elc
        else:
            raise ValueError("Unsupported elc temperature function type. Use 'default' or a callable.")

        if callable(temp_src_profile_ion):
            self.temp_ion_srcOMP = temp_src_profile_ion
        elif temp_src_profile_ion == "default":
            self.temp_ion_srcOMP = self.default_temp_ion
        else:
            raise ValueError("Unsupported ion temperature function type. Use 'default' or a callable.")

    def gaussian_density(self, x, y=None, z=None):
        """Gaussian profile for density."""
        return self.n_srcOMP * (np.exp(-((x - self.x_srcOMP) ** 2) / (2.0 * self.sigma_srcOMP ** 2)) + self.floor_src)

    def exponential_density(self, x, y=None, z=None):
        """Exponential profile for density."""
        return self.n_srcOMP * (np.exp(-np.abs(x - self.x_srcOMP) / self.sigma_srcOMP) + self.floor_src)
    
    def default_temp_elc(self, x, y = None, z = None):
        mask = x < (self.x_srcOMP + 3 * self.sigma_srcOMP)
        fout = np.empty_like(x)
        fout[mask] = self.Te_srcOMP
        fout[~mask] = self.Te_srcOMP * 3.0 / 8.0
        return fout  

    def default_temp_ion(self, x, y = None, z = None):
        mask = x < (self.x_srcOMP + 3 * self.sigma_srcOMP)
        fout = np.empty_like(x)
        fout[mask] = self.Ti_srcOMP
        fout[~mask] = self.Ti_srcOMP * 3.0 / 8.0
        return fout  
    
    def density_srcOMP(self, x, y, z):
        return self.density_function(x,y,z)

    def temp_elc_srcOMP(self,x,y,z):
        return self.default_temp_elc(x,y,z)

    def temp_ion_srcOMP(self,x,y,z):
        return self.default_temp_ion(x,y,z)
    
    def plot(self, x_grid, y_const, z_const):
        # Evaluate the source profiles
        density = self.density_srcOMP(x_grid, y_const, z_const)
        temp_elc = self.temp_elc_srcOMP(x_grid, y_const, z_const)
        temp_ion = self.temp_ion_srcOMP(x_grid, y_const, z_const)
        
        # Create the subplots
        fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        
        # Plot density in the first subplot
        axs[0].plot(x_grid, density, label="Density", color="black", linestyle="-")
        axs[0].set_title(r"Source Profiles at $y = {:.2f}$, $z = {:.2f}$".format(y_const, z_const))
        axs[0].set_ylabel(r"Density [1/$m^3$]")
        axs[0].grid(True, linestyle="--", alpha=0.7)
        
        # Plot temperatures in the second subplot
        axs[1].plot(x_grid, temp_elc, label="Electron", color="blue", linestyle="--")
        axs[1].plot(x_grid, temp_ion, label="Ion", color="red", linestyle="-.")
        axs[1].set_xlabel(r"x-grid [m]")
        axs[1].set_ylabel(r"Temperature [J]")
        axs[1].grid(True, linestyle="--", alpha=0.7)
        axs[1].legend()
        
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    def info(self):
        """
        Prints out the current configuration of the OMP source profiles.
        """
        print(f"Source Density Amplitude (n_srcOMP): {self.n_srcOMP}")
        print(f"Source Center Position (x_srcOMP): {self.x_srcOMP}")
        print(f"Electron Temperature Source (Te_srcOMP): {self.Te_srcOMP}")
        print(f"Ion Temperature Source (Ti_srcOMP): {self.Ti_srcOMP}")
        print(f"Source Gaussian Standard Deviation (sigma_srcOMP): {self.sigma_srcOMP}")
        print(f"Floor Source Value (floor_src): {self.floor_src}")
        print(f"Density Function: {self.density_function.__name__}")
        print(f"Electron Temperature Function: {self.temp_elc_srcOMP.__name__}")
        print(f"Ion Temperature Function: {self.temp_ion_srcOMP.__name__}")