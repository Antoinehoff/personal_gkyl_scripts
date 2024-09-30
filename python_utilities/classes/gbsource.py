import numpy as np

class GBsource:
    """
    A class representing a Gaussian-Bell (GB) source model, used for describing
    plasma sources with Gaussian profiles in density and variable temperature models.
    """

    def __init__(self, n_srcGB, T_srcGB, x_srcGB, sigma_srcGB, 
                 bfac_srcGB, temp_model="constant", dens_model="singaus"):
        """
        Initialize the GBsource object with source parameters.
        
        Parameters:
        - n_srcGB     : Source density at the peak.
        - T_srcGB     : Source temperature (used for constant or quadratic models).
        - x_srcGB     : Source position along the x-axis.
        - sigma_srcGB : Width of the Gaussian profile in the x-direction.
        - bfac_srcGB  : Scaling factor for the z-dependence of the source.
        - temp_model  : Choose between 'constant' 
                        or 'quadratic' temperature models (default is 'constant').
        - dens_model  : Choose between 'singaus' (single Gaussian) 
                        or other density models (default is 'singaus').
        """
        self.n_srcGB = n_srcGB           # Peak source density
        self.T_srcGB = T_srcGB           # Source temperature
        self.x_srcGB = x_srcGB           # Position of the source along the x-axis
        self.sigma_srcGB = sigma_srcGB   # Gaussian width in the x-direction
        self.bfac_srcGB = bfac_srcGB     # Scaling factor for z-dependence
        self.temp_model = temp_model     # Temperature model selection
        self.dens_model = dens_model     # Density model selection

    def dens_source_singaus(self, x, z):
        """
        Calculate the density source using a single Gaussian model in the x-direction.
        
        Parameters:
        - x : x-coordinate of the point.
        - z : z-coordinate of the point.
        
        Returns:
        - The density source at (x, z) based on the Gaussian profile in x and 
          a sinusoidal attenuation in z.
        """
        # Gaussian profile in the x-direction
        env1 = np.exp(-np.power(x - self.x_srcGB, 2.) / (2. * np.power(self.sigma_srcGB, 2.)))
        
        # Sinusoidal profile with attenuation in the z-direction
        env2 = max(np.sin(z) * np.exp(-np.power(np.abs(z), 1.5) / (2. * np.power(self.bfac_srcGB, 2.))), 0.)

        # Return the product of the density, x-profile, and z-profile
        return self.n_srcGB * env1 * env2

    def temp_source_constant(self, x, z):
        """
        Retrieve the constant temperature source.
        
        Parameters:
        - x : x-coordinate of the point (ignored, since temperature is constant).
        - z : z-coordinate of the point (ignored, since temperature is constant).
        
        Returns:
        - The constant temperature of the source.
        """
        return self.T_srcGB

    def temp_source_quadratic(self, x, z):
        """
        Calculate the temperature source with a quadratic profile in x.
        
        Parameters:
        - x : x-coordinate of the point.
        - z : z-coordinate of the point.
        
        Returns:
        - The temperature at (x, z) based on a quadratic profile in x.
        """
        return self.T_srcGB * (1 - ((x - self.x_srcGB) ** 2) / (self.sigma_srcGB ** 2))

    def temp_source(self, x, z):
        """
        Main temperature source routine, which dynamically selects between the 
        constant or quadratic temperature model based on the 'temp_model' attribute.
        
        Parameters:
        - x : x-coordinate of the point.
        - z : z-coordinate of the point.
        
        Returns:
        - The temperature at (x, z) from the selected temperature model.
        """
        if self.temp_model == "constant":
            return self.temp_source_constant(x, z)
        elif self.temp_model == "quadratic":
            return self.temp_source_quadratic(x, z)
        else:
            raise ValueError(f"Unknown temperature model '{self.temp_model}'. \
                             Choose 'constant' or 'quadratic'.")

    def dens_source(self, x, z):
        """
        Main density source routine, which dynamically selects between different 
        density models based on the 'dens_model' attribute.
        
        Parameters:
        - x : x-coordinate of the point.
        - z : z-coordinate of the point.
        
        Returns:
        - The density at (x, z) from the selected density model.
        """
        if self.dens_model == "singaus":
            return self.dens_source_singaus(x, z)
        else:
            raise ValueError(f"Unknown density model '{self.dens_model}'. \
                             Currently supported: 'singaus'.")

    def info(self):
        """
        Print detailed information about the GBsource parameters.
        """
        info_message = f"""
        GBsource Information:
        
        - Peak Density (n_srcGB): {self.n_srcGB}
        - Temperature (T_srcGB): {self.T_srcGB}
        - Position (x_srcGB): {self.x_srcGB}
        - Width (sigma_srcGB): {self.sigma_srcGB}
        - Z-dependence Scaling (bfac_srcGB): {self.bfac_srcGB}
        - Temperature Model: {self.temp_model} (constant/quadratic)
        - Density Model: {self.dens_model} (singaus)
        """
        print(info_message)