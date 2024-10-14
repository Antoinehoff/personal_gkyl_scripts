import numpy as np
import sys,os
home_dir = os.path.expanduser("~")
sys.path.append(home_dir+'/personal_gkyl_scripts/python_utilities')
from tools.math_tools import custom_meshgrid, integral_xyz
class GBsource:
    """
    A class representing a Gaussian-Bell (GB) source model, used for describing
    plasma sources with Gaussian profiles in density and variable temperature models.
    """

    def __init__(self, n_srcGB, T_srcGB, x_srcGB, sigma_srcGB, bfac_srcGB, species,
                 dens_scale=1.0, temp_model="constant", dens_model="singaus", flux = None):
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
        self.nrate = n_srcGB           # Peak source density
        self.T     = T_srcGB           # Source temperature
        self.x     = x_srcGB           # Position of the source along the x-axis
        self.sigma = sigma_srcGB    # Gaussian width in the x-direction
        self.bfac  = bfac_srcGB     # Scaling factor for z-dependence
        self.temp_model = temp_model     # Temperature model selection
        self.dens_model = dens_model     # Density model selection
        self.dens_scale = dens_scale     # Density scaling parameter
        self.spec       = species
        self.flux       = flux

    def adapt_nrate(self,sim):
        #-We first normalize our source by its volume integral
        #-setup auxiliary grids for trapezoidal integration
        [x,y,z] = sim.geom_param.get_conf_grid()
        [X,Y,Z] = custom_meshgrid(x,y,z)
        #-evaluate the integrant on the mesh
        integrant = self.dens_source(X,Y,Z)*sim.geom_param.Jacobian
        #-Compute the volume integral
        integral = integral_xyz(x,y,z,integrant)
        # print(integral) # this has to be compared with after the normalization
        #-Normalize the source amplitude by the integral
        self.nrate /= integral
        #-We can perform again the volume integral to verify that it is equal to one
        # integrant = self.dens_source(X,Y,Z)*sim.geom_param.Jacobian
        # integral = integral_xyz(x,y,z,integrant)
        # print(integral) # this is equal to 1.0 now
        #-We now scale the nrate to match the initial particle loss rate
        #-Profile initial conditions
        def nT_IC(x, specie):
            n0 = specie.n0
            T0 = specie.T0
            dens = n0 * (0.5 * (1 + np.tanh(3 * (0.1 - 10 * x))) + 0.01)
            Temp = 6 * T0 * (0.5 * (1 + np.tanh(3 * (-0.1 - 10 * x))) + 0.01)
            return [dens,Temp]
        #-Get the inner radius IC value
        [n_x0, T_x0] = nT_IC(0,self.spec)
        pperp_x0 = 2.0/3.0*n_x0*T_x0 # pressure IC values
        #-Compute the initial ion loss rate from IC values at x=0
        GgB_x0,_,_ = sim.compute_GBloss(spec=self.spec,pperp_in=pperp_x0)
        #-Normalize the source amplitude by the integral
        self.nrate *= -GgB_x0

    def flux_model(self,x,y,z,b_in=-1):
        if self.dens_model == "singaus":
            if b_in == -1:
                b0 = self.bfac
            else:
                b0 = b_in
            flux = -np.sin(z) * np.exp(-np.power(np.abs(z), 1.5) / (2. * np.power(b0, 2.)))
        elif self.dens_model == "trugaus":
             if not (self.flux is None):
                flux = self.flux
             else:
                 raise ValueError(f"One must provide flux in attribute if using trugaus density model")
        else:
            raise ValueError(f"Unknown density model '{self.dens_model}'. \
                             Currently supported: 'singaus','trugaus'.")
        charge_sign = self.spec.q/np.abs(self.spec.q)
        return np.maximum(flux*charge_sign, 0.0)
    

    def dens_source(self, x, y, z):
        """
        Calculate the density source using a single Gaussian model in the x-direction.
        """
        # Gaussian profile in the x-direction (first envelope)
        env1 = np.exp(-np.power(x - self.x, 2.) / (2. * np.power(self.sigma, 2.)))
        # Model for the grad-B flux (b x gradB/B^2)
        env2 = self.flux_model(x,y,z)
        return self.nrate * env1 * env2

    def temp_source(self, x, z):
        return self.T

    def info(self):
        """
        Print detailed information about the GBsource parameters.
        """
        info_message = f"""
        GBsource Information:
        
        - Peak Density (n_srcGB): {self.nrate}
        - Temperature (T_srcGB): {self.T}
        - Position (x_srcGB): {self.x}
        - Width (sigma_srcGB): {self.sigma}
        - Z-dependence Scaling (bfac_srcGB): {self.bfac}
        - Temperature Model: {self.temp_model} (constant)
        - Density Model: {self.dens_model} (singaus)
        """
        print(info_message)