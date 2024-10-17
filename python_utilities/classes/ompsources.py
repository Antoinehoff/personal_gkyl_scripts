import numpy as np
from tools import math_tools

class OMPsources:
    def __init__(self, n_srcOMP, x_srcOMP, Te_srcOMP, Ti_srcOMP, sigma_srcOMP, floor_src):
        """
        Initializes the OMP source analysis with the required parameters.

        Parameters:
        n_srcOMP (float): Source density amplitude.
        x_srcOMP (float): Position for the source center.
        Te_srcOMP (float): Electron temperature source.
        Ti_srcOMP (float): Ion temperature source.
        sigma_srcOMP (float): Standard deviation of the source Gaussian.
        floor_src (float): Floor source value to avoid zero density.
        """
        self.n_srcOMP     = n_srcOMP
        self.x_srcOMP     = x_srcOMP
        self.Te_srcOMP    = Te_srcOMP
        self.Ti_srcOMP    = Ti_srcOMP
        self.sigma_srcOMP = sigma_srcOMP
        self.floor_src    = floor_src

    def density_srcOMP(self,x,y,z):
        """
        Computes the source term for density using the Gaussian profile.

        Parameters:
        x (ndarray): Position values for which the density source is calculated.

        Returns:
        ndarray: Density source term.
        """
        fout = self.n_srcOMP * (np.exp(-((x - self.x_srcOMP) ** 2) / (2.0 * self.sigma_srcOMP ** 2)) + self.floor_src)
        return fout

    def temp_elc_srcOMP(self,x,y,z):
        """
        Computes the electron temperature source term based on the position.

        Parameters:
        x (ndarray): Position values for which the electron temperature source is calculated.

        Returns:
        ndarray: Electron temperature source term.
        """
        mask = x < (self.x_srcOMP + 3 * self.sigma_srcOMP)
        fout = np.empty_like(x)
        fout[mask] = self.Te_srcOMP
        fout[~mask] = self.Te_srcOMP * 3.0 / 8.0
        return fout

    def temp_ion_srcOMP(self,x,y,z):
        """
        Computes the ion temperature source term based on the position.

        Parameters:
        x (ndarray): Position values for which the ion temperature source is calculated.

        Returns:
        ndarray: Ion temperature source term.
        """
        mask = x < (self.x_srcOMP + 3 * self.sigma_srcOMP)
        fout = np.empty_like(x)
        fout[mask] = self.Ti_srcOMP
        fout[~mask] = self.Ti_srcOMP * 3.0 / 8.0
        return fout