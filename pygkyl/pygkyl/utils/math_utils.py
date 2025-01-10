"""
math_utils.py

This module provides various mathematical utilities.

Functions:
- func_time_ave: Computes the time average of a list of arrays.
- func_calc_norm_fluc: Calculates normalized fluctuations.
- integral_xyz: Computes the volume integral over x, y, and z.
- integral_yz: Computes the integral over y and z.
- custom_meshgrid: Creates a custom meshgrid with natural orientation (x, y, z).

"""

import numpy as np

def func_time_ave(listIn):
    arrayOut = np.array(listIn)
    arrayOut = np.mean(arrayOut,axis=0)
    return arrayOut

def func_calc_norm_fluc(data2d, dataAve, dataNorm, Nt, Ny, Nx):
    data2dTot = np.reshape(data2d, (Nt*Ny,Nx))
    dataAve2d = np.array([dataAve,]*(Nt*Ny))
    delt = data2dTot - dataAve2d

    sigma = np.sqrt(np.mean(delt**2,axis=0)) # rms of density fluctuations
    delt_norm = sigma/dataNorm
    return delt, delt_norm

def integral_xyz(x,y,z,integrant_xyz):
    # Compute the volume integral (jacobian included in the integrand)
    integrant_xz  = np.trapz(integrant_xyz, x=x, axis=0)
    integrant_z   = np.trapz(integrant_xz,  x=y, axis=0)
    integral      = np.trapz(integrant_z,   x=z, axis=0)
    return integral

def integral_yz(y,z,integrant_yz):
    # Compute the volume integral (jacobian included in the integrand)
    integrant_z   = np.trapz(integrant_yz, x=y, axis=0)
    integral      = np.trapz(integrant_z,  x=z, axis=0)
    return integral

def custom_meshgrid(x,y,z=0):
    # custom meshgrid function to have natural orientation (x,y,z)
    if np.isscalar(z):
        Y,X = np.meshgrid(y,x)
        return [X,Y]
    else:
        Y,X,Z = np.meshgrid(y,x,z)
        return [X,Y,Z]