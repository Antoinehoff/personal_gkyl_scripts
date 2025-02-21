"""
math_tools.py

This file contains some useful functions for mathematical operations.

Functions:
- func_time_ave: Computes the time average of a list of arrays.
- func_calc_norm_fluc: Calculates normalized fluctuations.
- integral_vol: Computes the volume integral over x, y, and z.
- integral_surf: Computes the surface integral over x and y.
- custom_meshgrid: Creates a custom meshgrid with natural orientation (x, y, z).
- create_uniform_array: Creates a 1D array with uniform spacing.
- closest_index: Finds the closest index in an array to a given value.
- gradient: Computes the gradient of an array along a specified axis.

"""

import numpy as np
import scipy.integrate as scpy_int

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

def integral_vol(x,y,z,integrant_xyz):
    # Compute the volume integral (jacobian included in the integrand)
    integrant_xz  = np.trapz(integrant_xyz, x=x, axis=0)
    integrant_z   = np.trapz(integrant_xz,  x=y, axis=0)
    integral      = np.trapz(integrant_z,   x=z, axis=0)
    return integral

def integral_surf(x,y,integrant_xy):
    # Compute the line integral (jacobian included in the integrand)
    integral_y   = np.trapz(integrant_xy, x=x, axis=0)
    integral     = np.trapz(integral_y,   x=y, axis=0)
    return integral

def gradient(array,grid,axis):
    if(len(array[axis])>1):
        return np.gradient(array, grid, axis=axis)
    else:
        return np.zeros_like(array)
    
def integrate(function, a, b, args,  method='trapz', Np=16):
    if method == 'quad':
        return scpy_int.quad(function, a, b, args=args)
    elif 'trapz' in method:
        Np = method.replace('trapz','')
        Np = int(Np) if Np else 32
        x = np.linspace(a, b, Np)
        integrant = function(x,args)
        return np.trapz(integrant,x,axis=0), 0

def custom_meshgrid(x,y,z=0):
    # custom meshgrid function to have natural orientation (x,y,z)
    if np.isscalar(z):
        Y,X = np.meshgrid(y,x)
        return [X,Y]
    else:
        Y,X,Z = np.meshgrid(y,x,z)
        return [X,Y,Z]
    
def create_uniform_array(a, N):
    """
    Create a 1D array of size N that goes from a[0] to a[-1] with uniform spacing.
    
    Parameters:
    a (ndarray): Input 1D array.
    N (int): Desired size of the output array.
    
    Returns:
    b (ndarray): A 1D array of size N with uniform spacing from a[0] to a[-1].
    """
    # Generate a uniformly spaced array of size N between a[0] and a[-1]
    b = np.linspace(a[0], a[-1], N)
    return b

def closest_index(array,value):
    return np.abs(array - value).argmin()