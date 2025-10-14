"""
math_tools.py

This file contains some useful functions for mathematical operations.

Functions:
- func_time_ave: Computes the time average of a list of arrays.
- func_calc_norm_fluc: Calculates normalized fluctuations.
- integral_vol: Computes the volume integral over x, y, and z.
- integral_surf: Computes the surface integral over x and y.
- custom_meshgrid: Creates a custom meshgrid with natural orientation (x, y, z).
- adapt_size: Creates a 1D array with an adapted size, trying to keep the spacing.
- closest_index: Finds the closest index in an array to a given value.
- gradient: Computes the gradient of an array along a specified axis.

"""

import numpy as np
import scipy.integrate as scpy_int
from scipy.interpolate import griddata as sp_interp
from scipy.ndimage import uniform_filter

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
        if len(x.shape) > 1 and len(y.shape) > 1:
            return [x,y]
        Y,X = np.meshgrid(y,x)
        return [X,Y]
    else:
        if len(x.shape) > 1 and len(y.shape) > 1 and len(z.shape) > 1:
            return [x,y,z]
        Y,X,Z = np.meshgrid(y,x,z)
        return [X,Y,Z]
    
def adapt_size(a, N):
    """
    Create a 1D array of size N that goes from a[0] to a[-1] trying to keep the spacing.
    """
    # fit a polynomial to the data
    if len(a) < 2:
        return np.array([a[0]]*N)
    p = np.polyfit(np.arange(len(a)), a, deg=min(3,len(a)-1))
    # create a new array with the same spacing
    x_new = np.linspace(0, len(a)-1, N)
    a_new = np.polyval(p, x_new)
    return a_new

def zkxky_to_xy_const_z(array, iz):
    # Get shape of the phi array
    Nz, Nkx, Nky, = array.shape
    if iz < 0: #outboard midplane for negative iz
        iz = Nz // 2  # Using the middle value for z

    array = array[iz,:,:]
    array = kx_to_x(array,Nkx,-2)
    array = ky_to_y(array,Nky-1,-1)
    array = np.transpose(array)
    array = np.flip(np.fft.fftshift(array))
    return array.T
    
def closest_index(array, v):
    # Compute absolute differences between each element of the array and v
    absolute_diff = np.abs(array - v)
    
    # Find the index of the minimum difference
    closest_index = np.argmin(absolute_diff)
    closest_index = max(closest_index,1)
    return closest_index

def is_convertible_to_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def numpy_array_to_list(d):
    """
    Recursively convert NumPy arrays to lists within a dictionary.
    """
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, dict):
            d[key] = numpy_array_to_list(value)
    return d

def kx_to_x(var_kx, nx0, axis=-3):
    """ Perform inverse FFT on kx spectral direction of variable

    Note: The fft and ifft in python and GENE/IDL have the factor 1/N switched!
    :param var_kx: Variable in kx space
    :param nx0: Number of kx grid points
    :param axis: Which axis of var_kx is the x direction, by default the third last one
    :returns: variable in real x space
    """
    var_x = np.fft.ifft(nx0*var_kx, axis=axis)
    var_x = np.real_if_close(var_x, tol=1e5)
    return var_x


def ky_to_y(var_ky, nky0, axis=-2):
    """ Perform inverse FFT on ky spectral direction of variable

    The GENE data only include the non-negative ky components, so we need to use the real
    valued FFT routines
    Note: The fft and ifft in python and GENE/IDL have the factor 1/N switched!
    :param var_ky: Variable in kx space
    :param nky0: Number of ky grid points
    :param axis: Which axis of var_kx is the x direction, by default the third last one
    :returns: variable in real y space
    """
    # The GENE data only include the non-negative ky components, so we need to use the real
    # valued FFT routines
    var_y = np.fft.irfft(2*nky0*var_ky, n=2*nky0, axis=axis)
    var_y = np.real_if_close(var_y, tol=1e5)
    return var_y

def interp2D(x,y,fxy,r,t, method='cubic', periodicity=[False, False]):
    '''
    Interpolate a 2D function fxy defined on a grid of points (x,y) to new points (r,t).
    x,y are meshgrids of the same shape as fxy.
    r,t are not meshgrids.
    '''
    if periodicity[0]:
        # Extend the x array both sides
        dx = x[1,0]-x[0,0]
        dy = y[0,1]-y[0,0]
        x = np.concatenate((x[0:1,:] - dx, x, x[-2:-1,:] + dx),axis=0)
        y = np.concatenate((y[0:1,:] - dy, y, y[-2:-1,:] + dy),axis=0)
        fxy = np.concatenate((fxy[-2:-1,:], fxy, fxy[0:1,:]), axis=0)
    if periodicity[1]:
        # Extend the y array both sides
        dx = x[1,0]-x[0,0]
        dy = y[0,1]-y[0,0]
        x = np.concatenate((x[:,0:1] - dx, x, x[:,-2:-1] + dx),axis=1)
        y = np.concatenate((y[:,0:1] - dy, y, y[:,-2:-1] + dy),axis=1)
        fxy = np.concatenate((fxy[:,-2:-1], fxy, fxy[:,0:1]), axis=1)
            
    return sp_interp((x.flatten(),y.flatten()), fxy.flatten(), (r,t), method=method)

def smooth2D(array, kernel_size=3):
    """
    Smooth a 2D array using a simple moving average filter.

    Parameters:
    array (ndarray): Input 2D array to be smoothed.
    kernel_size (int): Size of the smoothing kernel (must be odd).

    Returns:
    smoothed_array (ndarray): Smoothed 2D array.
    """

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    
    smoothed_array = uniform_filter(array, size=kernel_size, mode='reflect')
    return smoothed_array