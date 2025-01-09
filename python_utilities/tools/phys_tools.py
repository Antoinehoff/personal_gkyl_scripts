# This file contains functions for calculating various physical quantities
import numpy as np

def thermal_vel(temperature, mass):
    '''
    Calculate the thermal velocity of a particle.
    Parameters:
    temperature (float): Temperature of the particle [eV].
    mass (float): Mass of the particle [kg].
    Returns:
    vth (float): Thermal velocity of the particle [m/s].
    '''
    vth = np.sqrt(temperature / mass)
    return vth

def gyrofrequency(charge, mass, Bfield):
    '''
    calculate the gyrofrequency of a particle.
    Parameters:
    charge (float): Charge of the particle [C].
    mass (float): Mass of the particle [kg].
    Bfield (float): Magnetic field strength [T].
    Returns:
    omega (float): Gyrofrequency of the particle [rad/s].
    '''
    omega = charge * Bfield / mass
    return omega

def larmor_radius(charge, mass, temperature, Bfield):
    '''
    calculate the larmor radius of a particle.
    Parameters:
    charge (float): Charge of the particle [C].
    mass (float): Mass of the particle [kg].
    temperature (float): Temperature of the particle [eV].
    Bfield (float): Magnetic field strength [T].
    Returns:
    rho_L (float): Larmor radius of the particle [m].
    '''
    omega = gyrofrequency(charge, mass, Bfield)
    velocity = thermal_vel(temperature, mass)
    rho_L = velocity / omega
    return rho_L

def banana_width(charge, mass, temperature, Bfield, qfactor, epsilon):
    '''
    calculate the banana width of a particle.
    Parameters:
    charge (float): Charge of the particle [C].
    mass (float): Mass of the particle [kg].
    temperature (float): Temperature of the particle [eV].
    Bfield (float): Magnetic field strength [T].
    qfactor (float): Safety factor.
    epsilon (float): Inverse aspect ratio.
    Returns:
    rho_b (float): Banana width of the particle [m].
    '''
    rho_L = larmor_radius(charge, mass, temperature, Bfield)
    rho_b = qfactor * rho_L / np.sqrt(epsilon)
    return rho_b