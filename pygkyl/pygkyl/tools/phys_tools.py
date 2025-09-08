"""
phys_tools.py

This file contains functions for calculating various physical quantities.

Functions:
- thermal_vel: Calculate the thermal velocity of a particle.
- gyrofrequency: Calculate the gyrofrequency of a particle.
- larmor_radius: Calculate the Larmor radius of a particle.
- banana_width: Calculate the banana width of a particle.

"""

import numpy as np

eV = 1.602e-19 # electron volt [J]
eps0 = 8.854e-12 # vacuum permittivity [F/m]
hbar = 6.62606896e-34/(2.0*np.pi)  # reduced Planck constant [J*s]

def thermal_vel(temperature, mass):
    '''
    Calculate the thermal velocity of a particle.
    Parameters:
    temperature (float): Temperature of the particle [J].
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
    temperature (float): Temperature of the particle [J].
    Bfield (float): Magnetic field strength [T].
    Returns:
    rho_L (float): Larmor radius of the particle [m].
    '''
    omega = gyrofrequency(charge, mass, Bfield)
    velocity = thermal_vel(temperature, mass)
    rho_L = velocity / omega
    return np.abs(rho_L)

def banana_width(charge, mass, temperature, Bfield, qfactor, epsilon):
    '''
    calculate the banana width of a particle.
    Parameters:
    charge (float): Charge of the particle [C].
    mass (float): Mass of the particle [kg].
    temperature (float): Temperature of the particle [J].
    Bfield (float): Magnetic field strength [T].
    qfactor (float): Safety factor.
    epsilon (float): Inverse aspect ratio.
    Returns:
    rho_b (float): Banana width of the particle [m].
    '''
    rho_L = larmor_radius(charge, mass, temperature, Bfield)
    rho_b = qfactor * rho_L / np.sqrt(epsilon)
    return rho_b

def plasma_frequency(density, charge, mass):
    '''
    Calculate the plasma frequency.
    Parameters:
    density (float): Density of the plasma [m^-3].
    charge (float): Charge of the particle [C].
    mass (float): Mass of the particle [kg].
    Returns:
    omega_p (float): Plasma frequency [rad/s].
    '''
    omega_p = np.sqrt( density * charge**2 / (eps0 * mass))
    return omega_p

def coulomb_logarithm(n_s, q_s, m_s, T_s, n_r, q_r, m_r, T_r, Bfield=0.0):
    '''
    Calculate the Coulomb logarithm for collision between species s and r.
    Parameters:
    n_s, n_r (float): Densities of species s and r [m^-3].
    q_s, q_r (float): Charges of species s and r [C].
    m_s, m_r (float): Masses of species s and r [kg].
    T_s, T_r (float): Temperatures of species s and r [J].
    Bfield (float): Magnetic field strength [T] (default: 0.0).
    Returns:
    log_lambda (float): Coulomb logarithm [dimensionless].
    '''
    # Reduced mass
    m_sr = m_s * m_r / (m_s + m_r)
    
    # Relative velocity squared
    u_squared = 3 * T_r / m_r + 3 * T_s / m_s
    u = np.sqrt(u_squared)
    
    # Sum over both species for plasma effects
    alpha_sum = 0.0
    for n, q, m, T in [(n_s, q_s, m_s, T_s), (n_r, q_r, m_r, T_r)]:
        omega_p_alpha = plasma_frequency(n, q, m)
        omega_c_alpha = gyrofrequency(q, m, Bfield)
        
        denominator = T / m + 3 * T_s / m_s
        alpha_sum += (omega_p_alpha**2 + omega_c_alpha**2) / denominator
    
    # Classical distance of closest approach
    d_classical = np.abs(q_s * q_r) / (4 * np.pi * eps0 * m_sr * u_squared)
    
    # Quantum distance
    # d_quantum = hbar / (2 * np.exp(0.5) * m_sr * u)
    
    # Maximum distance (for each array element)
    # d_max = np.maximum(d_classical, d_quantum)
    d_max = d_classical  # Using classical distance for simplicity
    
    # Coulomb logarithm
    argument = 1 + (alpha_sum**(-1)) * (d_max**(-2))
    log_lambda = 0.5 * np.log(argument)
    
    return log_lambda

def collision_freq(n_s, q_s, m_s, T_s, n_r, q_r, m_r, T_r, Bfield=0.0):
    '''
    Calculate the collision frequency between two species.
    Parameters:
    n_s, n_r (float): Densities of species s and r [m^-3].
    q_s, q_r (float): Charges of species s and r [C].
    m_s, m_r (float): Masses of species s and r [kg].
    T_s, T_r (float): Temperatures of species s and r [J].
    Bfield (float): Magnetic field strength [T] (default: 0.0).
    Returns:
    nu (float): Collision frequency [s^-1].
    '''
    log_lambda_sr = coulomb_logarithm(n_s, q_s, m_s, T_s, n_r, q_r, m_r, T_r, Bfield)
    
    # Thermal velocities
    v_ts = thermal_vel(T_s, m_s)
    v_tr = thermal_vel(T_r, m_r)
    
    
    nu_norm =  1 / m_s * (1/m_s + 1/m_r) * (q_s**2 * q_r**2 * log_lambda_sr) / (3 * (2*np.pi)**(3/2) * eps0**2)
    
    cross_factor = 2.0 if np.abs(m_s - m_r) / m_r >= 1e-16 else 1.0
    
    nu = cross_factor * nu_norm * n_r/(v_ts**2 + v_tr**2)**(3/2)
    
    return nu

def nustar(n_s, q_s, m_s, T_s, n_r, q_r, m_r, T_r, Bfield=0.0, q0=1.0, R0=1.0, r0=0.0):
    '''
    Calculate the collisional parameter nu*.
    Parameters:
    n_s, n_r (float): Densities of species s and r [m^-3].
    q_s, q_r (float): Charges of species s and r [C].
    m_s, m_r (float): Masses of species s and r [kg].
    T_s, T_r (float): Temperatures of species s and r [J].
    Bfield (float): Magnetic field strength [T] (default: 0.0).
    q0 (float): Reference safety factor [] (default: 1.0).
    R0 (float): Reference major radius [m] (default: 1.0).
    r0 (float): Reference minor radius [m] (default: 0.2).
    Returns:
    nu_star (float): Collisional parameter nu* [s^-1].
    '''
    
    nu = collision_freq(n_s, q_s, m_s, T_s, n_r, q_r, m_r, T_r, Bfield)
    v_ts = thermal_vel(T_s, m_s)
    
    epsilon = r0 / R0  # Inverse aspect ratio
    nu_star = nu * (q0 * R0 / (epsilon**(3/2)*v_ts))  # Collisional parameter
    
    return nu_star

def plasma_permittivity(n, q, rho, T):
    '''
    Calculate the plasma permittivity.
    Parameters:
    q (float): Charge of the particle [C].
    rho (float): Larmor radius of the particle [m].
    T (float): Temperature of the particle [J].
    Returns:
    epsilon (float): Plasma permittivity [F/m].
    '''
    return n * q**2 * rho**2 / T