import numpy as np
import pygkyl

eV = 1.602e-19 # elementary charge in coulombs
amu = 1.66e-27 # atomic mass unit in kg

# Gyroaveraging functions
def gamma0_ex(kperp, rho): 
    return np.exp(-(kperp*rho)**2)*np.i0((kperp*rho)**2) # Eq. (8) in Held et al. 2023

def gamma1_ex(kperp, rho): 
    return np.exp(-0.5*(kperp*rho)**2) # Eq. (7) in Held et al. 2023

def gamma2_ex(kperp, rho): 
    return -0.5*(kperp*rho)**2*np.exp(-0.5*(kperp*rho)**2)

def gamma3_ex(kperp, rho): 
    return ((kperp*rho)**4/4 - (kperp*rho)**2)*gamma1_ex(kperp,s)

def gamma1_pade(kperp, rho): 
    return 1.0/(1.0 + 0.5*(kperp*rho)**2) # Eq. (4a) in Held et al. 2023

def gamma0_pade(kperp, rho): 
    return 1.0/(1.0 + (kperp*rho)**2) # Eq. (4b) in Held et al. 2023

def pol_s(kperp, n, q, m, T, Bref, G0func=gamma0_ex):
    rho = pygkyl.phys_tools.larmor_radius(q, m, T, Bref)
    return q**2 * n / T * (1.0 - G0func(kperp, rho))

def pol_species(kperp, specie:pygkyl.Species, Bref, G0func=gamma0_ex):
    return pol_s(kperp, specie.n0, specie.q, specie.m, specie.T0, Bref, G0func)

def pol_op(kperp, n_list, q_list, m_list, T_list, Bref, G0func=gamma0_ex):
    polop = 0.0
    for n, q, m, T in zip(n_list, q_list, m_list, T_list):
        polop += pol_s(kperp, n, q, m, T, Bref, G0func)
    return polop

class Context:
    def __init__(self, species_list:pygkyl.Species=[], Bref=2.0, qref=1, Tref=100, nref=1e19, mref=2.014*amu, Nrho=150, N=96*2):
        self.species = species_list
        self.Bref = Bref # reference magnetic field in tesla
        self.qref = qref*eV # reference charge in coulombs
        self.Tref = Tref*eV # reference temperature in joules
        self.nref = nref # reference density in m^-3
        self.mref = mref # reference mass in kg
        self.Nrho = Nrho # number of Larmor radii to simulate
        self.N = N # number of grid points in k-space
        self.refresh()
        
    def refresh(self):
        self.rhoref = pygkyl.phys_tools.larmor_radius(self.qref, self.mref, self.Tref, self.Bref) # reference Larmor radius in meters
        self.Lperp = self.Nrho*self.rhoref
        self.kmin = 2.0*np.pi/self.Lperp # minimum wavenumber in 1/m
        self.kmax = self.N/2*self.kmin # maximum wavenumber in 1/m
        self.kgrid = np.linspace(self.kmin, self.kmax, self.N)
        for s in self.species:
            s.set_gyromotion(self.Bref)
        if len(self.species) > 0:
            self.rhogrid = np.linspace(0.9*min([s.rho for s in self.species]), 1.1*max([s.rho for s in self.species]), 100) # gyroradius grid in m
            self.ntot_ion = sum([s.n0 for s in self.species if s.q > 0])
            self.qtot_ion = sum([s.q * s.n0 for s in self.species if s.q > 0])
            self.qavg_ion = sum([s.q * s.n0 for s in self.species if s.q > 0])/self.ntot_ion
            self.Tavg_ion = sum([s.T0 * s.n0 for s in self.species if s.q > 0])/self.ntot_ion
            self.mavg_ion = sum([s.m * s.n0 for s in self.species if s.q > 0])/self.ntot_ion
            self.rhoavg_ion = pygkyl.phys_tools.larmor_radius(self.qtot_ion, self.mavg_ion, self.Tavg_ion, self.Bref)
        else:
            self.rhogrid = np.array([self.rhoref])
            self.ntot_ion = self.nref
            self.qtot_ion = self.qref
            self.Tavg_ion = self.Tref
            self.mavg_ion = self.mref
            self.rhoavg_ion = self.rhoref
    
    def add_species(self, species:pygkyl.Species):
        species.set_gyromotion(self.Bref)
        self.species.append(species)
        
    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.refresh()
        
    def species_set(self, specie_name, **kwargs):
        for s in self.species:
            if s.name == specie_name:
                for key, value in kwargs.items():
                    setattr(s, key, value)
        self.refresh()
        
    def species_get(self, specie_name, key):
        for s in self.species:
            if s.name == specie_name:
                return getattr(s, key)
        return None

    def polarization_op(self, kperp, G0func):
        T_list = [s.T0 for s in self.species]
        n_list = [s.n0 for s in self.species]
        q_list = [s.q for s in self.species]
        m_list = [s.m for s in self.species]
        return pol_op(kperp, n_list, q_list, m_list, T_list, self.Bref, G0func)

    def polarization_op_single_ion(self, kperp, n, q, m, T, G0func):
        return pol_op(kperp, [n], [q], [m], [T], self.Bref, G0func)
