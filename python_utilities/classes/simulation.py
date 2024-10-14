import numpy as np
import postgkyl as pg
from .numparam import NumParam
from .physparam import PhysParam
from .dataparam import DataParam
from .species   import Species
from .geomparam import GeomParam
from .gbsource  import GBsource
from .frame     import Frame
import copy
class Simulation:
    def __init__(self):
        """
        Initialize all attributes to None and setup normalization.
        """
        self.phys_param = None  # Physical parameters (e.g., constants like eps0, eV)
        self.num_param  = None  # Numerical parameters (e.g., grid size)
        self.data_param = None  # Data parameters (e.g., file paths)
        self.geom_param = None  # Geometric parameters (e.g., axis positions)
        self.GBsource   = None  # Source model of the simulation
        self.species    = {}    # Dictionary of species (e.g., ions, electrons)
        self.normalization = None
        self.norm_log   = []    # Normalization log to output normalization info

    def set_phys_param(self, eps0, eV, mp, me, B_axis):
        """
        Set physical parameters like permittivity, electron volts, masses, and magnetic field.
        """
        self.phys_param = PhysParam(eps0, eV, mp, me, B_axis)
    
    def set_geom_param(self, R_axis, Z_axis, R_LCFSmid, a_shift, q0, kappa, delta, x_LCFS, geom_type='Miller'):
        """
        Set geometric parameters related to the shape and size of the plasma (e.g., axis positions, LCFS).
        """
        self.geom_param = GeomParam(
            R_axis=R_axis, Z_axis=Z_axis, R_LCFSmid=R_LCFSmid, 
            a_shift=a_shift, q0=q0, kappa=kappa, delta=delta, 
            x_LCFS=x_LCFS, geom_type=geom_type, B0=self.phys_param.B_axis
        )

    def set_data_param(self, expdatadir, g0simdir, simname, simdir, fileprefix, 
                       wkdir, BiMaxwellian=True, species = {}):
        """
        Set data parameters like directories for experimental and simulation data, file names, and options.
        """
        self.data_param = DataParam(
            expdatadir, g0simdir, simname, simdir, 
            fileprefix, wkdir, BiMaxwellian, species
        )
        self.set_num_param()  # Automatically set numerical parameters based on data

    def set_num_param(self):
        """
        Set numerical parameters like grid size by loading and processing data.
        """
        data = pg.data.GData(self.data_param.datadir + 'xyz.gkyl')
        normgrids = data.get_grid()
        normx, normy, normz = normgrids[0], normgrids[1], normgrids[2]
        
        Nx = (normx.shape[0] - 2) * 2  # Double resolution in x
        Ny = (normy.shape[0] - 2) * 2  # Double resolution in y
        Nz = normz.shape[0] * 4        # Increase resolution in z
        
        self.num_param = NumParam(Nx, Ny, Nz, Nvp=None, Nmu=None)  # Set numerical grid

    def set_GBsource(self, n_srcGB, T_srcGB, x_srcGB, sigma_srcGB, bfac_srcGB, species, 
                     temp_model="constant", dens_model="singaus"):
        """
        Set the gradB source moodel parameters (density, temperature, etc.).
        """
        self.GBsource = GBsource(
            n_srcGB=n_srcGB, T_srcGB=T_srcGB, x_srcGB=x_srcGB, 
            sigma_srcGB=sigma_srcGB, bfac_srcGB=bfac_srcGB, 
            temp_model=temp_model, dens_model=dens_model, species = species
        )

    def set_species(self, name, m, q, T0, n0):
        """
        Add a species (e.g., ion or electron) to the simulation, and compute its gyromotion features.
        """
        s_ = Species(name, m, q, T0, n0)
        s_.set_gyromotion(self.phys_param.B_axis)
        self.species[name] = s_
        # Update the normalization with all available species
        self.normalization = DataParam.get_default_units_dict(self.species) 

    def add_species(self, species):
        """
        Add an existing species object to the simulation and compute its gyromotion.
        """
        species.set_gyromotion(self.phys_param.B_axis)
        self.species[species.name] = species
        # Update the normalization with all available species
        self.normalization = DataParam.get_default_units_dict(self.species) 

    def get_c_s(self):
        """
        Calculate and return the ion sound gyroradius (rho_s).
        """
        return np.sqrt(self.species['elc'].T0 / self.species['ion'].m)
    
    def get_rho_s(self):
        """
        Calculate and return the ion sound gyroradius (rho_s).
        """
        return self.get_c_s()/self.species['ion'].omega_c
    
    def get_filename(self,fieldname,tf):
        dataname = self.data_param.data_files_dict[fieldname+'file']
        return "%s-%s_%d.gkyl"%(self.data_param.fileprefix,dataname,tf)
    
    def get_GBloss_t(self, spec, twindow, ix=0):
        """
        Compute the grad-B (GB) particle loss over time for a given species.
        """
        time, GBloss_t = [], []
        # Precompute vGB_x for the given flux surface
        self.geom_param.compute_bxgradBoB2()
        # Loop over time frames in twindow
        for tf in twindow:
            GBloss, t, _ = self.compute_GBloss(spec,tf,ix=0,compute_bxgradBoB2=False)
            # Append corresponding GBloss value and time
            GBloss_t.append(GBloss)
            time.append(t)
        return GBloss_t, time

    def compute_GBloss(self,spec,tf=-1,ix=0,compute_bxgradBoB2=True,pperp_in=-1):
        if compute_bxgradBoB2:
            self.geom_param.compute_bxgradBoB2()
        # Check if we provided pperp
        if pperp_in == -1:
            # Not provided, get pressure from the simulation data at tf
            field_name = 'pperp' + spec.name[0]
            frame = Frame(self, field_name, tf, load=True)
            # multiply by mass since temperature moment is stored as T/m
            # we also make sure to remove any normalization
            pperp = frame.values[ix, :, :] * spec.m * self.normalization[field_name+'scale']
            time  = frame.time
        else:
            # Take provided value
            pperp = pperp_in
            time  = None

        #--build the integrand
        #-total version (/!\ this compensate loss and gain)
        integrand = pperp*self.geom_param.bxgradBoB2[0, ix, :, :]/spec.q
        # the simulation cannot gain particle, so we consider only losses
        integrand[integrand > 0.0] = 0.0
        # Calculate GB loss for this time frame
        GBloss_z = np.trapz(integrand, x=self.geom_param.y, axis=0)
        GBloss   = np.trapz(GBloss_z, x=self.geom_param.z, axis=0)
        return GBloss, time, GBloss_z

    def reset_normalization(self,key):
        # Get the default dictionary
        default_dict = DataParam.get_default_units_dict(self.species)
        # allows to reset the normalization of key to the default value
        adds = ['scale','shift','symbol','units']
        for add in adds:
            self.normalization[key+add]  = default_dict[key+add]

    def set_normalization(self,key,scale,shift,symbol,units):
        # allows to set the normalization of key
        self.normalization[key+'scale']  = scale
        self.normalization[key+'shift']  = shift
        self.normalization[key+'symbol'] = symbol
        self.normalization[key+'units']  = units

    def normalize(self, key, norm):
        scale = 0
        ion = self.species['ion']
        #-- Time scale
        if norm == 'mus':
            scale  = 1e-6
            shift  = 0
            symbol = 't'
            units  = r'$\mu s$'
        elif norm in ['t v_{ti}/R', 't*vti/R', 'vti/R']:
            scale  = self.geom_param.R_axis / ion.vt
            shift  = 0
            symbol = r'$t v_{ti}/R$'
            units  = ''
        #-- Length scale
        elif norm in ['rho', 'minor radius']:
            scale  = self.geom_param.a_mid
            shift  = (self.geom_param.x_LCFS - self.geom_param.a_mid) / scale
            symbol = r'$\rho$'
            units  = ''
        elif norm in ['x/rho', 'rho_L', 'Larmor radius']:
            scale  = ion.rho
            shift  = 0
            symbol = r'$%s/\rho_s$' % key
            units  = ''
        elif norm in ['R-Rlcfs', 'LCFS shift', 'LCFS']:
            scale  = 1.0
            shift  = self.geom_param.x_LCFS
            symbol = r'$R-R_{LCFS}$'
            units  = 'm'
        #-- Velocity normalization
        elif norm in ['vt', 'thermal velocity']:
            for spec in self.species.values():
                if key == 'upar%s'%spec.nshort:
                    scale  = spec.vt
                    shift  = 0
                    symbol = r'$u_{\parallel %s}/v_{t0 %s}$'%(spec.nshort,spec.nshort)
                    units  = ''
        #-- Temperature normalization
        elif norm == 'eV':
            for spec in self.species.values():
                if key == 'T%s'%spec.name[0]:
                    scale  = self.phys_param.eV / spec.m
                    symbol = r'$T_%s$'%spec.name[0]
                elif key == 'Tpar%s'%spec.name[0]:
                    scale  = self.phys_param.eV / spec.m
                    symbol = r'$T_{\parallel %s}$'%spec.name[0]
                elif key == 'Tperp%s'%spec.name[0]:
                    scale  = self.phys_param.eV / spec.m
                    symbol = r'$T_{\perp %s}$'%spec.name[0]
            shift = 0
            units = 'eV'
            if not self.data_param.BiMaxwellian:
                scale /= self.phys_param.eV / 3.0
        #-- Preessure normalization
        elif norm == 'beta':
            mu0 = 4*np.pi*1e-7
            for spec in self.species.values():
                if key == 'ppar%s'%spec.nshort:
                    scale = 0.01*(self.geom_param.B0**2/(2*mu0)) * 1.0/spec.m
                    symbol = r'$\beta_{\parallel %s}$'%spec.nshort
                elif key == 'pperp%s'%spec.nshort:
                    scale = 0.01*(self.geom_param.B0**2/(2*mu0)) * 1.0/spec.m
                symbol = r'$\beta_{\perp %s}$'%spec.nshort
            shift = 0
            units = '%'
            if not self.data_param.BiMaxwellian:
                scale /= self.phys_param.eV / 3.0
        #-- Grouped normalization
        if key.lower() == 'temperatures':
            for spec in self.species.values():
                self.normalize(    'T%s'%spec.nshort, norm)
                self.normalize( 'Tpar%s'%spec.nshort, norm)
                self.normalize('Tperp%s'%spec.nshort, norm)
        elif key.lower() == 'fluid velocities':
            for spec in self.species.values():
                self.normalize('upar%s'%spec.nshort, norm)
        elif key.lower() == 'pressures':
            for spec in self.species.values():
                self.normalize('ppar%s'%spec.nshort,  norm)
                self.normalize('pperp%s'%spec.nshort, norm)

        else:
            #-- Apply normalization or handle unknown norm
            if scale != 0:
                self.set_normalization(key=key, scale=scale, shift=shift, symbol=symbol, units=units)
                self.norm_log.append(f'{key} is now normalized to {norm}')
            else:
                print(f"Warning: The normalization '{norm}' for '{key}'"+
                      " is not recognized. Please check the inputs or refer"+
                      " to the documentation for valid options:")
                self.norm_help()
    
    def norm_help(self):
        help_message = """
        Available Normalizations:
        
        1. **Time Normalizations**:
        - 'mus': Microseconds (µs)
        - 'vti/R': Time normalized by ion thermal velocity over major radius (t v_{ti}/R)
        
        2. **Length Normalizations**:
        - 'rho' or 'minor radius': Normalized to the minor radius (ρ)
        - 'x/rho' or 'rho_L' or 'Larmor radius': Normalized to the Larmor radius (ρ_L)
        - 'R-Rlcfs' or 'LCFS shift' or 'LCFS': Shift relative to 
            the Last Closed Flux Surface (R - R_LCFS)
        
        3. **Velocity Normalizations**:
        - 'thermal velocity' (when key is 'upare' or 'upari'): Parallel velocities 
            normalized by thermal velocity
        
        4. **Temperature Normalizations**:
        - 'eV': Energy in electron volts (eV), applicable to various temperature components:
            * 'Tpare': Parallel electron temperature (T_{\\parallel e})
            * 'Tperpe': Perpendicular electron temperature (T_{\\perp e})
            * 'Tpari': Parallel ion temperature (T_{\\parallel i})
            * 'Tperpi': Perpendicular ion temperature (T_{\\perp i})
        
        **Grouped Normalizations**:
        - 'temperatures': Normalizes all temperature components (Tpare, Tperpe, Tpari, Tperpi)
        - 'fluid velocities': Normalizes both parallel electron and ion velocities (upare, upari)

        Note: you can setup a custom normalization using the set_normalization routine, e.g.
                simulation.set_normalization(
                    key='ni', scale=ion.n0, shift=0, symbol=r'$n_i/n_0$', units=''
                    )
              sets a normalization of the ion density to n0
        """
        print(help_message)

    def norm_info(self):
        for msg in self.norm_log:
            print(msg)