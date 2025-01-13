import os
import numpy as np
import scipy as scp
import postgkyl as pg
from .numparam import NumParam
from .physparam import PhysParam
from .dataparam import DataParam
from .species import Species
from .geomparam import GeomParam
from .gbsource import GBsource
from .ompsources import OMPsources
from .frame import Frame
from ..tools import math_tools
import matplotlib.pyplot as plt  # Add this import if not already present

class Simulation:
    """
    Manages the setup, parameters, and data for a plasma simulation.

    Methods:
    - __init__: Initializes the Simulation object.
    - set_phys_param: Sets physical parameters like permittivity, electron volts, masses, and magnetic field.
    - set_geom_param: Sets geometric parameters related to the shape and size of the plasma.
    - set_data_param: Sets data parameters like directories for experimental and simulation data, file names, and options.
    - set_num_param: Sets numerical parameters like grid size by loading the xyz.gkyl file or the nodes.gkyl file.
    - set_GBsource: Sets the gradB source model parameters.
    - set_OMPsources: Sets the OMP source parameters.
    - set_species: Adds a species to the simulation and computes its gyromotion features.
    - add_species: Adds an existing species object to the simulation and computes its gyromotion.
    - get_c_s: Calculates and returns the ion sound speed.
    - get_rho_s: Calculates and returns the ion sound gyroradius.
    - get_filename: Constructs the filename for a given field and time frame.
    - get_GBloss_t: Computes the grad-B particle loss over time for a given species.
    - compute_GBloss: Computes the grad-B particle loss for a given species at a specific time frame.
    - get_source_power: Computes the input power from the source term analytical profile or diagnostic.
    - get_source_particle: Computes the input particle from the source term analytical profile or diagnostic.
    - reset_normalization: Resets the normalization of a key to the default value.
    - set_normalization: Sets the normalization of a key.
    - normalize: Normalizes a key based on the specified normalization type.
    - norm_help: Displays available normalizations.
    - norm_info: Displays the normalization log.
    - display_available_fields: Displays available fields and their symbols and units.
    - add_source: Adds a source to the simulation.
    - plot_all_sources: Plots the profiles of all sources in the sources dictionary.
    - source_info: Combines get_source_particle, get_source_power, and plot_sources to provide comprehensive source information.
    """
    def __init__(self,dimensionality='3x2v'):
        self.dimensionality = dimensionality # Dimensionality of the simulation (e.g., 3x2v, 2x2v)
        self.phys_param = None  # Physical parameters (e.g., constants like eps0, eV)
        self.num_param  = None  # Numerical parameters (e.g., grid size)
        self.data_param = None  # Data parameters (e.g., file paths)
        self.geom_param = None  # Geometric parameters (e.g., axis positions)
        self.GBsource   = None  # Source model of the simulation
        self.OMPsources = None  # Energy source analytical profiles
        self.species    = {}    # Dictionary of species (e.g., ions, electrons)
        self.normalization = None
        self.norm_log   = []    # Normalization log to output normalization info
        self.sources = {}  # Dictionary to store sources

    def set_phys_param(self, eps0, eV, mp, me, B_axis):
        """
        Set physical parameters like permittivity, electron volts, masses, and magnetic field.
        """
        self.phys_param = PhysParam(eps0, eV, mp, me, B_axis)
    
    def set_geom_param(self, R_axis=None, Z_axis=None, R_LCFSmid=None, a_shift=None, kappa=None, 
                       delta=None, x_LCFS=None, q0 = None, x_out = None, geom_type='Miller', qprofile='default'):
        """
        Set geometric parameters related to the shape and size of the plasma (e.g., axis positions, LCFS).
        """
        self.geom_param = GeomParam(
            R_axis=R_axis, Z_axis=Z_axis, R_LCFSmid=R_LCFSmid, 
            a_shift=a_shift, q0=q0, kappa=kappa, delta=delta, 
            x_LCFS=x_LCFS, geom_type=geom_type, B0=self.phys_param.B_axis,
            x_out = x_out, qprofile=qprofile
        )

    def set_data_param(self, simdir, fileprefix, expdatadir="", g0simdir="", simname="",
                       wkdir = "", BiMaxwellian=True, species = {}):
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
        Set numerical parameters like grid size by loading 
        the xyz.gkyl file or the nodes.gkyl file. Depends on the gkyl version...
        """
        # Define file paths
        file1 = os.path.join(self.data_param.datadir, 'xyz.gkyl')
        file2 = self.data_param.fileprefix+'-nodes.gkyl'
        # Check if 'xyz.gkyl' exists
        if os.path.isfile(file1):
            filename = file1
        # If not, check if 'nodes.gkyl' exists
        elif os.path.isfile(file2):
            filename = file2
        # If neither file is found, print a message
        else:
            print("Neither 'xyz.gkyl' nor 'PREFIX-nodes.gkyl' was found with prefix:")
            print(self.data_param.fileprefix)
        # Load
        data = pg.data.GData(filename)
        normgrids = data.get_grid()
        if len(normgrids) == 3:
            normx, normy, normz = normgrids[0], normgrids[1], normgrids[2]
        elif len(normgrids) == 2:
            normx, normz = normgrids[0], normgrids[1]
            normy = np.array([0])
        
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

    def set_OMPsources(self, n_srcOMP, x_srcOMP, Te_srcOMP, Ti_srcOMP, sigma_srcOMP, floor_src,
                       density_src_profile = "default", temp_src_profile_elc = "default", temp_src_profile_ion = "default"):
        self.OMPsources = OMPsources(
            n_srcOMP, x_srcOMP, Te_srcOMP, Ti_srcOMP, sigma_srcOMP, floor_src, 
            density_src_profile, temp_src_profile_elc, temp_src_profile_ion
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
    
    def get_GBloss_t(self, spec, twindow, ix=0, losstype='particle', integrate = False):
        """
        Compute the grad-B (GB) particle loss over time for a given species.
        """
        time, GBloss_t = [], []
        # Precompute vGB_x for the given flux surface
        self.geom_param.compute_bxgradBoB2()
        # Loop over time frames in twindow
        for tf in twindow:
            GBloss, t, _ = self.compute_GBloss(spec,tf,ix=0,compute_bxgradBoB2=False,losstype=losstype)
            # Append corresponding GBloss value and time
            GBloss_t.append(GBloss)
            time.append(t)

        if integrate:
            t_in_s = [t_*self.normalization['tscale'] for t_ in time] #time in second to integrate a per sec value
            GBloss_t = scp.integrate.cumtrapz(GBloss_t,x=t_in_s, initial=0)

        return GBloss_t, time

    def compute_GBloss(self,spec,tf=-1,ix=0,losstype='particle',compute_bxgradBoB2=True,pperp_in=-1,T_in=-1):
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

        if losstype == 'energy':
            if T_in == -1:
                # Not provided, get pressure from the simulation data at tf
                field_name = 'T' + spec.name[0]
                frame = Frame(self, field_name, tf, load=True)
                # multiply by mass since temperature moment is stored as T/m
                # we also make sure to remove any normalization and set in MJ
                Ttot = frame.values[ix, :, :] * spec.m * self.normalization[field_name+'scale']/1e6
            else:
                Ttot = T_in
        elif losstype == 'particle':
            Ttot = 1.0 # we cancel the temp multiplication
        elif not losstype in ['energy','particle']:
            print(f"Warning: The loss type '{losstype}' "+
                      " is not recognized. must be energy or particle")
            
        #--build the integrand
        #-total version (/!\ this compensate loss and gain)
        integrand = Ttot*pperp*self.geom_param.bxgradBoB2[0, ix, :, :]/spec.q
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
            symbol = r'$r/a$'
            units  = ''
        elif norm in ['x/rho', 'rho_L', 'Larmor radius']:
            scale  = ion.rho
            shift  = 0
            symbol = r'$%s/\rho_{0i}$' % key
            units  = ''
        elif norm in ['R-Rlcfs', 'LCFS shift', 'LCFS']:
            scale  = 1.0
            shift  = self.geom_param.x_LCFS
            symbol = r'$R-R_{LCFS}$'
            units  = 'm'
        elif norm in ['pi']:
            scale  = np.pi
            shift  = 0
            symbol = r'$z/\pi$'
            units  = ''
        #-- Velocity normalization
        elif norm in ['vt', 'thermal velocity']:
            for spec in self.species.values():
                if key == 'upar%s'%spec.nshort:
                    scale  = spec.vt
                    shift  = 0
                    symbol = r'$u_{\parallel %s}/v_{t0 %s}$'%(spec.nshort,spec.nshort)
                    units  = ''

        #-- Energy normalization
        elif norm == 'MJ':
            scale = 1e6
            for spec in self.species.values():
                if key == 'Wkin%s'%spec.name[0]:
                    symbol = r'$W_{k,%s}$'%spec.name[0]
                if key == 'Wflu%s'%spec.name[0]:
                    symbol = r'$W_{f,%s}$'%spec.name[0]
                elif key == 'Wpot%s'%spec.name[0]:
                    symbol = r'$W_{p,%s}$'%spec.name[0]
                elif key == 'Wtot%s'%spec.name[0]:
                    symbol = r'$W_{%s}$'%spec.name[0]
            if key == 'Wkin':
                symbol = r'$W_{k}$'
            if key == 'Wflu':
                symbol = r'$W_{f}$'
            if key == 'Wpot':
                symbol = r'$W_{p}$'
            if key == 'Welf':
                symbol = r'$W_{E}$'
            if key == 'Wtot':
                symbol = r'$W$'
            shift = 0
            units = r'MJ/m$^3$'
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
                scale = 0.01*(self.geom_param.B0**2/(2*mu0)) * 1.0/spec.m
                if key == 'ppar%s'%spec.nshort:
                    symbol = r'$\beta_{\parallel %s}$'%spec.nshort
                elif key == 'pperp%s'%spec.nshort:
                    symbol = r'$\beta_{\perp %s}$'%spec.nshort
                elif key == 'p%s'%spec.nshort:
                    symbol = r'$\beta_{%s}$'%spec.nshort
            shift = 0
            units = r'$\%$'
            if not self.data_param.BiMaxwellian:
                scale /= self.phys_param.eV / 3.0
        elif norm == 'Pa':
            for spec in self.species.values():
                scale = 1/spec.m
                if key == 'ppar%s'%spec.nshort:
                    symbol = r'$p_{\parallel %s}$'%spec.nshort
                elif key == 'pperp%s'%spec.nshort:
                    symbol = r'$p_{\perp %s}$'%spec.nshort
                elif key == 'p%s'%spec.nshort:
                    symbol = r'$p_{%s}$'%spec.nshort
            shift = 0
            units = r'Pa'
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
                self.normalize(    'p%s'%spec.nshort, norm)
                self.normalize( 'ppar%s'%spec.nshort,  norm)
                self.normalize('pperp%s'%spec.nshort, norm)

        elif key.lower() == 'energies':
            self.normalize('Wkin',norm)
            self.normalize('Wflu',norm)
            self.normalize('Wpot',norm)
            self.normalize('Welf',norm)
            self.normalize('Wtot',norm)
            for spec in self.species.values():
                self.normalize('Wkin%s'%spec.nshort,norm)
                self.normalize('Wflu%s'%spec.nshort,norm)
                self.normalize('Wpot%s'%spec.nshort,norm)
                self.normalize('Wtot%s'%spec.nshort,norm)

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

    def display_available_fields(self):
        default_dict = DataParam.get_default_units_dict(self.species)
        # Create a table to display the data
        print(f"| {'Quantity':<15} | {'Symbol':<30} | {'Units':<20} |")
        print(f"|{'-' * 17}|{'-' * 32}|{'-' * 22}|")

        for key in default_dict:
            if key.endswith('symbol'):  # Check for a specific type of key
                quantity = key[:-6]  # Remove the suffix to get the base name
                if not quantity in ['x','y','z','ky','wavelen','vpar','mu','t','fi']:
                    symbol = default_dict[f'{quantity}symbol']
                    units = default_dict.get(f'{quantity}units', 'N/A')
                    # components = default_dict.get(f'{quantity}compo', 'N/A')
                    # Format as a table row
                    print(f"| {quantity:<15} | {symbol:<30} | {units:<20} |")

    def add_source(self, name, source):
        """
        Add a source to the simulation.
        """
        self.sources[name] = source

    def plot_sources(self,y_const=0,z_const=0):
        """
        Plot the profiles of all sources in the sources dictionary using various cuts.
        """
        x_grid, y_grid, z_grid = self.geom_param.grids

        fig, axs = plt.subplots(len(self.sources), 3, figsize=(18, 4 * len(self.sources)), sharex='col')
        
        if len(self.sources) == 1:
            axs = [axs]  # Ensure axs is iterable if there's only one subplot
        
        for ax_row, (name, source) in zip(axs, self.sources.items()):
            # Plot density profiles
            ax_row[0].plot(x_grid, source.density_src(x_grid, y_const, z_const), label="Density", color="black", linestyle="-")
            ax_row[0].set_title(f"Density Profile: {name}")
            ax_row[0].set_ylabel(r"Density [1/m$^3$]")
            ax_row[0].grid(True, linestyle="--", alpha=0.7)
            
            # Plot electron temperature profiles
            ax_row[1].plot(x_grid, source.temp_profile_elc(x_grid, y_const, z_const), label="Electron Temp", color="blue", linestyle="--")
            ax_row[1].set_title(f"Electron Temp Profile: {name}")
            ax_row[1].set_ylabel("Temperature [J]")
            ax_row[1].grid(True, linestyle="--", alpha=0.7)
            
            # Plot ion temperature profiles
            ax_row[2].plot(x_grid, source.temp_profile_ion(x_grid, y_const, z_const), label="Ion Temp", color="red", linestyle="-.")
            ax_row[2].set_title(f"Ion Temp Profile: {name}")
            ax_row[2].set_ylabel("Temperature [J]")
            ax_row[2].grid(True, linestyle="--", alpha=0.7)
        
        axs[-1][0].set_xlabel("x-grid [m]")
        axs[-1][1].set_xlabel("x-grid [m]")
        axs[-1][2].set_xlabel("x-grid [m]")
        plt.tight_layout()
        plt.show()

    def get_source_power(self, type='profile'):
        [x, y, z] = self.geom_param.get_conf_grid()
        [X, Y, Z] = math_tools.custom_meshgrid(x, y, z)

        if type == 'profile':  # Compute the input power from the source term analytical profile
            integrant = np.zeros_like(X)
            for source in self.sources.values():
                integrant += 1.5 * source.density_profile(X, Y, Z) * source.temp_profile_elc(X, Y, Z)
                integrant += 1.5 * source.density_profile(X, Y, Z) * source.temp_profile_ion(X, Y, Z)
        else:  # Compute the input power from the source term diagnostic
            M2e = Frame(self, 'M2e_src', tf=0, load=True)
            M2i = Frame(self, 'M2i_src', tf=0, load=True)
            integrant = 0.5 * self.species['elc'].m * M2e.values + 0.5 * self.species['ion'].m * M2i.values

        # multiply by Jacobian
        integrant *= self.geom_param.Jacobian
        # Integrate source terms (volume or surface)
        if self.dimensionality == '3x2v':
            pow_in = math_tools.integral_vol(x, y, z, integrant)
            print("Total input power: %g kW" % (pow_in / 1e3))
        elif self.dimensionality == '2x2v':
            pow_in = math_tools.integral_surf(x, z, integrant[:, 0, :])
            print("Lineic input power: %g kW/m" % (pow_in / 1e3))
        return pow_in

    def get_source_particle(self, type='profile'):
        [x, y, z] = self.geom_param.get_conf_grid()
        [X, Y, Z] = math_tools.custom_meshgrid(x, y, z)

        if type == 'profile':  # Compute the input particle from the source term analytical profile
            integrant = np.zeros_like(X)
            for source in self.sources.values():
                integrant += source.density_profile(X, Y, Z)
        else:  # Compute the input particle from the source term diagnostic
            M0e = Frame(self, 'ne_src', tf=0, load=True)
            M0i = Frame(self, 'ni_src', tf=0, load=True)
            integrant = M0e.values + M0i.values

        # multiply by Jacobian
        integrant *= self.geom_param.Jacobian
        # Integrate source terms (volume or surface)
        if self.dimensionality == '3x2v':
            part_in = math_tools.integral_vol(x, y, z, integrant)
            print("Total input particle: %g part/s" % (part_in))
        elif self.dimensionality == '2x2v':
            part_in = math_tools.integral_surf(x, z, integrant[:, 0, :])
            print("Lineic input particle: %g part/s/m" % (part_in))
        return part_in

    def source_info(self, type='profile', y_const=0, z_const=0):
        """
        Combines get_source_particle, get_source_power, and plot_sources to provide comprehensive source information.
        
        Parameters:
        type (str): The type of source term ('profile' or 'diagnostic').
        y_const (float): The constant y value for the profiles.
        z_const (float): The constant z value for the profiles.
        """
        print("-- Source Informations --")
        self.get_source_particle(type=type)
        self.get_source_power(type=type)
        if len(self.sources) > 0:
            self.plot_sources(y_const=y_const, z_const=z_const)