import os
import numpy as np
import scipy as scp
import postgkyl as pg
from .numparam import NumParam
from .physparam import PhysParam
from .dataparam import DataParam
from .geomparam import GeomParam
from .normalization import Normalization
from .frame import Frame
from ..tools import math_tools, phys_tools, DG_tools
from ..interfaces.flaninterface import FlanInterface
import copy

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
    - add_source: Adds a source to the simulation.
    - plot_all_sources: Plots the profiles of all sources in the sources dictionary.
    - source_info: Combines get_source_particle, get_source_power, and plot_sources to provide comprehensive source information.
    """
    def __init__(self,dimensionality='3x2v',porder=1,ptype='ser',code='gkeyll', flandatapath=None):
        self.dimensionality = dimensionality # Dimensionality of the simulation (e.g., 3x2v, 2x2v)
        self.cdim = int(dimensionality[0])  # Configuration space dimension
        self.ndim = int(dimensionality[0]) + int(dimensionality[2])  # Total dimension
        self.phys_param = PhysParam()  # Physical parameters (eps0, eV, mp, me)
        self.num_param  = NumParam()  # Numerical parameters (Nx, Ny, Nz, Nvp, Nmu)
        self.data_param = DataParam()  # Data parameters (e.g., file paths)
        self.geom_param = GeomParam()  # Geometric parameters (e.g., axis positions)
        self.species    = {}    # Dictionary of species (e.g., ions, electrons)
        self.normalization = None # Normalization units for the simulation data
        self.fields_info = {} # Dictionary to store field informations like symbols, units etc.
        self.sources = {}  # Dictionary to store sources
        self.polyOrder = porder
        self.basisType = ptype
        self.polprojInsets = None # Custom poloidal projection inset.
        self.code = code # Code used for the simulation (e.g., gkeyll or gyacomo)
        self.flandatapath = flandatapath  # Data from FLAN interface, if applicable
        self.flan = None
        self.flanframes = []
        self.gyac = None  # Gyacomo interface, if applicable

    def set_phys_param(self, eps0 = 8.854e-12, eV = 1.602e-19, mp = 1.673e-27, me = 9.109e-31):
        """
        Set physical parameters like permittivity, electron volts, masses, and magnetic field.
        """
        self.phys_param = PhysParam(eps0=eps0, eV=eV, mp=mp, me=me)
    
    def set_geom_param(self, R_axis=None, Z_axis=None, R_LCFSmid=None, a_shift=None, kappa=None, B_axis = None,
                       delta=None, x_LCFS=None, x_out = None, geom_type='Miller', qprofile_R='default', qfit = []):
        """
        Set geometric parameters related to the shape and size of the plasma (e.g., axis positions, LCFS).
        """
        self.geom_param = GeomParam(
            R_axis=R_axis, Z_axis=Z_axis, R_LCFSmid=R_LCFSmid, 
            a_shift=a_shift, kappa=kappa, delta=delta, 
            x_LCFS=x_LCFS, geom_type=geom_type, B_axis=B_axis,
            x_out = x_out, qprofile_R=qprofile_R, qfit = qfit,
            cdim = self.cdim
        )

    def set_data_param(self, simdir, fileprefix, expdatadir="", g0simdir="", simname="",
                       wkdir = "", species = {}, set_num_param=True, get_available_frames=True):
        """
        Set data parameters like directories for experimental and simulation data, file names, and options.
        """
        if simdir[-1] != '/': simdir += '/'
        self.data_param = DataParam(
            expdatadir=expdatadir, g0simdir=g0simdir, simname=simname, simdir=simdir, 
            prefix=fileprefix, wkdir=wkdir, species=species
        )
        if set_num_param:
            self.set_num_param()  # Automatically set numerical parameters based on data
        if get_available_frames:
            self.available_frames = self.data_param.get_available_frames(self) # Get available frames for the simulation

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
            print("Neither 'xyz.gkyl' nor 'PREFIX-nodes.gkyl' was found with prefix: "+ self.data_param.fileprefix)
        # Load
        data = pg.data.GData(filename)
        normgrids = data.get_grid()
        if len(normgrids) == 3:
            normx, normy, normz = normgrids[0], normgrids[1], normgrids[2]
        elif len(normgrids) == 2:
            normx, normz = normgrids[0], normgrids[1]
            normy = np.array([0])
        elif len(normgrids) == 1:
            normx = np.array([0])
            normy = np.array([0])
            normz = normgrids[0]
        
        Nx = (normx.shape[0] - 2) * 2  # Double resolution in x
        Ny = (normy.shape[0] - 2) * 2  # Double resolution in y
        Nz = normz.shape[0] * 4        # Increase resolution in z
        
        self.num_param = NumParam(Nx, Ny, Nz, Nvp=None, Nmu=None)  # Set numerical grid

    def add_species(self, species):
        """
        Add an existing species object to the simulation and compute its gyromotion.
        """
        species.set_gyromotion(self.geom_param.B0)
        self.species[species.name] = species
        # Update the normalization with all available species
        self.normalization = Normalization(self) 
        self.fields_info = self.normalization.dict

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
        dataname = self.data_param.file_info_dict[fieldname+'file']
        return "%s-%s_%d.gkyl"%(self.data_param.fileprefix,dataname,tf)
    
    def get_frame(self, fieldName, timeFrame, load=True):
        """
        Get the frame for a given field and time frame.
        """
        return Frame(self, fieldName, timeFrame, load=load)
    
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
            t_in_s = [t_*self.normalization.dict['tscale'] for t_ in time] #time in second to integrate a per sec value
            GBloss_t = scp.integrate.cumtrapz(GBloss_t,x=t_in_s, initial=0)

        return GBloss_t, time
    
    def set_flandata(self, path):
        self.flandatapath = path
        self.flan = FlanInterface(path)
        self.flanframes = self.flan.avail_frames

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
            pperp = frame.values[ix, :, :] * spec.m * self.normalization.dict[field_name+'scale']
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
                Ttot = frame.values[ix, :, :] * spec.m * self.normalization.dict[field_name+'scale']/1e6
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

    def add_source(self, name, source):
        self.sources[name] = copy.deepcopy(source)

    def get_source_power(self, profileORgkyldata='profile', remove_GB_loss=False):
        """
        Compute the input power from the source term.

        Parameters:
        type (str): Type of source term ('profile' or 'src_diag').
        remove_GB_loss (bool): Whether to remove the grad-B drift loss.

        Returns:
        pow_in (float): Input power.
        """

        [x, y, z] = self.geom_param.get_conf_grid()
        [X, Y, Z] = math_tools.custom_meshgrid(x, y, z)
        
        if (profileORgkyldata == 'profile'):  # Compute the input power from the source term analytical profile
            integrant = np.zeros_like(X)
            for source in self.sources.values():
                integrant += 1.5 * source.density_profile(X, Y, Z) * source.temp_profile_elc(X, Y, Z)
                integrant += 1.5 * source.density_profile(X, Y, Z) * source.temp_profile_ion(X, Y, Z)
        elif (profileORgkyldata == 'gkyldata'):  # Compute the input power from the source term diagnostic
            M2e = Frame(self, 'src_M2e', tf=0, load=True)
            M2i = Frame(self, 'src_M2i', tf=0, load=True)
            integrant = 0.5 * self.species['elc'].m * M2e.values + 0.5 * self.species['ion'].m * M2i.values
        else:
            raise ValueError("Invalid type. Choose 'profile' or 'gkyldata'.")
        # multiply by Jacobian
        integrant *= self.geom_param.Jacobian
        # Integrate source terms (volume or surface)
        if self.dimensionality == '3x2v':
            pow_in = math_tools.integral_vol(x, y, z, integrant)
            print("Total input power: %g kW" % (pow_in / 1e3))
        elif self.dimensionality == '2x2v':
            pow_in = math_tools.integral_surf(x, z, integrant[:, 0, :])
            print("Lineic input power: %g kW/m" % (pow_in / 1e3))

        if remove_GB_loss:
            # Compute the banana width to estimate how much of the source is lost to the grad-B drift
            banana_width = self.get_banana_width(x=0,z=0)
            # apply a filter to the integrant for all x < banana_width
            integrant_bw = integrant.copy()
            integrant_bw[X < banana_width] = 0.0
            if self.dimensionality == '3x2v':
                pow_bw_in = math_tools.integral_vol(x, y, z, integrant_bw)
            elif self.dimensionality == '2x2v':
                pow_bw_in = math_tools.integral_surf(x, z, integrant_bw[:, 0, :])
            print("Grad-B drift loss: %g %%" % (100 * (pow_in - pow_bw_in) / pow_in))
            print("(input power after grad-B drift: %g kW)" % (pow_bw_in / 1e3))

        return pow_in

    def get_source_particle(self, profileORgkyldata='profile', remove_GB_loss=False):
        """
        Compute the input particle from the source term.
        """
        [x, y, z] = self.geom_param.get_conf_grid()
        [X, Y, Z] = math_tools.custom_meshgrid(x, y, z)

        if profileORgkyldata == 'profile':  # Compute the input particle from the source term analytical profile
            integrant = np.zeros_like(X)
            for source in self.sources.values():
                integrant += source.density_profile(X, Y, Z)
        elif profileORgkyldata == 'gkyldata':  # Compute the input particle from the source term diagnostic
            M0e = Frame(self, 'n_srce', tf=0, load=True)
            M0i = Frame(self, 'n_srci', tf=0, load=True)
            integrant = M0e.values + M0i.values
        else:
            raise ValueError("Invalid type. Choose 'profile' or 'gkyldata'.")
        integrant *= self.geom_param.Jacobian

        if self.dimensionality == '3x2v':
            part_in = math_tools.integral_vol(x, y, z, integrant)
            print("Total input particle: %g part/s" % (part_in))
        elif self.dimensionality == '2x2v':
            part_in = math_tools.integral_surf(x, z, integrant[:, 0, :])
            print("Lineic input particle: %g part/s/m" % (part_in))

        if remove_GB_loss:
            banana_width = self.get_banana_width(x=0,z=0)
            integrant_bw = integrant.copy()
            integrant_bw[X < banana_width] = 0.0
            if self.dimensionality == '3x2v':
                part_bw_in = math_tools.integral_vol(x, y, z, integrant_bw)
            elif self.dimensionality == '2x2v':
                part_bw_in = math_tools.integral_surf(x, z, integrant_bw[:, 0, :])
            print("Grad-B drift loss: %g %%" % (100 * (part_in - part_bw_in) / part_in))
            print("(input particle after grad-B drift: %g part/s)" % (part_bw_in))

        return part_in

    def get_banana_width(self, x=0.0, z=0.0, spec='ion'):
        '''
        calculate the banana width of a particle.
        Parameters:
        simulation (pygkyl.simulations.simulation.Simulation): Simulation object.
        x (float): x-coordinate of the source location [m]. Default is 0.0.
        z (float): z-coordinate of the source location [m]. Default is 0.0.
        spec (str): Species name. Default is 'ion'.
        Returns:
        rho_b (float): Banana width of the particle [m].
        '''
        # get inner wall indices
        iw_ix = np.argmin(np.abs(x))
        iw_iy = 0
        iw_iz = np.argmin(np.abs(z))
        # get temperature of the source at the inner wall
        spec_short = spec[0]
        M2i = Frame(self, 'src_M2'+spec_short, tf=0, load=True, normalize=False)
        M0i = Frame(self, 'src_M0'+spec_short, tf=0, load=True, normalize=False)
        Ti_iw = self.species[spec].m * M2i.values[iw_ix, iw_iy, iw_iz] / M0i.values[iw_ix, iw_iy, iw_iz] 
        # get magnetic field at the inner wall
        Bfield = Frame(self, 'Bmag', tf=0, load=True).values[iw_ix, iw_iy, iw_iz]
        qfactor = self.geom_param.qprofile_x(0)
        epsilon = self.geom_param.get_epsilon()
        banana_width = phys_tools.banana_width(
            self.species[spec].q, self.species[spec].m, Ti_iw, Bfield, qfactor, epsilon)
        return banana_width
    
    def info(self):
        """
        Print the simulation information.
        """
        print("Simulation info:")
        print("geom_param:")
        print(self.geom_param.info())
        print("phys_param:")
        print(self.phys_param.info())
        print("num_param:")
        print(self.num_param.info())
        print("data_param:")
        print(self.data_param.info())
        print("species:")
        for spec in self.species.values():
            print(spec.info())
        print("normalization:")
        print(self.normalization.info())
        print("sources:")
        for source in self.sources.values():
            print(source.info())
        print("DG_basis:")
        print(self.DG_basis.info())
        
    def get_collision_times(self, Bfield=None, print_table=True):
        """
        Compute collision times between all species in the simulation.
        
        Parameters:
        -----------
        Bfield : float, optional
            Magnetic field strength [T]. If None, uses B_axis from geometry.
        print_table : bool, optional
            Whether to print a formatted table of collision times.
            
        Returns:
        --------
        dict : Dictionary containing collision frequencies and times between all species pairs.
        """
        if Bfield is None:
            if self.geom_param is None or self.geom_param.B0 is None:
                raise ValueError("No magnetic field provided and no B_axis in geometry parameters.")
            Bfield = self.geom_param.B0
        
        species_list = list(self.species.values())
        n_species = len(species_list)
        
        collision_data = {}
        
        # Compute collision frequencies and times for all species pairs (including self-collisions)
        for i, species_s in enumerate(species_list):
            for j, species_r in enumerate(species_list):
                # Compute collision frequency
                nu_sr = phys_tools.collision_freq(
                    species_s.n0, species_s.q, species_s.m, species_s.T0,
                    species_r.n0, species_r.q, species_r.m, species_r.T0,
                    Bfield
                )
                
                loglambda_sr = phys_tools.coulomb_logarithm(
                    species_s.n0, species_s.q, species_s.m, species_s.T0,
                    species_r.n0, species_r.q, species_r.m, species_r.T0,
                    Bfield
                )
                
                # Compute collision parameter
                nustar_sr = phys_tools.nustar(
                    species_s.n0, species_s.q, species_s.m, species_s.T0,
                    species_r.n0, species_r.q, species_r.m, species_r.T0,
                    Bfield, self.geom_param.q0, self.geom_param.R0, self.geom_param.r0
                )
                
                # Collision time is inverse of frequency
                tau_sr = 1.0 / nu_sr if nu_sr > 0 else np.inf
                
                pair_name = f"{species_s.name}-{species_r.name}"
                
                collision_data[pair_name] = {
                    'frequency': nu_sr,
                    'nustar': nustar_sr,
                    'loglambda': loglambda_sr,
                    'time': tau_sr
                }
        
        if print_table:
            print("\n" + "="*80)
            print("REFERENCE COLLISION TIMES BETWEEN SPECIES")
            print("="*80)
            print(f"Magnetic field: {Bfield:.2e} T")
            print("-"*80)
            
            # Print species information
            print("Species parameters:")
            for species in species_list:
                print(f"  {species.name}: n0={species.n0:.2e} m^-3, T0={species.T0/phys_tools.eV:.0f} eV")
            print("-"*80)
            
            print(f"{'Species Pair':<15} {'Frequency [s^-1]':<15} {'Time [s]':<15} {'tc_s0/R':<15} {'nu*':<15}")
            print("-"*80)
            
            # Sort by frequency (highest to lowest)
            sorted_pairs = sorted(collision_data.items(), key=lambda x: x[1]['frequency'], reverse=True)
            
            R_axis = self.geom_param.R_axis if self.geom_param else 1.0
            c_s0 = self.get_c_s()
            
            for pair, data in sorted_pairs:
                freq_str = f"{data['frequency']:.2e}"
                nustar_str = f"{data['nustar']:.2e}"
                coulomb_log_str = f"{data['loglambda']:.2f}"
                time_str = f"{data['time']:.2e}" if data['time'] != np.inf else "∞"
                
                tau_norm = data['time'] * c_s0 / R_axis
                tau_norm_str = f"{tau_norm:.2e}" if tau_norm != np.inf else "∞"
                
                print(f"{pair:<15} {freq_str:<15} {time_str:<15} {tau_norm_str:<15} {nustar_str:<15}")
            
            print("-"*80)
        
        return collision_data