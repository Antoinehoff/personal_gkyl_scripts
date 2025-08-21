import h5py
import numpy as np
from ..tools import math_tools as tools
from ..classes.dataparam import DataParam
from ..classes.simulation import Simulation
from ..classes.species import Species
from ..projections.poloidalprojection import Inset
from ..classes.normalization import Normalization
from ..configs.vessel_data import d3d_vessel_data
import os
import matplotlib.pyplot as plt

class GyacomoInterface:
    # Gyacomo interface for reading data from Gyacomo HDF5 files
    params = None
    # Gyacomo normalization parameters
    l0 = 1.0 # perpendicular length scale
    R0 = 1.0 # parallel length scale
    u0 = 1.0 # velocity scale
    n0 = 1.0 # density scale
    T0 = 1.0 # temperature scale
    t0 = 1.0 # time scale
    phi0 = 1.0 # potential scale
    
    def __init__(self, simulation, simdir, simidx):
        self.simdir = simdir
        if isinstance(simidx, int):
            self.simidx = [simidx]
        else:
            self.simidx = simidx
        
        self.filename0 = self.simdir + f'/outputs_{self.simidx[0]:02d}.h5'
        self.file_no_ext = self.filename0[:-5]
        if not os.path.exists(self.filename0):
            raise FileNotFoundError(f"File {self.filename0} does not exist.")        
        
        self.field_map = {
            'phi': ('phi', '', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$\phi e / T_e$']),
            'psi': ('psi', '', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$\psi$']),
            'ni': ('dens', 'ion', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$n_i$']),
            'Ni00': ('Na00', 'ion', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$N_{i00}$']),
            'ne': ('dens', 'elc', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$n_e$']),
            'Ne00': ('Na00', 'elc', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$N_{e00}$']),
            'Tperpi': ('Tper', 'ion', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$T_{\perp i}$']),
            'Tperpe': ('Tper', 'elc', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$T_{\perp e}$']),
            'Tpari': ('Tpar', 'ion', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$T_{\parallel i}$']),
            'Tpare': ('Tpar', 'elc', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$T_{\parallel e}$']),
            'Ti': ('temp', 'ion', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$T_i$']),
            'Te': ('temp', 'elc', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$T_e$']),
            'upari': ('upar', 'ion', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$u_{\parallel i}$']),
            'upare': ('upar', 'elc', '3D', [r'$x/\rho_s$', r'$y/\rho_s$', r'$z/\rho_s$', r'$u_{\parallel e}$']),
            'Nipjz': ('Napjz', 'ion', '3D', [r'$p$', r'$j$', r'$z$', r'$N_{i}^{pj}$']),
            'Nepjz': ('Napjz', 'elc', '3D', [r'$p$', r'$j$', r'$z$', r'$N_e^{pj}$']),
        }
        self.field_map['field'] = self.field_map['phi']
        
        self.load_params()
        
        self.normalization = {}
        if simulation is not None:
            self.l0 = simulation.get_rho_s()
            self.R0 = simulation.geom_param.R0
            self.u0 = simulation.get_c_s()
            self.n0 = 1/self.l0/self.l0/self.R0
            self.T0 = simulation.species['elc'].T0
            self.t0 = self.R0 / self.u0
            self.phi0 = self.T0 / simulation.species['ion'].q
            self.load_grids()
            self.load_available_frames()  
            self.available_frames = {}
            for key in self.field_map:
                self.available_frames[key] = self.get_available_frames(key)
            
    def outfile(self,idx):
        # remove XX.h5 from filename0
        file_no_ext = self.filename0[:-5]
        # add the new index
        new_filename = file_no_ext + f'{idx:02d}.h5'
        return new_filename

    def load_grids(self):
        with h5py.File(self.filename0, 'r') as file:
            # Load the grids
            self.kxgrid = file[f'data/grid/coordkx'][:]
            self.kygrid = file[f'data/grid/coordky'][:]
            self.zgrid  = file[f'data/grid/coordz'][:]
            self.pgrid  = file[f'data/grid/coordp'][:]
            self.jgrid  = file[f'data/grid/coordj'][:]
            self.Nx     = self.kxgrid.size
            self.Nky    = self.kygrid.size
            self.Ny     = 2*(self.Nky-1)
            self.Nz     = self.zgrid.size
            self.Lx     = 2*np.pi/self.kxgrid[1] * self.l0
            self.Ly     = 2*np.pi/self.kygrid[1] * self.l0
            self.Lz     = self.zgrid[-1]-self.zgrid[0]
            self.xgrid  = np.linspace(-self.Lx/2,self.Lx/2,self.Nx+1)
            self.ygrid  = np.linspace(-self.Ly/2,self.Ly/2,self.Ny+1)
            
            # add an additional point to the z grid
            self.zgrid = np.append(self.zgrid, self.zgrid[-1] + (self.zgrid[-1] - self.zgrid[-2]))

    def load_group(self,group):
        data = {}
        with h5py.File(self.filename0, 'r') as file:
            g_  = file[f"data/"+group]
            for key in g_.keys():
                name='data/'+group+'/'+key
                data[key] = file.get(name)[:]
        return data

    def load_params(self):
        jobid = self.filename0[-5:-3]
        with h5py.File(self.filename0, 'r') as file:
            nml_str = file[f"files/STDIN."+jobid][0]
            nml_str = nml_str.decode('utf-8')
            self.params = self.read_namelist(nml_str)

    def load_available_frames(self):
        self.time0D = {'t': [], 'tidx': [], 'sidx': []}
        self.time3D = {'t': [], 'tidx': [], 'sidx': []}
        self.time5D = {'t': [], 'tidx': [], 'sidx': []}
        for sidx in self.simidx:
            with h5py.File(self.outfile(sidx), 'r') as file:
                time0D  = file['data/var0d/time']
                time0D  = time0D[:]
                time3D  = file['data/var3d/time']
                time3D  = time3D[:]
                time5D  = file['data/var5d/time']
                time5D  = time5D[:]
                
            self.time0D['t'].append(time0D)
            self.time0D['tidx'].append([j+1 for j in range(len(time0D))])
            self.time0D['sidx'].append([sidx for j in range(len(time0D))])
            
            self.time3D['t'].append(time3D)
            self.time3D['tidx'].append([j+1 for j in range(len(time3D))])
            self.time3D['sidx'].append([sidx for j in range(len(time3D))])
            
            self.time5D['tidx'].append([j+1 for j in range(len(time5D))])
            self.time5D['t'].append(time5D)
            self.time5D['sidx'].append([sidx for j in range(len(time5D))])
            
        self.time0D['t'] = np.concatenate(self.time0D['t'])
        self.time0D['tidx'] = np.concatenate(self.time0D['tidx'])
        self.time0D['sidx'] = np.concatenate(self.time0D['sidx'])
        
        self.time3D['t'] = np.concatenate(self.time3D['t'])
        self.time3D['tidx'] = np.concatenate(self.time3D['tidx'])
        self.time3D['sidx'] = np.concatenate(self.time3D['sidx'])
        
        self.time5D['t'] = np.concatenate(self.time5D['t'])
        self.time5D['tidx'] = np.concatenate(self.time5D['tidx'])
        self.time5D['sidx'] = np.concatenate(self.time5D['sidx'])

    def get_available_frames(self, fieldname):
        _, _, dimensionality, _ = self.field_map[fieldname]
        if dimensionality == '0D':
            return [i for i in range(len(self.time0D['t']))]
        elif dimensionality == '3D':
            return [i for i in range(len(self.time3D['t']))]
        elif dimensionality == '5D':
            return [i for i in range(len(self.time5D['t']))]
        else:
            raise ValueError(f"Unknown dimensionality: {dimensionality}")

    def load_data(self, fieldname, tframe, xyz=True):
        tframe = max(1, tframe)  # Ensure tframe is at least 1 (Fortran style)
        name, species, dimensionality, _ = self.field_map[fieldname]
        if dimensionality == '0D':
            return self.load_data_0D(name)
        elif dimensionality == '3D':
            return self.load_data_3D(name, tframe, xyz=xyz, species=species)
        elif dimensionality == '5D':
            return self.load_data_5D(name, tframe)
        else:
            raise ValueError(f"Unknown dimensionality: {dimensionality}")

    def load_data_0D(self, dname):
        spec_idx = 1 if dname[-1] == 'e' else 0
        dname = dname[:-1]
        time = []
        var0D = []
        for sidx in self.simidx:
            outfile = self.outfile(sidx)
            with h5py.File(outfile, 'r') as file:
                # Load time data
                t = file['data/var0d/time']
                time.append(t[:])
                nt = len(t[:])
                y = file['data/var0d/'+dname]
                y = np.reshape(y[:], (nt, 2))
                var0D.append(y[:,spec_idx])
        time = np.concatenate(time)
        var0D = np.concatenate(var0D)
                
        return [time], var0D

    def load_data_3D(self, dname, tidx, xyz=True, species='ion'):
        if isinstance(tidx, float):
            tidx = np.argmin(np.abs(self.time3D['t'] - tidx))
        time = self.time3D['t'][tidx]
        fidx = self.time3D['tidx'][tidx]
        sidx = self.time3D['sidx'][tidx]
        outfile = self.outfile(sidx)
        with h5py.File(outfile, 'r') as file:
            # load time
            t  = file['data/var3d/time']
            t  = t[:]
            # Load data
            try:
                data = file[f'data/var3d/{dname}/{fidx:06d}']
            except:
                g_ = file[f'data/var3d/']
                print('Dataset: '+f'data/var3d/{dname}/{fidx:06d}'+' not found in '+ outfile)
                print('Available fields: ')
                msg = ''
                for key in g_:
                    msg = msg + key + ', '
                print(msg)
            # Select the first species for species dependent fields
            ispecies = 1 if species == 'elc' else 0
            if not (dname == 'phi' or dname == 'psi'):
                data = data[:,:,:,ispecies]
            else:
                data = data[:]
            data = data['real']+1j*data['imaginary'] 
            grids = [self.kxgrid, self.kygrid, self.zgrid]
            if xyz:
                data_real = np.zeros([self.Nx,self.Ny,self.Nz])
                for iz in range(self.Nz):
                    data_real[:,:,iz] = tools.zkxky_to_xy_const_z(data, iz)
                data = data_real
                grids = [self.xgrid, self.ygrid, self.zgrid]
            
        return grids, time, data

    def load_data_5D(self,dname,tf):
        with h5py.File(self.filename0, 'r') as file:
            # load time
            time  = file['data/var5d/time']
            time  = time[:]
            # Load data
            try:
                data = file[f'data/var5d/{dname}/{tf:06d}']
            except:
                g_ = file[f'data/var5d/']
                print('Dataset: '+f'data/var5d/{dname}/{tf:06d}'+' not found')
                print('Available fields: ')
                msg = ''
                for key in g_:
                    msg = msg + key + ', '
                print(msg)
                exit()
            # Select the first species for species dependent fields
            data = data[:]
            data = data['real']+1j*data['imaginary'] 
            grids = [self.kxgrid, self.kygrid, self.zgrid, self.pgrid, self.jgrid]
        return grids, time[tf], data

    # Function to read all namelists from a file
    def read_namelist(self,nml_str):
        Nspecies = 1
        all_namelists = {}
        current_namelist = None
        nml_str = nml_str.split('\n')
        for line in nml_str:
            line = line.split('!')
            line = line[0]
            if line.startswith('&'):
                current_namelist = line[1:].strip()
                if current_namelist == 'SPECIES':
                    current_namelist = current_namelist + "_" + str(Nspecies)
                    Nspecies = Nspecies + 1
                all_namelists[current_namelist] = {}
            elif line.startswith('/'):
                current_namelist = None
            elif current_namelist:
                parts = line.split('=')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().rstrip(',').strip("'").strip()
                    if tools.is_convertible_to_float(value):
                        all_namelists[current_namelist][key] = float(value)
                    else:
                        all_namelists[current_namelist][key] = value
        return all_namelists

    def read_data_std(self,stdout):
        dict       = {"t":[],"Pxi":[],"Pxe":[],"Qxi":[],"Qxe":[]}
        with open(stdout, 'r') as file:
            for line in file:
                a = line.split('|')
                for i in a[1:-1]:
                    b = i.split('=')
                    dict[b[0].strip()].append(float(b[1]))
        return dict
    
    def process_fieldname(self, fieldname):

        if fieldname not in self.field_map:
            raise ValueError(f"Unknown fieldname: {fieldname}")

        name, species, dimensionality, symbols = self.field_map[fieldname]
        return name, species, dimensionality, symbols
    
    def adapt_data_param(self, simulation):
        data_param = DataParam(species=simulation.species, checkfiles=False)
        grid_names = ['x', 'y', 'z', 'p', 'j']
        for key in self.field_map:
            data_param.file_info_dict[key+'file'] = key
            data_param.file_info_dict[key+'compo'] = [key]
            data_param.file_info_dict[key+'comp'] = 0
            data_param.file_info_dict[key+'receipe'] = None
            data_param.file_info_dict[key+'gnames'] = grid_names[:3]
            data_param.file_info_dict[key+'symbol'] = self.field_map[key][3][-1]
            data_param.field_info_dict[key+'colormap'] = 'bwr'
        return data_param
    
    def adapt_normalization(self, simulation):
        normalization = Normalization(simulation) 
        grid_names = ['x', 'y', 'z', 'p', 'j']
        for key in self.field_map:
            normalization.dict[key+'file'] = key
            normalization.dict[key+'compo'] = [key]
            normalization.dict[key+'comp'] = 0
            normalization.dict[key+'receipe'] = None
            normalization.dict[key+'colormap'] = 'bwr'
            normalization.dict[key+'gnames'] = grid_names[:3]
            normalization.dict[key+'symbol'] = self.field_map[key][3][-1]
            normalization.dict[key+'units'] = ''
            normalization.dict[key+'colormap'] = 'bwr'

        normalization.change('t', self.t0/1.0e-6, 0, r'$t$', r'$\mu$s')
        normalization.change('x', 1.0, 0, r'$x$', r'm')
        normalization.change('y', 1.0, 0, r'$y$', r'm')   
        return normalization
    
    def set_mksa_normalization(self, normalization):
        normalization.change('phi', self.phi0, 0, r'$\phi$', 'V')
        
    def plot_fluxes(self, dnames):
        
        if isinstance(dnames, str):
            dnames = [[dnames]]
        elif isinstance(dnames, list) and all(isinstance(d, str) for d in dnames):
            dnames = [dnames]
        
        nsubplots = len(dnames)
        nrow = nsubplots // 2 + nsubplots % 2
        ncol = 2 if nsubplots > 1 else 1
        fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow), sharex=True)
        
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()
        elif nsubplots == 1:
            axs = [axs]
        spidx = 0  
        for spname in dnames:
            cidx = 0
            for name in spname:
                vlabel = self.plot_flux_single(name, axs[spidx])
                cidx += 1
            spidx += 1
            if cidx == 1:
                axs[0].set_ylabel(vlabel)
            else:
                axs[0].legend()
    
        xlabel = r'$t c_s/R_0$'
        axs[-1].set_xlabel(xlabel)
        
    def plot_flux_single(self, name, ax):
            if name not in ['gflux_xi', 'hflux_xi', 'pflux_xi', 'gflux_xe', 'hflux_xe', 'pflux_xe']:
                raise ValueError(f"Unknown 0D fieldname: {name}, only 'gflux_xs', 'hflux_xs', 'pflux_xs' (s=e,i), are supported.")
            grids, values = self.load_data_0D(name)
            time = grids[0]
            spec = name[-1]
            if name.startswith('gflux'):
                vlabel = r'$G_{x,e}$' if spec == 'e' else r'$G_{x,i}$'
            elif name.startswith('hflux'):
                vlabel = r'$Q_{x,e}$' if spec == 'e' else r'$Q_{x,i}$'
            elif name.startswith('pflux'):
                vlabel = r'$\Gamma_{p,e}$' if spec == 'e' else r'$\Gamma_{p,i}$'
            ax.plot(time, values, label=vlabel)
            
            return vlabel

def get_gyacomo_sim_config(configName,simdir,simidx):
    # Load the parameters of the simulation
    gyac = GyacomoInterface(None,simdir,simidx)
    params = gyac.params.copy()
    if 'multiscale' in configName:
        return get_gyacomo_multiscale_config(simdir,simidx,params)
    if 'cbc' in configName:
        return get_gyacomo_cbc_config(simdir,simidx,params)
    else:
        print("Use default Gyacomo CBC configuration.")
        return get_gyacomo_cbc_config(simdir,simidx,params)
        
def get_gyacomo_cbc_config(simdir,simidx,params):
    '''
    This function returns a simulation object for analyzing a Gyacomo simulation.
    '''
    R_axis = 1.7074685 # DIII-D
    B_axis = 2.5
    amid = 0.64
    R_LCFSmid = R_axis + amid
    r0 = 0.5*amid
    simulation = Simulation(dimensionality='3x2v', code='gyacomo')

    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )
    def qprofile(R):
        r = R - R_axis
        q0 = params['GEOMETRY']['q0']
        s0 = params['GEOMETRY']['shear']
        return q0 * (1 + s0 * (r - r0) / r0)

    simulation.set_geom_param(
        B_axis      = B_axis, # Magnetic field at magnetic axis [T]
        R_axis      = R_axis, # Magnetic axis major radius
        Z_axis      = 0.0, # Magnetic axis height (assumed zero)
        R_LCFSmid   = R_LCFSmid, # Major radius of LCFS at the midplane
        a_shift     = 0.0, # Parameter in Shafranov shift
        kappa       = params['GEOMETRY']['kappa'],
        delta       = params['GEOMETRY']['delta'],
        qprofile_R  = qprofile, # Safety factor
        x_LCFS      = 1.0, # position of the LCFS (dummy, will be updated later)
        x_out       = 0.0 # SOL domain width (gyacomo does not simulate SOL)
    )
    
    # Define the species
    # Temperature and density are taken from Greenfield et al. 1997, Nucl. Fusion 37 1215
    simulation.add_species(Species(name='ion',
                m=simulation.phys_param.mp, # Ion mass (proton), Deutrerium is 2.01410177811
                q=simulation.phys_param.eV,
                T0=1500*simulation.phys_param.eV, 
                n0=4e19))
    simulation.add_species(Species(name='elc',
                m=simulation.phys_param.me, 
                q=-simulation.phys_param.eV,
                T0=1500*simulation.phys_param.eV, 
                n0=4e19))
    
    Lx = params['GRID']['Lx'] * simulation.get_rho_s()
    Rcenter = R_axis + r0
    x_LCFS = R_LCFSmid - Rcenter
    x_in = R_LCFSmid - r0 - Lx / 2.0
    x_out = R_LCFSmid - r0 + Lx / 2.0
    simulation.geom_param.change(x_LCFS= x_LCFS, x_in=x_in, x_out=x_out)

    simulation.gyac = GyacomoInterface(simulation,simdir,simidx)
    
    simulation.available_frames = simulation.gyac.available_frames
    simulation.data_param = simulation.gyac.adapt_data_param(simulation=simulation)
    simulation.normalization = simulation.gyac.adapt_normalization(simulation=simulation)
    
    # Add a custom poloidal projection inset to position the inset according to geometry.
    simulation.polprojInsets = [
        Inset(
            lowerCornerRelPos=[0.4,0.3],
            xlim = [2.12,2.25],
            ylim = [-0.15,0.15],
            markLoc=[1,4])
    ]
    
    # Add discharge ID
    simulation.dischargeID = 'GYACOMO, Cyclone Base Case'
    
    # Add vessel data filename
    simulation.geom_param.vesselData = d3d_vessel_data

    # Add view points for the toroidal projection
    simulation.geom_param.camera_global = {
        'position':(2.3, 2.3, 0.75),
        'looking_at':(0, 0, 0),
            'zoom': 1.0
    }
    # Cameras for 1:2 formats
    simulation.geom_param.camera_zoom_1by2 = {   
        'position':(1.2, 1.2, 0.6),
        'looking_at':(0., 0.75, 0.1),
        'zoom': 1.0
    }

    return simulation

def get_gyacomo_multiscale_config(simdir,simidx,params):
    '''
    This function returns a simulation object for analyzing a Gyacomo simulation.
    '''
    R_axis = 1.7074685 # DIII-D
    B_axis = 2.5
    amid = 0.64
    R_LCFSmid = R_axis + amid
    r0 = 0.95*amid
    simulation = Simulation(dimensionality='3x2v', code='gyacomo')

    simulation.set_phys_param(
        eps0 = 8.854e-12,       # Vacuum permittivity [F/m]
        eV = 1.602e-19,         # Elementary charge [C]
        mp = 1.673e-27,         # Proton mass [kg]
        me = 9.109e-31,         # Electron mass [kg]
    )
    def qprofile(R):
        r = R - R_axis
        q0 = params['GEOMETRY']['q0']
        s0 = params['GEOMETRY']['shear']
        return q0 * (1 + s0 * (r - r0) / r0)

    simulation.set_geom_param(
        B_axis      = B_axis, # Magnetic field at magnetic axis [T]
        R_axis      = R_axis, # Magnetic axis major radius
        Z_axis      = 0.0, # Magnetic axis height (assumed zero)
        R_LCFSmid   = R_LCFSmid,   # Major radius of LCFS at the midplane
        a_shift     = 0.0, # Parameter in Shafranov shift
        kappa       = params['GEOMETRY']['kappa'],
        delta       = params['GEOMETRY']['delta'],
        qprofile_R  = qprofile, # Safety factor
        x_LCFS      = 1.0, # position of the LCFS (dummy, will be updated later)
        x_out       = 0.0 # SOL domain width (gyacomo does not simulate SOL)
    )
    
    # Define the species
    # Temperature and density are taken from Greenfield et al. 1997, Nucl. Fusion 37 1215
    simulation.add_species(Species(name='ion',
                m=simulation.phys_param.mp, # Ion mass (proton), Deutrerium is 2.01410177811
                q=simulation.phys_param.eV,
                T0=1500*simulation.phys_param.eV, 
                n0=4e19))
    simulation.add_species(Species(name='elc',
                m=simulation.phys_param.me, 
                q=-simulation.phys_param.eV,
                T0=1500*simulation.phys_param.eV, 
                n0=4e19))

    Lx = params['GRID']['Lx'] * simulation.get_rho_s()
    Rcenter = R_axis + r0
    x_LCFS = R_LCFSmid - Rcenter
    x_in = R_LCFSmid - r0 - Lx / 2.0
    x_out = R_LCFSmid - r0 + Lx / 2.0
    simulation.geom_param.change(x_LCFS= x_LCFS, x_in=x_in, x_out=x_out)
    
    simulation.gyac = GyacomoInterface(simulation,simdir,simidx)
    
    # Set up the flux tube size within the cartesian domain.
    simulation.geom_param.x_in = (amid - r0 + Lx/2.0)
    simulation.geom_param.x_LCFS = R_LCFSmid - (R_axis + r0)
    simulation.geom_param.x_out = -(amid - r0 - Lx/2.0)
    simulation.geom_param.update_geom_params()
    
    
    simulation.available_frames = simulation.gyac.available_frames
    simulation.data_param = simulation.gyac.adapt_data_param(simulation=simulation)
    simulation.normalization = simulation.gyac.adapt_normalization(simulation=simulation)
    
    # Add a custom poloidal projection inset to position the inset according to geometry.
    simulation.polprojInsets = [
        Inset(
            lowerCornerRelPos=[0.4,0.3],
            xlim = [2.2,2.3],
            ylim = [-0.05,0.05],
            zoom = 3.0,
            markLoc=[1,4])
    ]
    
    # Add discharge ID
    simulation.dischargeID = 'GYACOMO, DIII-D #186473'
    
    # Add vessel data filename
    simulation.geom_param.vesselData = d3d_vessel_data

    # Add view points for the toroidal projection
    simulation.geom_param.camera_global = {
        'position':(2.5, 2.52, 0.6),
        'looking_at':(0.0, -0.2, -0.2),
        'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_lower = {   
        'position':(0.83, 0.78, -0.1),
        'looking_at':(0., 0.74, -0.19),
        'zoom': 1.0
    }
    simulation.geom_param.camera_zoom_obmp = {
        'position':(0.4, 0.9, 0.0),
        'looking_at':(0.0, 0.98, 0.0),
            'zoom': 1.0
    }

    return simulation