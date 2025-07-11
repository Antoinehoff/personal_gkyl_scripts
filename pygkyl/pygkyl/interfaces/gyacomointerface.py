import h5py
import numpy as np
from ..tools import math_tools as tools
from ..classes.dataparam import DataParam
from ..classes.normalization import Normalization
import os

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
    V0 = 1.0 # potential scale
    
    def __init__(self,filename,simulation):
        self.filename = filename
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist.")        
        
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
        
        self.normalization = {}
        self.l0 = simulation.get_rho_s()
        self.R0 = simulation.geom_param.R0
        self.u0 = simulation.get_c_s()
        self.n0 = 1/self.l0/self.l0/self.R0
        self.T0 = simulation.species['elc'].T0
        self.t0 = self.R0 / self.u0
        self.V0 = self.T0 / simulation.species['ion'].q
        
        self.load_grids()
        self.load_params()
        self.load_available_frames()
        
        self.available_frames = {}
        for key in self.field_map:
            self.available_frames[key] = self.get_available_frames(key)
        

    def load_grids(self):
        with h5py.File(self.filename, 'r') as file:
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
        with h5py.File(self.filename, 'r') as file:
            g_  = file[f"data/"+group]
            for key in g_.keys():
                name='data/'+group+'/'+key
                data[key] = file.get(name)[:]
        return data

    def load_params(self):
        jobid = self.filename[-5:-3]
        with h5py.File(self.filename, 'r') as file:
            nml_str = file[f"files/STDIN."+jobid][0]
            nml_str = nml_str.decode('utf-8')
            self.params = self.read_namelist(nml_str)

    def load_available_frames(self):
        with h5py.File(self.filename, 'r') as file:
            self.time0D  = file['data/var0d/time']
            self.time0D  = self.time0D[:]
            self.time3D  = file['data/var3d/time']
            self.time3D  = self.time3D[:]
            self.time5D  = file['data/var5d/time']
            self.time5D  = self.time5D[:]

    def get_available_frames(self, fieldname):
        _, _, dimensionality, _ = self.field_map[fieldname]
        if dimensionality == '0D':
            return [i for i in range(1,len(self.time0D))]
        elif dimensionality == '3D':
            return [i for i in range(1,len(self.time3D))]
        elif dimensionality == '5D':
            return [i for i in range(1,len(self.time5D))]
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
        with h5py.File(self.filename, 'r') as file:
            # Load time data
            time  = file['data/var0d/time']
            time  = time[:]
            var0D = file['data/var0d/'+dname]
            var0D = var0D[:]
            grids = [time]
        return grids, var0D

    def load_data_3D(self, dname, tf, xyz=True, species='ion'):
        with h5py.File(self.filename, 'r') as file:
            # load time
            time  = file['data/var3d/time']
            time  = time[:]
            # Load data
            try:
                data = file[f'data/var3d/{dname}/{tf:06d}']
            except:
                g_ = file[f'data/var3d/']
                print('Dataset: '+f'data/var3d/{dname}/{tf:06d}'+' not found in '+self.filename)
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
            
            # exchange first and second dimensions of the data
            #data = np.transpose(data, (1, 0, 2))
        return grids, time[tf], data

    def load_data_5D(self,dname,tf):
        with h5py.File(self.filename, 'r') as file:
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
        normalization.change('phi', self.V0, 0, r'$\phi$', 'V')
        

        return normalization