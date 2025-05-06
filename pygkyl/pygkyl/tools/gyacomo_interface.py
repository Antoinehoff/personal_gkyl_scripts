import h5py
import numpy as np
import copy as cp
from ..tools import math_tools as tools

class GyacomoInterface:
    
    def __init__(self,filename):
        self.filename = filename
        self.load_grids()
        self.load_params()

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
            self.Lx     = 2*np.pi/self.kxgrid[1]
            self.Ly     = 2*np.pi/self.kygrid[1]
            self.Lz     = self.zgrid[-1]-self.zgrid[0]
            self.xgrid  = np.linspace(-self.Lx/2,self.Lx/2,self.Nx)
            self.ygrid  = np.linspace(-self.Ly/2,self.Ly/2,self.Ny)

    def load_group(self,group):
        data = {}
        with h5py.File(self.filename, 'r') as file:
            g_  = file[f"data/"+group]
            for key in g_.keys():
                name='data/'+group+'/'+key
                data[key] = file.get(name)[:]
        return data

    def load_3Dfield(self,field):
        data = {}
        with h5py.File(self.filename, 'r') as file:
            g_  = file[f"data/var3d/"+field]
            for key in g_.keys():
                name='data/var3d/'+field+'/'+key
                data[key] = file.get(name)[:]
        return data

    def load_params(self):
        jobid = self.filename[-5:-3]
        with h5py.File(self.filename, 'r') as file:
            nml_str = file[f"files/STDIN."+jobid][0]
            nml_str = nml_str.decode('utf-8')
            self.params = self.read_namelist(nml_str)


    def load_data_0D(self, dname):
        with h5py.File(self.filename, 'r') as file:
            # Load time data
            time  = file['data/var0d/time']
            time  = time[:]
            var0D = file['data/var0d/'+dname]
            var0D = var0D[:]
        return time, var0D

    def load_data_3D_frame(self,dname,tframe, xyz=True, species='ion'):
        with h5py.File(self.filename, 'r') as file:
            # load time
            time  = file['data/var3d/time']
            time  = time[:]
            # find frame
            iframe = tools.closest_index(time,tframe)
            tf     = time[iframe]
            # Load data
            try:
                data = file[f'data/var3d/{dname}/{iframe:06d}']
            except:
                g_ = file[f'data/var3d/']
                print('Dataset: '+f'data/var3d/{dname}/{iframe:06d}'+' not found')
                print('Available fields: ')
                msg = ''
                for key in g_:
                    msg = msg + key + ', '
                print(msg)
                exit()
            # Select the first species for species dependent fields
            ispecies = 1 if species == 'elc' else 0
            if not (dname == 'phi' or dname == 'psi'):
                data = data[:,:,:,ispecies]
            else:
                data = data[:]
            data = data['real']+1j*data['imaginary'] 
            if xyz:
                data_real = np.zeros([self.Nx,self.Ny,self.Nz])
                for iz in range(self.Nz):
                    data_real[:,:,iz] = tools.zkxky_to_xy_const_z(data, iz)
                data = data_real
        return time,data,tf

    def load_data_5D_frame(self,dname,tframe):
        with h5py.File(self.filename, 'r') as file:
            # load time
            time  = file['data/var5d/time']
            time  = time[:]
            # find frame
            iframe = tools.closest_index(time,tframe)
            tf     = time[iframe]
            # Load data
            try:
                data = file[f'data/var5d/{dname}/{iframe:06d}']
            except:
                g_ = file[f'data/var5d/']
                print('Dataset: '+f'data/var5d/{dname}/{iframe:06d}'+' not found')
                print('Available fields: ')
                msg = ''
                for key in g_:
                    msg = msg + key + ', '
                print(msg)
                exit()
            # Select the first species for species dependent fields
            data = data[:]
            data = data['real']+1j*data['imaginary'] 
        return time,data,tf

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