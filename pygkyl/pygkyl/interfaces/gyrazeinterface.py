import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

from ..classes import Frame, Simulation

class GyrazeInterface:
    def __init__(self, simulation:Simulation, **kwargs):
        self.simulation = simulation
        self.filter_negativity : bool = kwargs.get('filter_negativity', False)
        self.number_datasets : bool = kwargs.get('number_datasets', False)
        self.outfilename : str = kwargs.get('outfilename', 'data.h5')

        self.frames = self.simulation.available_frames['ion']
        self.nspec = len(self.simulation.species)
        self.me = self.simulation.species['elc'].m
        self.mi = self.simulation.species['ion'].m
        self.mioverme = self.mi/self.me
        self.e = np.abs(self.simulation.species['elc'].q)
        self.Bmag = None
        self.phi = None
        self.ne = None
        self.ni = None
        self.Te = None
        self.Ti = None
        self.fe = None
        self.fi = None
        self.gamma = None
        self.xgrid = None
        self.ygrid = None
        self.zgrid = None
        self.vparegrid = None
        self.muegrid = None
        self.vparigrid = None
        self.muigrid = None
        self.B0 = None
        self.phi0 = None
        self.ne0 = None
        self.ni0 = None
        self.Te0 = None
        self.Ti0 = None
        self.fe0 = None
        self.fi0 = None
        self.gamma0 = None
        self.nioverne = None
        self.TioverTe = None
        self.alphadeg = None
        self.vpare_norm = None
        self.mue_norm = None
        self.vpari_norm = None
        self.mui_norm = None
        self.fe_mpe_args_text = None
        self.fe_mpe_text = None
        self.fi_mpi_args_text = None
        self.fi_mpi_text = None
        self.input_physparams_text = None
        self.skip_point = False
        self.nsample = None
        self.nskipped = None

    def load_frames(self, timeframe):
        self.Bmag = Frame(simulation=self.simulation,fieldname='Bmag',tf=timeframe,load=True)
        self.phi = Frame(simulation=self.simulation,fieldname='phi',tf=timeframe,load=True)
        self.ne = Frame(simulation=self.simulation,fieldname='ne',tf=timeframe,load=True)
        self.ni = Frame(simulation=self.simulation,fieldname='ni',tf=timeframe,load=True)
        self.Te = Frame(simulation=self.simulation,fieldname='Te',tf=timeframe,load=True)
        self.Ti = Frame(simulation=self.simulation,fieldname='Ti',tf=timeframe,load=True)
        self.fe = Frame(simulation=self.simulation,fieldname='fe',tf=timeframe,load=True)
        self.fi = Frame(simulation=self.simulation,fieldname='fi',tf=timeframe,load=True)
        self.gamma = Frame(simulation=self.simulation,fieldname='rhoe_lambdaD',tf=timeframe,load=True)

    def eval_frames(self, x, y, z):
        self.B0 = self.Bmag.get_value([x, y, z])
        self.phi0 = self.phi.get_value([x, y, z])
        self.ne0 = self.ne.get_value([x, y, z])
        self.ni0 = self.ni.get_value([x, y, z])
        self.Te0 = self.Te.get_value([x, y, z])
        self.Ti0 = self.Ti.get_value([x, y, z])
        self.gamma0 = self.gamma.get_value([x, y, z])
        self.fe0 = self.fe.get_value([x, y, z, 'all', 'all'])
        self.fi0 = self.fi.get_value([x, y, z, 'all', 'all'])
        # skip this point if negativity is found
        if np.any([self.Ti0<0, self.Te0<0, self.ne0<0, self.ni0<0]):
            self.skip_point = True
            return
        if self.filter_negativity:
            self.fe0[self.fe0<0] = 0.0
            self.fi0[self.fi0<0] = 0.0
        self.nioverne = self.ni0/self.ne0
        self.TioverTe = self.Ti0/self.Te0
        self.vpare_norm = self.vparegrid / np.sqrt(self.Te0*self.e/self.me)
        self.mue_norm = self.muegrid / (self.Te0*self.e/self.B0)
        self.vpari_norm = self.vparigrid / np.sqrt(self.Ti0*self.e/self.mi)
        self.mui_norm = self.muigrid / (self.Ti0*self.e/self.B0)

    def get_grids(self):
        self.xgrid = self.fe.new_grids[0]
        self.ygrid = self.fe.new_grids[1]
        self.zgrid = self.fe.new_grids[2]

        self.vparegrid = self.fe.new_grids[3]
        self.muegrid = self.fe.new_grids[4]

        self.vparigrid = self.fi.new_grids[3]
        self.muigrid = self.fi.new_grids[4]

    def get_ranges(self, xmin, xmax, Nx, Ny, zplane):
        ixmin = np.argmin(np.abs(self.xgrid - xmin))
        ixmax = np.argmin(np.abs(self.xgrid - xmax))
        iymin = 0
        iymax = len(self.ygrid)-1
        if zplane=='upper':
            izplanes = [np.argmax(self.zgrid)]
        elif zplane=='lower':
            izplanes = [np.argmin(self.zgrid)]
        elif zplane=='both':
            izplanes = [np.argmin(self.zgrid), np.argmax(self.zgrid)]

        xindices = np.linspace(ixmin, ixmax, Nx, dtype=int)
        yindices = np.linspace(iymin, iymax, Ny, dtype=int)
        # remove duplicates
        xindices = np.unique(xindices)
        yindices = np.unique(yindices)
        return xindices, yindices, izplanes

    def write_F_mps_files(self, munorm, vpnorm, f0, species):
        species_str = 'e' if species=='elc' else 'i'
        with open(f'F{species_str}_mp{species_str}_args.txt', 'w') as f:
            f.write(' '.join(map(str, munorm)) + '\n')
            f.write(' '.join(map(str, vpnorm)))
        np.savetxt(f'F{species_str}_mp{species_str}.txt', f0.squeeze().T)

    def write_input_physparams_file(self):
        with open('input_physparams.txt', 'w') as f:
            f.write('#set type_distfunc_entrance (= ADHOC or other string)\n')
            f.write('GKEYLL\n')
            f.write('#set alphadeg\n')
            f.write(f'{self.alphadeg}\n')
            f.write('#set gamma_ref (keep zero to solve only magnetic presheath)\n')
            f.write(f'{self.gamma0}\n')
            f.write('#set nspec\n')
            f.write(f'{self.nspec}\n')
            f.write('#set nioverne\n')
            f.write(f'{self.nioverne}\n')
            f.write('#set TioverTe\n')
            f.write(f'{self.TioverTe}\n')
            f.write('#set mioverme\n')
            f.write(f'{self.mioverme}\n')
            f.write('#set set_current (flag)\n')
            f.write('0\n')
            f.write('#set target_current or phi_wall\n')
            f.write(f'{self.phi0}\n')

    def read_txt_files(self):
        with open('Fe_mpe_args.txt', 'r') as f:
            self.fe_mpe_args_text = f.read()
        with open('Fe_mpe.txt', 'r') as f:
            self.fe_mpe_text = f.read()
        with open('Fi_mpi_args.txt', 'r') as f:
            self.fi_mpi_args_text = f.read()
        with open('Fi_mpi.txt', 'r') as f:
            self.fi_mpi_text = f.read()
        with open('input_physparams.txt', 'r') as f:
            self.input_physparams_text = f.read()

    def append_h5file(self,hf,x0,y0,z0,tf):
        # Create a new group for each (x0, y0, z0) triplet
        if self.number_datasets:
            group_name = f'{self.nsample:06d}'
        else:
            group_name = f'x_{x0:.3f}_y_{y0:.3f}_z_{z0:.3f}_alpha_{self.alphadeg:.3f}_tf_{tf}'
        grp = hf.create_group(group_name)
        # Store text file contents as strings
        grp.create_dataset('Fe_mpe_args.txt', data=self.fe_mpe_args_text, dtype=h5py.string_dtype(encoding='utf-8'))
        grp.create_dataset('Fe_mpe.txt', data=self.fe_mpe_text, dtype=h5py.string_dtype(encoding='utf-8'))
        grp.create_dataset('Fi_mpi_args.txt', data=self.fi_mpi_args_text, dtype=h5py.string_dtype(encoding='utf-8'))
        grp.create_dataset('Fi_mpi.txt', data=self.fi_mpi_text, dtype=h5py.string_dtype(encoding='utf-8'))
        grp.create_dataset('input_physparams.txt', data=self.input_physparams_text, dtype=h5py.string_dtype(encoding='utf-8'))
        # add metadata attributes
        grp.attrs['x0'] = x0
        grp.attrs['y0'] = y0
        grp.attrs['z0'] = z0
        grp.attrs['alphadeg'] = self.alphadeg
        grp.attrs['tf'] = tf
        grp.attrs['B0'] = self.B0
        grp.attrs['phi0'] = self.phi0
        grp.attrs['ne0'] = self.ne0
        grp.attrs['ni0'] = self.ni0
        grp.attrs['Te0'] = self.Te0
        grp.attrs['Ti0'] = self.Ti0
        grp.attrs['gamma0'] = self.gamma0
        grp.attrs['nioverne'] = self.nioverne
        grp.attrs['TioverTe'] = self.TioverTe
        grp.attrs['mioverme'] = self.mioverme
        grp.attrs['mi'] = self.mi
        grp.attrs['me'] = self.me
        grp.attrs['e'] = self.e
        grp.attrs['simprefix'] = self.simulation.data_param.fileprefix

    def clean_txt_files(self):
        os.remove('Fe_mpe_args.txt')
        os.remove('Fe_mpe.txt')
        os.remove('Fi_mpi_args.txt')
        os.remove('Fi_mpi.txt')
        os.remove('input_physparams.txt')

    def generate(self,tf,xmin,xmax,Nxsample,Nysample,alphadeg,zplane='both',verbose=False):
        self.alphadeg = alphadeg
        self.load_frames(tf)
        self.get_grids()
        xindices, yindices, izplanes = self.get_ranges(xmin, xmax, Nxsample, Nysample, zplane)

        with h5py.File(self.outfilename, 'w') as hf:
            hf.attrs['description'] = 'Gyraze input data files from Gkeyll simulation'
            self.nsample = 0
            self.nskipped = 0
            # Sample points in the (x,y) plane
            for ix in xindices:
                for iy in yindices:
                    for izplane in izplanes:
                        if verbose: print(f'ix={ix}, x0={self.xgrid[ix]:.3f}, iy={iy}, y0={self.ygrid[iy]:.3f}, iz={izplane}, z0={self.zgrid[izplane]:.3f}')
                        x0 = self.xgrid[ix]
                        y0 = self.ygrid[iy]
                        z0 = self.zgrid[izplane]

                        self.eval_frames(x0, y0, z0)

                        if self.skip_point:
                            if verbose: print(f'Skipping point due to negativity in Ti, Te, ni, or ne')
                            self.skip_point = False
                            self.nskipped += 1
                            continue                    

                        self.write_F_mps_files(self.mue_norm, self.vpare_norm, self.fe0, 'elc')
                        self.write_F_mps_files(self.mui_norm, self.vpari_norm, self.fi0, 'ion')
                        self.write_input_physparams_file()
                            
                        # Read the contents of the text files
                        self.read_txt_files()

                        self.append_h5file(hf, x0, y0, z0, tf)

                        self.clean_txt_files()

                        self.nsample += 1
            hf.attrs['nsample'] = self.nsample
        print(f'Wrote {self.nsample} datasets to {self.outfilename}, skipped {self.nskipped} datasets due to negative moments.')