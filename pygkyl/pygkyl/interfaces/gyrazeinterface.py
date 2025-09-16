import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

from ..classes import Frame, Simulation

class GyrazeInterface:
    def __init__(self, simulation:Simulation):
        self.simulation = simulation
        self.frames = self.simulation.available_frames['ion']

    def load_frames(self, timeframe):
        Bmag = Frame(simulation=self.simulation,fieldname='Bmag',tf=timeframe,load=True)
        phi = Frame(simulation=self.simulation,fieldname='phi',tf=timeframe,load=True)
        ne = Frame(simulation=self.simulation,fieldname='ne',tf=timeframe,load=True)
        ni = Frame(simulation=self.simulation,fieldname='ni',tf=timeframe,load=True)
        Te = Frame(simulation=self.simulation,fieldname='Te',tf=timeframe,load=True)
        Ti = Frame(simulation=self.simulation,fieldname='Ti',tf=timeframe,load=True)
        fe = Frame(simulation=self.simulation,fieldname='fe',tf=timeframe,load=True)
        fi = Frame(simulation=self.simulation,fieldname='fi',tf=timeframe,load=True)
        gamma = Frame(simulation=self.simulation,fieldname='rhoe_lambdaD',tf=timeframe,load=True)
        return Bmag, phi, ne, ni, Te, Ti, fe, fi, gamma

    def eval_frames(self,frames, x, y, z):
        values = []
        for frame in frames:
            if frame.dimensionality == 3:
                values.append(frame.get_values([x, y, z]))
            elif frame.dimensionality == 5:
                values.append(frame.get_values([x, y, z, 'all', 'all']))
                
    def get_grids(self,distf_frame:Frame):
        xgrid = distf_frame.new_grids[0]
        ygrid = distf_frame.new_grids[1]
        zgrid = distf_frame.new_grids[2]
        vpargrid = distf_frame.new_grids[3]
        mugrid = distf_frame.new_grids[4]
        return xgrid, ygrid, zgrid, vpargrid, mugrid

    def get_ranges(self, xgrid, ygrid, zgrid, xmin, xmax, Nx, Ny, zplane):
        ixmin = np.argmin(np.abs(xgrid - xmin))
        ixmax = np.argmin(np.abs(xgrid - xmax))
        iymin = 0
        iymax = len(ygrid)-1
        if zplane=='upper':
            izplane = np.argmax(zgrid)
        elif zplane=='lower':
            izplane = np.argmin(zgrid)
        
        xindices = np.linspace(ixmin, ixmax, Nx, dtype=int)
        yindices = np.linspace(iymin, iymax, Ny, dtype=int)
        # remove duplicates
        xindices = np.unique(xindices)
        yindices = np.unique(yindices)
        return xindices, yindices, izplane


    def generate_gyraze_data(self):
        tf = self.frames_phsp[-1]

        xmin = 1.1
        xmax = 1.3
        Nxsample = 3 # number of x samples
        Nysample = 4 # number of y samples
        zplane = 'upper' # 'upper' or 'lower' side of the limiter
        alphadeg = 0.1 # angle of the magnetic field to the wall in degree
        nspec = 2

        # ----- END OF USER INPUTS -----

        # Load the frames
        Bmag, phi, ne, ni, Te, Ti, fe, fi, gamma = self.load_frames(tf)
        xgrid, ygrid, zgrid, vparegrid, muegrid = self.get_grids(fe)
        xgrid, ygrid, zgrid, vparigrid, muigrid = self.get_grids(fi)

        me = self.simulation.species['elc'].m
        mi = self.simulation.species['ion'].m
        mioverme = mi/me
        e = np.abs(self.simulation.species['elc'].q)

        xindices, yindices, izplane = self.get_ranges(xgrid, ygrid, zgrid, xmin, xmax, Nxsample, Nysample, zplane)
        # Sample points in the (x,y) plane
        for ix in xindices:
            for iy in yindices:
                print(f'ix={ix}, x0={xgrid[ix]:.3f}, iy={iy}, y0={ygrid[iy]:.3f}, iz={izplane}, z0={zgrid[izplane]:.3f}')
                x0 = xgrid[ix]
                y0 = ygrid[iy]
                z0 = zgrid[izplane]
                
                B0, phi0, ne0, ni0, Te0, Ti0, fe0, fi0, gamma0 = self.eval_frames([Bmag, phi, ne, ni, Te, Ti, fe, fi, gamma], x0, y0, z0)

                nioverne = ni0/ne0
                TioverTe = Ti0/Te0

                vpnorme = vparegrid / np.sqrt(Te0*e/me)
                munorme = muegrid / (Te0*e/B0)
                vpnormi = vparigrid / np.sqrt(Ti0*e/mi)
                munormi = muigrid / (Ti0*e/B0)

                ## Filter negativity in fe0 and fi0
                fe0[fe0<0] = 0.0
                fi0[fi0<0] = 0.0

                # Create Fe_mpe_args.txt file
                with open('Fe_mpe_args.txt', 'w') as f:
                    # Write xperp values in the first line
                    f.write(' '.join(map(str, munorme)) + '\n')
                    # Write spar values in the second line
                    f.write(' '.join(map(str, vpnorme)))
                # Write Fe_mpe.txt file with electron distribution function values
                np.savetxt('Fe_mpe.txt', fe0.squeeze().T)
                            
                # Same for the ions
                with open('Fi_mpi_args.txt', 'w') as f:
                    # Write xperp values in the first line
                    f.write(' '.join(map(str, munormi)) + '\n')
                    # Write spar values in the second line
                    f.write(' '.join(map(str, vpnormi)))  
                np.savetxt('Fi_mpi.txt', fi0.squeeze().T)

                # Create input_physparams.txt file with the physical parameters
                with open('input_physparams.txt', 'w') as f:
                    f.write('#set type_distfunc_entrance (= ADHOC or other string)\n')
                    f.write('GKEYLL\n')
                    f.write('#set alphadeg\n')
                    f.write(f'{alphadeg}\n')
                    f.write('#set gamma_ref (keep zero to solve only magnetic presheath)\n')
                    f.write(f'{gamma0}\n')
                    f.write('#set nspec\n')
                    f.write(f'{nspec}\n')
                    f.write('#set nioverne\n')
                    f.write(f'{nioverne}\n')
                    f.write('#set TioverTe\n')
                    f.write(f'{TioverTe}\n')
                    f.write('#set mioverme\n')
                    f.write(f'{mioverme}\n')
                    f.write('#set set_current (flag)\n')
                    f.write('0\n')
                    f.write('#set target_current or phi_wall\n')
                    f.write(f'{phi0}\n')
                    
                # Read the contents of the text files
                with open('Fe_mpe_args.txt', 'r') as f:
                    fe_mpe_args_text = f.read()
                with open('Fe_mpe.txt', 'r') as f:
                    fe_mpe_text = f.read()
                with open('Fi_mpi_args.txt', 'r') as f:
                    fi_mpi_args_text = f.read()
                with open('Fi_mpi.txt', 'r') as f:
                    fi_mpi_text = f.read()
                with open('input_physparams.txt', 'r') as f:
                    input_physparams_text = f.read()

                # Create an HDF5 file and store the data from the text files
                with h5py.File('data.h5', 'w') as hf:
                    # Store text file contents as strings
                    hf.create_dataset('Fe_mpe_args.txt', data=fe_mpe_args_text, dtype=h5py.string_dtype(encoding='utf-8'))
                    hf.create_dataset('Fe_mpe.txt', data=fe_mpe_text, dtype=h5py.string_dtype(encoding='utf-8'))
                    hf.create_dataset('Fi_mpi_args.txt', data=fi_mpi_args_text, dtype=h5py.string_dtype(encoding='utf-8'))
                    hf.create_dataset('Fi_mpi.txt', data=fi_mpi_text, dtype=h5py.string_dtype(encoding='utf-8'))
                    hf.create_dataset('input_physparams.txt', data=input_physparams_text, dtype=h5py.string_dtype(encoding='utf-8'))
                    
                # Remove the intermediate text files
                os.remove('Fe_mpe_args.txt')
                os.remove('Fe_mpe.txt')
                os.remove('Fi_mpi_args.txt')
                os.remove('Fi_mpi.txt')
                os.remove('input_physparams.txt')