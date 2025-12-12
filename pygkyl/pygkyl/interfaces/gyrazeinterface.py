import attr
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import time
from typing import List, Union, Optional
from ..classes import Frame, Simulation

def get_local_maxwellian(n0, upar0, T0, m, B0, vpar, mu):
    """Compute local Maxwellian distribution function values.
    
    Parameters:
    -----------
    n0 : float
        Density (m^-3)
    T0 : float
        Temperature (eV)
    m : float
        Particle mass (kg)
    B0 : float
        Magnetic field strength (T)
    vpar : ndarray
        Parallel velocity grid (m/s)
    mu : ndarray
        Magnetic moment grid (J/T)
        
    Returns:
    --------
    f0 : ndarray
        Local Maxwellian distribution function values on the (vpar, mu) grid.
    """
    eV_to_J = 1.602176634e-19  # Conversion factor from eV to J
    T0_J = T0 * eV_to_J  # Convert temperature to Joules
    vth = np.sqrt(2 * T0_J / m)
    VPAR, MU = np.meshgrid(vpar, mu, indexing='ij')
    
    prefactor = n0 / ((np.pi**1.5) * vth**3)
    exp_arg = -((VPAR-upar0)**2 + 2 * MU * B0 / m) / vth**2

    return prefactor * np.exp(exp_arg)

class GyrazeNumparams:
    '''Class to hold Gyraze numerical parameters.'''
    def __init__(self):
        self.max_it = 1000
        self.init_grid = 1.0
        self.sys_siz = 25.0
        self.gridsize_mp = 0.35
        self.gridsize_ds = 0.125
        self.maxmu = 8.0
        self.maxvpar = 7.0
        self.maxvpar_i = 5.0
        self.dmu = 0.2
        self.dvpar = 0.1
        self.dvpar_i = 0.2
        self.smallgamma = 0.1999
        self.tol_mp0 = 0.001
        self.tol_mp1 = 0.005
        self.tol_ds0 = 0.01
        self.tol_ds1 = 0.03
        self.tol_j = 0.005
        self.weight_mp = 0.3
        self.weight_ds = 1.0
        self.weight_j = 0.2
        self.margin_mp = 0.04
        self.margin_ds = 0.04
        self.zoom_mp = 3
        self.zoom_ds = 1

class GyrazeAttribute:
    '''Class to hold a single Gyraze attribute, providing methods to load and evaluate it.'''
    def __init__(self, fieldname, label, units, vmin=None, vmax=None, manual=False):
        self.fieldname = fieldname
        self.label = label
        self.units = units
        self.vmin = vmin
        self.vmax = vmax
        self.manual = manual
        self.frame = None
        self.v0 = None
        self.tf = None
        
        self.check_lims = self.check_lims_disabled
        self.check_lims_upper = self.check_lims_disabled
        self.check_lims_lower = self.check_lims_disabled
        
        if fieldname in ['fe', 'fi']:
            self.eval = self.eval5d
            self.filter_negativity = self.filter_negativity_active
        else:
            self.eval = self.eval3d
            self.filter_negativity = self.filter_negativity_disabled
            self.check_lims = self.check_lims_active
            
        if manual:
            self.filter_negativity = self.filter_negativity_disabled
            self.load = lambda simulation, tf: None
            self.eval = lambda x, y, z: None
        else:
            self.load = self.load_active
        
    def load_active(self, simulation, tf):
        self.tf = tf
        self.frame = Frame(simulation=simulation,fieldname=self.fieldname,tf=tf,load=True)
           
    def eval3d(self, x, y, z):
        self.v0 = self.frame.get_value([x, y, z])

    def eval5d(self, x, y, z):
        self.v0 = self.frame.get_value([x, y, z, 'all', 'all'])
        
    def set_vlim(self, value, which='min'):
        self.check_lims = self.check_lims_active
        if which=='max':
            self.vmax = value
            self.check_lims_upper = self.check_lims_upper_active
        elif which=='min':
            self.vmin = value
            self.check_lims_lower = self.check_lims_lower_active
        
    def check_negativity(self):
        if self.fieldname in ['Te', 'Ti', 'ne', 'ni']:
            if np.any(self.v0 < 0):
                return True
        return False
    
    def check_lims_disabled(self):
        return False
    
    def check_lims_active(self):
        if self.check_lims_upper() or self.check_lims_lower():
            return True
        return False
    
    def check_lims_upper_active(self):
        if np.any(self.v0 > self.vmax):
            return True
        return False

    def check_lims_lower_active(self):
        if np.any(self.v0 < self.vmin):
            return True
        return False

    def filter_negativity_active(self):
        self.v0[self.v0 < 0] = 0.0
        
    def filter_negativity_disabled(self):
        pass

class GyrazeDataset:
    '''Class to hold all required Gyraze data and metadata about a single simulation timestep.'''
    def __init__(self, me = 9.10938356e-31, mi = 1.6726219e-27, no_distf=False):
        self.me = me
        self.mi = mi
        self.no_distf = no_distf
        self.attributes = {}
        self.attributes['B'] = GyrazeAttribute('Bmag', r'$B$', 'T')
        self.attributes['phi'] = GyrazeAttribute('phi', r'$\phi$', 'V')
        self.attributes['ne'] = GyrazeAttribute('ne', r'$n_e$', 'm$^{-3}$')
        self.attributes['ni'] = GyrazeAttribute('ni', r'$n_i$', 'm$^{-3}$')
        self.attributes['Te'] = GyrazeAttribute('Te', r'$T_e$', 'eV')
        self.attributes['Ti'] = GyrazeAttribute('Ti', r'$T_i$', 'eV')
        self.attributes['gamma'] = GyrazeAttribute('rhoe_lambdaD', r'$\rho_e/\lambda_D$', '')
        self.attributes['phi_norm'] = GyrazeAttribute('phi_norm', r'$e\phi/T_e$', '', manual=True)
        self.attributes['nioverne'] = GyrazeAttribute('nioverne', r'$n_i/n_e$', '', manual=True)
        self.attributes['TioverTe'] = GyrazeAttribute('TioverTe', r'$T_i/T_e$', '', manual=True)
        self.attributes['Epare_norm'] = GyrazeAttribute('Epare_norm', r'$1/2 m_e v_\parallel^2/T_{e0}$', '', manual=True)
        self.attributes['Eperpe_norm'] = GyrazeAttribute('Eperpe_norm', r'$\mu B_0/T_{e0}$', '', manual=True)
        self.attributes['Epari_norm'] = GyrazeAttribute('Epari_norm', r'$1/2 m_i v_\parallel^2/T_{i0}$', '', manual=True)
        self.attributes['Eperpi_norm'] = GyrazeAttribute('Eperpi_norm', r'$\mu B_0/T_{i0}$', '', manual=True)
        self.grids = {}
        if not no_distf:
            self.attributes['fe'] = GyrazeAttribute('fe', r'$f_e$', 'm$^{-6}$s$^3$')
            self.attributes['fi'] = GyrazeAttribute('fi', r'$f_i$', 'm$^{-6}$s$^3$')
            self.attributes['Fe'] = GyrazeAttribute('Fe', r'$F_{e0}$', 'm$^{-6}$s$^3$', manual=True)
            self.attributes['Fi'] = GyrazeAttribute('Fi', r'$F_{i0}$', 'm$^{-6}$s$^3$', manual=True)
            self.attributes['fi_norm'] = GyrazeAttribute('fi_norm', r'$f_i/F_{i0}$', '', manual=True)
            self.attributes['fe_norm'] = GyrazeAttribute('fe_norm', r'$f_e/F_{e0}$', '', manual=True)
            self.eval_local_maxwellians = self.eval_local_maxwellians_on
        else:
            self.eval_local_maxwellians = self.eval_local_maxwellians_off
        self.Fe0 = None
        self.Fi0 = None
        self.Tref = None
        self.tf = None
        self.t0 = None

    def load(self, simulation, tf):
        self.tf = tf
        # Load all attributes
        for attr in self.attributes.values():
            attr.load(simulation, tf)
        # Extract time and grids
        self.t0 = self.attributes['B'].frame.time
        self.grids['x'] = self.attributes['B'].frame.new_grids[0]
        self.grids['y'] = self.attributes['B'].frame.new_grids[1]
        self.grids['z'] = self.attributes['B'].frame.new_grids[2]
        self.grids['vpare'] = self.attributes['fe'].frame.new_grids[3]
        self.grids['mue'] = self.attributes['fe'].frame.new_grids[4]
        self.grids['vpari'] = self.attributes['fi'].frame.new_grids[3]
        self.grids['mui'] = self.attributes['fi'].frame.new_grids[4]
        
    def eval_local_maxwellians_on(self,species):
        s = species[0]
        m = self.attributes['f'+s].frame.simulation.species[species].m
        n0 = self.attributes['n'+s].v0
        upar0 = 0.0 # TODO add flow velocity support
        T0 = self.attributes['T'+s].v0
        B0 = self.attributes['B'].v0
        return get_local_maxwellian(n0, upar0, T0, m, B0, self.grids['vpar'+s], self.grids['mu'+s])

    def eval_local_maxwellians_off(self, species):
        return 0
    
    def eval(self, x, y, z):
        for attr in self.attributes.values():
            attr.eval(x, y, z)
        
        self.Tref = self.attributes['Te'].v0
        self.nref = self.attributes['ne'].v0
        self.attributes['phi_norm'].v0 = self.attributes['phi'].v0 / self.Tref
        self.attributes['nioverne'].v0 = self.attributes['ni'].v0 / self.nref
        self.attributes['TioverTe'].v0 = self.attributes['Ti'].v0 / self.Tref
        self.attributes['Epare_norm'].v0 = 0.5*self.me*(self.grids['vpare'])**2 / (self.Tref * 1.602e-19)
        self.attributes['Epari_norm'].v0 = 0.5*self.mi*(self.grids['vpari'])**2 / (self.Tref * 1.602e-19)
        self.attributes['Eperpe_norm'].v0 = self.grids['mue'] * self.attributes['B'].v0 / (self.Tref * 1.602e-19)
        self.attributes['Eperpi_norm'].v0 = self.grids['mui'] * self.attributes['B'].v0 / (self.Tref * 1.602e-19)
        self.attributes['Fe'].v0 = self.eval_local_maxwellians('elc')
        self.attributes['Fi'].v0 = self.eval_local_maxwellians('ion')
        self.attributes['fe_norm'].v0 = self.attributes['fe'].v0 #/ self.attributes['Fe'].v0
        self.attributes['fi_norm'].v0 = self.attributes['fi'].v0 #/ self.attributes['Fi'].v0

    def filter(self):
        for attr in self.attributes.values():
            if attr.check_negativity():
                return True
            if attr.check_lims():
                return True
        return False
    
    def filter_negativity(self):
        for attr in self.attributes.values():
            attr.filter_negativity()

class GyrazeInterface:
    def __init__(self, simulation:Simulation, **kwargs):
        self.simulation = simulation
        self.alphadeg : float = kwargs.get('alphadeg', 5.0)
        self.filter_negativity : bool = kwargs.get('filter_negativity', False)
        self.no_distf : bool = kwargs.get('no_distf', False)
        self.number_datasets : bool = kwargs.get('number_datasets', False)
        self.outfilename : str = kwargs.get('outfilename', 'data.h5')

        self.frames = self.simulation.available_frames['ion']
        self.nspec = len(self.simulation.species)
        self.me = self.simulation.species['elc'].m
        self.mi = self.simulation.species['ion'].m
        self.mioverme = self.mi/self.me
        self.e = np.abs(self.simulation.species['elc'].q)
        self.dataset = GyrazeDataset(self.me, self.mi, self.no_distf)

        self.fe_mpe_args_text = None
        self.fe_mpe_text = None
        self.fi_mpe_args_text = None
        self.fi_mpe_text = None
        self.input_physparams_text = None
        self.input_numparams_text = None
        self.skip_point = False
        self.nsample = None
        self.nskipped = None
        self.required_attrs = [
            'x0', 'y0', 'z0', 'alphadeg', 'tf', 't0', 'B0', 'phi0',
            'ne0', 'ni0', 'Te0', 'Ti0', 'gamma0', 'nioverne',
            'TioverTe', 'mioverme', 'mi', 'me', 'e', 'simprefix'
        ]
        self.input_file_names = [
            'Fe_mpe_args.txt', 'Fe_mpe.txt', 
            'Fi_mpe_args.txt', 'Fi_mpe.txt', 
            'input_physparams.txt', 'input_numparams.txt'
        ]
        
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None
        self.NxSOL = None
        self.NySOL = None
        self.time_frames = None
        self.setup_default_ranges()
        
        self.numparams = GyrazeNumparams()

    def setup_default_ranges(self):
        x = self.simulation.geom_param.x
        dx = x[1] - x[0]
        self.xmin = self.simulation.geom_param.x_LCFS + dx
        self.xmax = self.simulation.geom_param.Lx - dx
        self.ymin = self.simulation.geom_param.y.min()
        self.ymax = self.simulation.geom_param.y.max()
        self.zmin = self.simulation.geom_param.z.min()
        self.zmax = self.simulation.geom_param.z.max()
        self.NxSOL = len(x[(x>self.xmin) & (x<self.xmax)])
        self.NySOL = len(self.simulation.geom_param.y)
        self.time_frames = self.simulation.available_frames['ion']
        
    def load_frames(self, timeframe):
        self.dataset.load(self.simulation, timeframe)
        
    def update_numparams(self):
        ## TODO: check if we need to provide smart numerical params something like:
        # self.numparams.maxmu = self.dataset.attributes['Eperpe_norm'].v0.max()
        # self.numparams.maxvpar = self.dataset.attributes['Epare_norm'].v0.max()
        # self.numparams.maxvpar_i = self.dataset.attributes['Epari_norm'].v0.max()
        # self.numparams.dmu = np.abs(self.dataset.attributes['Eperpe_norm'].v0[1] - self.dataset.attributes['Eperpe_norm'].v0[0])
        # self.numparams.dvpar = np.abs(self.dataset.attributes['Epare_norm'].v0[1] - self.dataset.attributes['Epare_norm'].v0[0])
        # self.numparams.dvpar_i = np.abs(self.dataset.attributes['Epari_norm'].v0[1] - self.dataset.attributes['Epari_norm'].v0[0])
        # self.numparams.smallgamma = self.dataset.attributes['gamma'].v0 # TODO not convinced here
        pass

    def eval_frames(self, x, y, z):
        self.dataset.eval(x, y, z)

        if self.dataset.filter():
            self.skip_point = True
            return
        
        if self.filter_negativity:
            self.dataset.filter_negativity()
            
        self.update_numparams()

    def get_ranges(self, xmin, xmax, Nx, Ny, zplane):
        ixmin = np.argmin(np.abs(self.dataset.grids['x'] - xmin))
        ixmax = np.argmin(np.abs(self.dataset.grids['x'] - xmax))
        iymin = 0
        iymax = len(self.dataset.grids['y'])-1
        if zplane=='upper':
            izplanes = [np.argmax(self.dataset.grids['z'])]
        elif zplane=='lower':
            izplanes = [np.argmin(self.dataset.grids['z'])]
        elif zplane=='both':
            izplanes = [np.argmin(self.dataset.grids['z']), np.argmax(self.dataset.grids['z'])]

        xindices = np.linspace(ixmin, ixmax, Nx, dtype=int)
        yindices = np.linspace(iymin, iymax, Ny, dtype=int)
        # remove duplicates
        xindices = np.unique(xindices)
        yindices = np.unique(yindices)
        return xindices, yindices, izplanes

    def generate_F_mps_content(self, Eperpnorm, Eparnorm, f0):
        # Find the index of positive vpar
        ipos = np.argmin(np.abs(self.dataset.grids['vpare'] - 0.0))
        
        # select Eparnorm and f0 for positive vpar only
        Eparnorm = Eparnorm[ipos:]
        f0 = f0[ipos:,:]
        
        # Generate args content
        args_content = ' '.join(map(str, Eperpnorm)) + '\n' + ' '.join(map(str, Eparnorm))

        # Generate f0 content using StringIO to mimic savetxt behavior
        f0_buffer = io.StringIO()
        np.savetxt(f0_buffer, f0.squeeze().T, fmt='%.16e')
        f0_content = f0_buffer.getvalue().strip()  # Remove any trailing whitespace
        f0_buffer.close()
        return args_content, f0_content

    def generate_input_physparams_content(self):
        content = (
            '#set type_distfunc_entrance (= ADHOC or other string)\n'
            'GKEYLL data v0.1\n'
            '#set alphadeg\n'
            f'{self.alphadeg}\n'
            '#set gammaflag\n'
            '1\n'
            '#set gamma_ref\n'
            f"{self.dataset.attributes['gamma'].v0}\n"
            '#set nspec\n'
            f'{self.nspec-1}\n'
            '#set nioverne\n'
            f"{self.dataset.attributes['nioverne'].v0}\n"
            '#set TioverTe\n'
            f"{self.dataset.attributes['TioverTe'].v0}\n"
            '#set mioverme\n'
            f'{self.mioverme}\n'
            '#set set_current (flag)\n'
            '0\n'
            '#set target_current or phi_wall\n'
            f"{self.dataset.attributes['phi_norm'].v0}\n"
        )
        return content
    
    def generate_input_numparams_content(self):
        content = (
            '# Automatically generated Gyraze numerical parameters file using Gkeyll data\n'
            '#set MAX_IT\n'
            f'{self.numparams.max_it}\n'
            '#set INITIAL_GRID_PARAMETER SYS_SIZ GRIDSIZE_MP, GRIDSIZE_DS\n'
            f'{self.numparams.init_grid} {self.numparams.sys_siz} {self.numparams.gridsize_mp} {self.numparams.gridsize_ds}\n'
            '#set MAXMU MAXVPAR MAXVPAR_I DMU DVPAR DVPAR_I /// need to update this\n'
            f'{self.numparams.maxmu} {self.numparams.maxvpar} {self.numparams.maxvpar_i} {self.numparams.dmu} {self.numparams.dvpar} {self.numparams.dvpar_i}\n'
            '#set SMALLGAMMA\n'
            f'{self.numparams.smallgamma}\n'
            '#set tol_MP[0] tol_MP[1] tol_DS[0] tol_DS[1] tol_j\n'
            f'{self.numparams.tol_mp0} {self.numparams.tol_mp1} {self.numparams.tol_ds0} {self.numparams.tol_ds1} {self.numparams.tol_j}\n'
            '#set WEIGHT_MP WEIGHT_DS WEIGHT_j\n'
            f'{self.numparams.weight_mp} {self.numparams.weight_ds} {self.numparams.weight_j}\n'
            '#set MARGIN_MP MARGIN_DS\n'
            f'{self.numparams.margin_mp} {self.numparams.margin_ds}\n'
            '#set ZOOM_MP ZOOM_DS\n'
            f'{self.numparams.zoom_mp} {self.numparams.zoom_ds}\n'
        )
        return content
    
    def generate_input_files(self):
        if self.no_distf:
            self.fe_mpe_args_text = ''
            self.fe_mpe_text = ''
            self.fi_mpe_args_text = ''
            self.fi_mpe_text = ''
        else:
            self.fe_mpe_args_text, self.fe_mpe_text = self.generate_F_mps_content(
                self.dataset.attributes['Eperpe_norm'].v0,
                self.dataset.attributes['Epare_norm'].v0, 
                self.dataset.attributes['fe'].v0)
            self.fi_mpe_args_text, self.fi_mpe_text = self.generate_F_mps_content(
                self.dataset.attributes['Eperpi_norm'].v0, 
                self.dataset.attributes['Epari_norm'].v0, 
                self.dataset.attributes['fi'].v0)
        self.input_physparams_text = self.generate_input_physparams_content()
        self.input_numparams_text = self.generate_input_numparams_content()

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
        grp.create_dataset('Fi_mpe_args.txt', data=self.fi_mpe_args_text, dtype=h5py.string_dtype(encoding='utf-8'))
        grp.create_dataset('Fi_mpe.txt', data=self.fi_mpe_text, dtype=h5py.string_dtype(encoding='utf-8'))
        grp.create_dataset('input_physparams.txt', data=self.input_physparams_text, dtype=h5py.string_dtype(encoding='utf-8'))
        grp.create_dataset('input_numparams.txt', data=self.input_numparams_text, dtype=h5py.string_dtype(encoding='utf-8'))
        # add metadata attributes
        grp.attrs['x0'] = x0
        grp.attrs['y0'] = y0
        grp.attrs['z0'] = z0
        grp.attrs['alphadeg'] = self.alphadeg
        grp.attrs['tf'] = tf
        grp.attrs['t0'] = self.dataset.t0
        grp.attrs['mi'] = self.mi
        grp.attrs['me'] = self.me
        grp.attrs['e'] = self.e
        grp.attrs['mioverme'] = self.mioverme
        grp.attrs['simprefix'] = self.simulation.data_param.fileprefix
        for attr_name, attr in self.dataset.attributes.items():
            if np.isscalar(attr.v0):
                ext0 = '' if attr.fieldname in ['TioverTe', 'mioverme', 'nioverne'] else '0'
                grp.attrs[attr_name + ext0] = attr.v0

    def generate(self, time_frames: Optional[Union[List[int], int]] = None, 
                 xmin: Optional[float] = None, 
                 xmax: Optional[float] = None, 
                 Nxsample: Optional[int] = None, 
                 Nysample: Optional[int] = None, 
                 alphadeg: Optional[float] = None, 
                 zplane: Optional[str] = None, 
                 filter_negativity: Optional[bool] = None,
                 no_distf: Optional[bool] = None,
                 lim_dict: Optional[dict] = None,
                 verbose: bool = False,):
        '''
        Main interface function to generate Gyraze input data files from Gkeyll simulation data.
        Parameters:
        -----------
        time_frames : list of int or int, optional
            List of time frames to process or single int. If None, use all available frames.
        xmin : float, optional
            Minimum x value for sampling. If None, use x_LCFS + dx.
        xmax : float, optional
            Maximum x value for sampling. If None, use Lx - dx.
        Nxsample : int, optional
            Number of x points to sample. If None, use all SOL points.
        Nysample : int, optional
            Number of y points to sample. If None, use all points.
        alphadeg : float, optional
            Angle in degrees between magnetic field and wall normal. If None, use 0.3.
        zplane : str, optional
            Which z-plane to sample. If None, both upper and lower side of the limiter. Options: 'upper', 'lower', 'both'.
        filter_negativity : bool, optional
            Filter negative values in the distribution functions and put them to zero.
            If provided, override the instance's filter_negativity setting.
        no_distf : bool, optional
            If True, do not include distribution function data in the output.
        lim_dict : dict, optional
            Dictionary specifying limits for filtering points.
            E.g., provided as {'phi': {'min': float, 'max': float}}, skip points where phi0 is outside this range.
        verbose : bool
            If True, print detailed information during processing.
        '''
        if isinstance(time_frames, int): time_frames = [time_frames]
        time_frames = time_frames if time_frames is not None else self.time_frames
        xmin = xmin if xmin is not None else self.xmin
        xmax = xmax if xmax is not None else self.xmax
        Nxsample = Nxsample if Nxsample is not None else self.NxSOL
        Nysample = Nysample if Nysample is not None else self.NySOL
        self.alphadeg = alphadeg if alphadeg is not None else self.alphadeg
        zplane = zplane if zplane is not None else 'both'
        
        if filter_negativity is not None:
            self.filter_negativity = filter_negativity
            
        if no_distf is not None:
            self.no_distf = no_distf
            
        if lim_dict is not None:
            for key in lim_dict.keys():
                for which, value in lim_dict[key].items():
                    self.dataset.attributes[key].set_vlim(value, which)

        if verbose: 
            print(f'Generating Gyraze input data for alphadeg={self.alphadeg}, time frames={time_frames}, x=[{xmin},{xmax}], Nx={Nxsample}, Ny={Nysample}, zplane={zplane}')
            expected_num = len(time_frames) * Nxsample * Nysample * (2 if zplane=='both' else 1)
            print(f'Expected number of datasets (before skipping negatives): {expected_num}')
            
        self.nsample = 0
        self.nskipped = 0
        
        with h5py.File(self.outfilename, 'w') as hf:
            hf.attrs['description'] = 'Gyraze input data files from Gkeyll simulation. \n\
                The txt files are in normalized units. The attributes are in SI units except for temperatures which are in eV'
            for tf in time_frames:
                self.load_frames(tf)
                xindices, yindices, izplanes = self.get_ranges(xmin, xmax, Nxsample, Nysample, zplane)
                t0 = self.dataset.t0
                for ix in xindices:
                    x0 = self.dataset.grids['x'][ix]
                    for iy in yindices:
                        y0 = self.dataset.grids['y'][iy]
                        for izplane in izplanes:
                            z0 = self.dataset.grids['z'][izplane]
                            
                            if verbose: print(f't={t0}, x0={x0:.3f}, y0={y0:.3f}, z0={z0:.3f}')

                            self.eval_frames(x0, y0, z0)

                            if self.skip_point:
                                if verbose: print(f'Skipping point due to negativity in Ti, Te, ni, or ne')
                                self.skip_point = False
                                self.nskipped += 1
                            else:
                                self.generate_input_files()
                                self.append_h5file(hf, x0, y0, z0, tf)
                                self.nsample += 1
                            
            hf.attrs['nsample'] = self.nsample
        print(f'Wrote {self.nsample} datasets to {self.outfilename}, skipped {self.nskipped} datasets due to negative moments.')


    def extract_dataset_as_files(self, group_name, output_dir=None):
        """
        Extract text files from a specific dataset group to disk for external use.
        
        Parameters:
        -----------
        group_name : str
            Name of the group to extract
        output_dir : str
            Directory to write the extracted files
        """
        if not os.path.exists(self.outfilename):
            print(f"ERROR: Output file {self.outfilename} not found")
            
        try:
            with h5py.File(self.outfilename, 'r') as hf:
                if group_name not in hf:
                    print(f"ERROR: Group {group_name} not found in file")
                
                grp = hf[group_name]
                
                if output_dir is None:
                    output_dir = f'gyraze_data_{group_name}'
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Extract each text file
                for filename in self.input_file_names:
                    if filename in grp:
                        content = grp[filename][()].decode('utf-8')
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, 'w') as f:
                            f.write(content)
                        print(f"Extracted {filepath} in ${output_dir}")
                    else:
                        print(f"WARNING: {filename} not found in group {group_name}")
                
                # Also write a metadata file
                metadata_file = os.path.join(output_dir, 'metadata.txt')
                with open(metadata_file, 'w') as f:
                    f.write(f"Dataset: {group_name}\n")
                    for attr_name in grp.attrs:
                        f.write(f"{attr_name}: {grp.attrs[attr_name]}\n")
                print(f"Extracted {metadata_file} in ${output_dir}")
                                
        except Exception as e:
            print(f"ERROR: Failed to extract dataset: {e}")

    def verify_h5_data(self, verbose=False, Nsamp=1):
        """
        Verify the written HDF5 file by checking data integrity and content.
        
        Parameters:
        -----------
        verbose : bool
            If True, print detailed information about each dataset
        Nsamp : int, optional
            If provided, randomly sample Nsamp datasets and compare with simulation data
            
        Returns:
        --------
        bool
            True if verification passes, False otherwise
        """
        if not os.path.exists(self.outfilename):
            print(f"ERROR: Output file {self.outfilename} not found")
            return False
            
        try:
            with h5py.File(self.outfilename, 'r') as hf:
                # Check file-level attributes
                expected_attrs = ['description', 'nsample']
                for attr in expected_attrs:
                    if attr not in hf.attrs:
                        print(f"ERROR: Missing file attribute: {attr}")
                        return False
                
                nsample_file = hf.attrs['nsample']
                if verbose:
                    print(f"File contains {nsample_file} datasets")
                    print(f"File description: {hf.attrs['description']}")
                
                # Get all group names for random sampling
                all_group_names = list(hf.keys())
                
                # Determine which groups to check
                if Nsamp is not None and Nsamp > 0:
                    if Nsamp > len(all_group_names):
                        print(f"WARNING: Requested {Nsamp} samples but only {len(all_group_names)} available")
                        Nsamp = len(all_group_names)
                    
                    # Randomly sample groups
                    rng = np.random.RandomState(int(time.time() % 1 * 1e6))  # Local RNG seeded with current time
                    selected_groups = rng.choice(all_group_names, size=Nsamp, replace=False)
                    print(f"Randomly selected {Nsamp} datasets for detailed verification")
                else:
                    selected_groups = all_group_names
                
                # Check each group/dataset
                group_count = 0
                comparison_passed = True
                
                for group_name in all_group_names:
                    group_count += 1
                    grp = hf[group_name]
                    
                    # Basic verification for all groups
                    if verbose and group_name in selected_groups:
                        print(f"\nVerifying group: {group_name}")
                    
                    # Check required datasets
                    for dataset_name in self.input_file_names:
                        if dataset_name not in grp:
                            print(f"ERROR: Missing dataset {dataset_name} in group {group_name}")
                            return False
                        
                        # Check if dataset is readable as string
                        try:
                            content = grp[dataset_name][()].decode('utf-8')
                            if len(content) == 0:
                                print(f"ERROR: Empty content in {dataset_name} of group {group_name}")
                                return False
                        except Exception as e:
                            print(f"ERROR: Cannot read {dataset_name} in group {group_name}: {e}")
                            return False
                    
                    # Check required attributes
                    for attr in self.required_attrs:
                        if attr not in grp.attrs:
                            print(f"ERROR: Missing attribute {attr} in group {group_name}")
                            return False
                    
                    # Check for physical consistency
                    if grp.attrs['Te0'] <= 0 or grp.attrs['Ti0'] <= 0:
                        print(f"ERROR: Non-positive temperatures in group {group_name}")
                        return False
                    
                    if grp.attrs['ne0'] <= 0 or grp.attrs['ni0'] <= 0:
                        print(f"ERROR: Non-positive densities in group {group_name}")
                        return False
                    
                    # Detailed comparison with simulation data for selected groups
                    if group_name in selected_groups and Nsamp is not None:
                        if not self._compare_with_simulation(grp, verbose):
                            comparison_passed = False
                    
                    if verbose and group_name in selected_groups:
                        print(f"  Position: ({grp.attrs['x0']:.3f}, {grp.attrs['y0']:.3f}, {grp.attrs['z0']:.3f})")
                        print(f"  Te0={grp.attrs['Te0']:.3e}, Ti0={grp.attrs['Ti0']:.3e}")
                        print(f"  ne0={grp.attrs['ne0']:.3e}, ni0={grp.attrs['ni0']:.3e}")
                        print(f"  B0={grp.attrs['B0']:.3e}, phi0={grp.attrs['phi0']:.3e}")
                
                # Verify group count matches nsample
                if group_count != nsample_file:
                    print(f"ERROR: Group count ({group_count}) doesn't match nsample attribute ({nsample_file})")
                    return False
                
                if Nsamp is not None and not comparison_passed:
                    print(f"ERROR: Simulation data comparison failed for some datasets")
                    return False
                
                print(f"SUCCESS: Verification passed for {group_count} datasets")
                if Nsamp is not None:
                    print(f"         Detailed comparison with simulation data passed for {len(selected_groups)} samples")
                return True
                
        except Exception as e:
            print(f"ERROR: Failed to verify file: {e}")
            return False

    def _compare_with_simulation(self, grp, verbose=False):
        """
        Compare HDF5 group data with simulation data at the same point.
        
        Parameters:
        -----------
        grp : h5py.Group
            HDF5 group containing the dataset
        verbose : bool
            If True, print comparison details
            
        Returns:
        --------
        bool
            True if comparison passes, False otherwise
        """
        try:
            # Extract position and timeframe from group attributes
            x0 = grp.attrs['x0']
            y0 = grp.attrs['y0'] 
            z0 = grp.attrs['z0']
            tf = grp.attrs['tf']
            
            if verbose:
                print(f"    Comparing with simulation at ({x0:.3f}, {y0:.3f}, {z0:.3f}), tf={tf}")
            
            # Load frames at the same timeframe (only if not already loaded or different timeframe)
            if (self.dataset.attributes['fe'].v0 is None or self.dataset.attributes['fe'].tf != tf):
                self.load_frames(tf)
            
            # Evaluate frames at the same position
            self.eval_frames(x0, y0, z0)
            
            # Skip if point was marked to skip (negative values)
            if self.skip_point:
                if verbose:
                    print(f"    Skipping comparison due to negative values in simulation")
                self.skip_point = False
                return True
            
            # Compare scalar values with tolerance
            tolerance = 1e-10
            tocompare = ['B0', 'phi0', 'ne0', 'ni0', 'Te0', 'Ti0', 'gamma0', 'nioverne', 'TioverTe']
            comparisons = [(name, self.dataset.attributes[name.replace('0', '')].v0, grp.attrs[name]) for name in tocompare]
            
            for field_name, sim_value, h5_value in comparisons:
                rel_error = abs(sim_value - h5_value) / (abs(sim_value) + 1e-15)
                if rel_error > tolerance:
                    print(f"    ERROR: {field_name} mismatch - sim: {sim_value:.6e}, h5: {h5_value:.6e}, rel_error: {rel_error:.6e}")
                    return False
                elif verbose:
                    print(f"    {field_name}: sim={sim_value:.6e}, h5={h5_value:.6e} ✓")
            
            # Compare distribution function data by reconstructing from text
            self.generate_input_files()
            
            # Compare Fe data
            fe_args_h5 = grp['Fe_mpe_args.txt'][()].decode('utf-8')
            fe_data_h5 = grp['Fe_mpe.txt'][()].decode('utf-8')
            
            if fe_args_h5.strip() != self.fe_mpe_args_text.strip():
                print(f"    ERROR: Fe_mpe_args.txt content mismatch")
                return False
            
            # Parse and compare Fe distribution data
            fe_h5_lines = [line for line in fe_data_h5.split('\n') if line.strip()]
            fe_sim_lines = [line for line in self.fe_mpe_text.split('\n') if line.strip()]
            
            if len(fe_h5_lines) != len(fe_sim_lines):
                print(f"    ERROR: Fe_mpe.txt line count mismatch - h5: {len(fe_h5_lines)}, sim: {len(fe_sim_lines)}")
                return False
            
            # Sample a few lines for comparison (full comparison would be too slow)
            sample_indices = np.linspace(0, len(fe_h5_lines)-1, min(5, len(fe_h5_lines)), dtype=int)
            for idx in sample_indices:
                h5_values = np.array(fe_h5_lines[idx].split(), dtype=float)
                sim_values = np.array(fe_sim_lines[idx].split(), dtype=float)
                
                if len(h5_values) != len(sim_values):
                    print(f"    ERROR: Fe_mpe.txt line {idx} value count mismatch")
                    return False
                
                max_rel_error = np.max(np.abs(h5_values - sim_values) / (np.abs(sim_values) + 1e-15))
                if max_rel_error > tolerance:
                    print(f"    ERROR: Fe_mpe.txt line {idx} values mismatch, max rel_error: {max_rel_error:.6e}")
                    return False
            
            # Similar check for Fi data
            fi_args_h5 = grp['Fi_mpe_args.txt'][()].decode('utf-8')
            if fi_args_h5.strip() != self.fi_mpe_args_text.strip():
                print(f"    ERROR: Fi_mpe_args.txt content mismatch")
                return False
            
            if verbose:
                print(f"    Distribution function data comparison passed ✓")
            
            return True
            
        except Exception as e:
            print(f"    ERROR: Simulation comparison failed: {e}")
            return False

    def collect_attribute(self, attr_name):
        """
        Collect a specific attribute from all datasets into a numpy array.
        
        Parameters:
        -----------
        attr_name : str
            Name of the attribute to collect
            
        Returns:
        --------
        np.ndarray
            Array of attribute values from all datasets
        """
        if not os.path.exists(self.outfilename):
            print(f"ERROR: Output file {self.outfilename} not found")
            return None
            
        values = []
        
        try:
            with h5py.File(self.outfilename, 'r') as hf:
                for group_name in hf.keys():
                    grp = hf[group_name]
                    if attr_name in grp.attrs:
                        values.append(grp.attrs[attr_name])
                    else:
                        print(f"WARNING: Attribute {attr_name} not found in group {group_name}")
                        values.append(np.nan)  # Use NaN for missing attributes
            
            return np.array(values)
            
        except Exception as e:
            print(f"ERROR: Failed to collect attribute: {e}")
            return None
        
    def info(self):
        """
        Print summary information about the generated HDF5 file.
        """
        if not os.path.exists(self.outfilename):
            print(f"ERROR: Output file {self.outfilename} not found")
            return
            
        try:
            with h5py.File(self.outfilename, 'r') as hf:
                print(f"File: {self.outfilename}")
                print(f"Description: {hf.attrs.get('description', 'N/A')}")
                nsample = hf.attrs.get('nsample', 0)
                print(f"Number of datasets: {nsample}")
                
                if nsample > 0:
                    first_group = list(hf.keys())[0]
                    grp = hf[first_group]
                    print(f"\nAttributes in first dataset ({first_group}):")
                    for attr in grp.attrs:
                        print(f"  {attr}: {grp.attrs[attr]}")
                    
                    print(f"\nDatasets in first dataset ({first_group}):")
                    for dset in grp:
                        print(f"  {dset}: shape {grp[dset].shape}, dtype {grp[dset].dtype}")
                        
        except Exception as e:
            print(f"ERROR: Failed to read file info: {e}")
    
    def load_h5_data(self, filename=None):
        """
        Load an existing HDF5 file for analysis.
        
        Parameters:
        -----------
        filename : str, optional
            Path to HDF5 file. If None, uses self.outfilename
        """
        if filename is not None:
            self.outfilename = filename
            
        if not os.path.exists(self.outfilename):
            print(f"ERROR: Output file {self.outfilename} not found")
            
        try:
            with h5py.File(self.outfilename, 'r') as hf:
                print(f"Loaded HDF5 file: {self.outfilename}")
                print(f"Description: {hf.attrs.get('description', 'N/A')}")
                nsample = hf.attrs.get('nsample', 0)
                print(f"Number of datasets: {nsample}")
                
        except Exception as e:
            print(f"ERROR: Failed to load HDF5 file: {e}")

    def plot_attribute_histograms(self, attributes: Optional[List[str]] = None, 
                                 bins: int = 25, figsize: tuple = (10, 10), 
                                 save_fig: Optional[str] = None, get_data_only = False):
        """
        Plot histograms of specified attributes from all datasets in the HDF5 file.
        
        Parameters:
        -----------
        attributes : list of str, optional
            List of attribute names to plot. If None, uses default list.
        bins : int
            Number of histogram bins
        figsize : tuple
            Figure size (width, height)
        save_fig : str, optional
            If provided, save figure to this filename
        get_data_only : bool, optional
            If True, only collect and return the data without plotting
        """
        if attributes is None:
            attributes = ['gamma0','nioverne','TioverTe','phi_norm0']
        
        if not os.path.exists(self.outfilename):
            print(f"ERROR: Output file {self.outfilename} not found")
            return
        
        # Collect data for all attributes
        data = {}
        for attr in attributes:
            values = self.collect_attribute(attr)
            if values is not None:
                # Remove NaN values
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    data[attr] = values
                else:
                    print(f"WARNING: No valid data found for attribute {attr}")
        
        if not data:
            print("ERROR: No valid data found for any attribute")
            return
        
        # Create subplot layout
        n_attrs = len(data)
        n_cols = 3
        n_rows = (n_attrs + n_cols - 1) // n_cols
        
        if get_data_only:
            return data
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot histograms
        for i, (attr, values) in enumerate(data.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Create histogram
            counts, bins_edges, patches = ax.hist(values, bins=bins, alpha=0.7, edgecolor='black')
            
            # Set labels and title
            ax.set_xlabel(self._get_attribute_label(attr))
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'$N = {len(values)}$\n $\mu = {np.mean(values):.3e}$\n $\sigma = {np.std(values):.3e}$'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(n_attrs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_fig}")
        
        plt.show()

    def plot_attribute_scatter(self, attr_x: str, attr_y: str, 
                              figsize: tuple = (8, 6), save_fig: Optional[str] = None, 
                              color_by: Optional[str] = None, colormap: str = 'viridis', 
                              fxy: Optional[callable] = None, get_data_only = False, **kwargs):
        """
        Create a scatter plot of two attributes with optional color coding and fit line.
        This method generates a scatter plot comparing two attributes from the collected data.
        Points can be colored by a third attribute, and the plot includes correlation 
        coefficient and sample size information. An optional fit function can be overlaid.
        Parameters
        ----------
            Attribute name for x-axis values.
            Attribute name for y-axis values.
        figsize : tuple, optional
            Figure size as (width, height) in inches. Default is (8, 6).
            If provided, saves the figure to this filename. Default is None.
            Attribute name to use for coloring points. If None, uses default colors.
            Default is None.
        colormap : str, optional
            Matplotlib colormap name for coloring points when color_by is specified.
            Default is 'viridis'.
        fxy : callable, optional
            Function y = f(x) to plot as a line overlay. Should accept numpy array
            of x values and return corresponding y values. Default is None.
        get_data_only : bool, optional
            If True, only collects and returns the data arrays without plotting.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib's scatter function.
            Common options include 'alpha', 's' (size), 'marker', etc.
        Returns
        -------
        None
            Displays the plot and optionally saves it to file.
        Notes
        -----
        - Automatically removes NaN values from the data before plotting
        - Displays correlation coefficient (r) and sample size (N) in the plot
        - Includes grid lines with reduced opacity for better readability
        - Uses tight layout for optimal spacing
        - Saves figure with high DPI (300) if save_fig is specified
        Examples
        --------
        >>> # Basic scatter plot
        >>> interface.plot_attribute_scatter('temperature', 'density')
        >>> # Colored by third attribute
        >>> interface.plot_attribute_scatter('temperature', 'density', 
        ...                                 color_by='pressure', colormap='plasma')
        >>> # With fit function
        >>> interface.plot_attribute_scatter('temp', 'dens', 
        ...                                 fit_func=lambda x: 2.5*x + 10)
        >>> # Custom styling and save
        >>> interface.plot_attribute_scatter('temp', 'dens', figsize=(10, 8),
        ...                                 save_fig='scatter.png', alpha=0.8, s=30)
        """
        if not os.path.exists(self.outfilename):
            print(f"ERROR: Output file {self.outfilename} not found")
            return
        
        # Collect data
        x_values = self.collect_attribute(attr_x)
        y_values = self.collect_attribute(attr_y)
        
        if x_values is None or y_values is None:
            print(f"ERROR: Could not collect data for {attr_x} or {attr_y}")
            return
        
        c_values = None
        if color_by is not None:
            c_values = self.collect_attribute(color_by)
            if c_values is None:
                print(f"WARNING: Could not collect data for {color_by}, using default colors")
                color_by = None
        
        # Remove NaN values
        if color_by is not None:
            valid_mask = ~(np.isnan(x_values) | np.isnan(y_values) | np.isnan(c_values))
            c_values = c_values[valid_mask]
        else:
            valid_mask = ~(np.isnan(x_values) | np.isnan(y_values))
        
        x_values = x_values[valid_mask]
        y_values = y_values[valid_mask]
        
        if len(x_values) == 0:
            print("ERROR: No valid data points found")
            return
        
        if get_data_only:
            return x_values, y_values
        
        # Create scatter plot
        plt.figure(figsize=figsize)
        scatter_kwargs = {'alpha': 0.6, 's': 20}
        scatter_kwargs.update(kwargs)
        
        if color_by is not None:
            scatter_kwargs['c'] = c_values
            scatter_kwargs['cmap'] = colormap
            scatter = plt.scatter(x_values, y_values, **scatter_kwargs)
            cbar = plt.colorbar(scatter)
            cbar.set_label(self._get_attribute_label(color_by))
        else:
            plt.scatter(x_values, y_values, **scatter_kwargs)
        
        # Plot fit function if provided
        if fxy is not None:
            x_fit = np.linspace(np.min(x_values), np.max(x_values), 100)
            try:
                y_fit = fxy(x_fit)
                plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='y = f(x)')
                plt.legend()
            except Exception as e:
                print(f"WARNING: Could not evaluate fit function: {e}")
        
        plt.xlabel(self._get_attribute_label(attr_x))
        plt.ylabel(self._get_attribute_label(attr_y))
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_coef = np.corrcoef(x_values, y_values)[0, 1]
        plt.text(0.02, 0.98, f'r = {corr_coef:.3f}\nN = {len(x_values)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_fig}")
        
        plt.show()
        
    def plot_data(self, **kwargs):
        """
        Create a matrix of scatter plots and histograms for the GYRAZE input attributes.
        ----------
        **kwargs : dict
            Additional keyword arguments passed to matplotlib's scatter function
            for the scatter plots. Common options include 'alpha', 's' (size), 'marker', etc.
        Returns
        -------
        None
            Displays the plot matrix.
        Notes
        -----
        - Automatically removes NaN values from the data before plotting
        Examples
        --------
        >>> interface.plot_data(alpha=0.5, s=15)
        """
        scatter_kwargs = {'alpha': 0.1, 's': 20}
        scatter_kwargs.update(kwargs)   
        attributes = ['gamma0','nioverne','TioverTe','phi_norm0']
        nattr = len(attributes)
        
        hist_data = self.plot_attribute_histograms(attributes=attributes, get_data_only=True)
        scatter_data = []
        for i in range(nattr):
            scatter_data.append([])
            for j in range(nattr):
                x_data, y_data = self.plot_attribute_scatter(attributes[j], attributes[i], get_data_only=True)
                scatter_data[i].append((attributes[j], attributes[i], x_data, y_data))

        ndiag = nattr
        fig, axs = plt.subplots(ndiag, ndiag, figsize=(12, 10))

        for k in range(ndiag):
            for l in range(ndiag):
                ax = axs[k, l]
                
                # Diagonal: histogram
                if k == l:
                    ax.hist(hist_data[attributes[k]], bins=25, alpha=0.7, edgecolor='black')
                    ax.set_xlabel(self._get_attribute_label(attributes[k]))
                    ax.set_ylabel('Count')
                    ax.grid(True, alpha=0.3)
                    # remove box around histogram
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.set_yticklabels([])
                    ax.set_ylabel('')
                    # Add statistics text
                    values = hist_data[attributes[k]]
                    stats_text = f'$N = {len(values)}$\n $\mu = {np.mean(values):.3e}$\n $\sigma = {np.std(values):.3e}$'
                    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Lower triangle: scatter plot
                elif k > l:
                    x_attr, y_attr, x_data, y_data = scatter_data[k][l]
                    sc = ax.scatter(x_data, y_data, **scatter_kwargs)
                    ax.set_xlabel(self._get_attribute_label(x_attr))
                    ax.set_ylabel(self._get_attribute_label(y_attr))
                    ax.grid(True, alpha=0.3)
                    # remove upper and right box lines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                else:
                    ax.axis('off')
                    
                # remove xtick labels for non-bottom plots
                if k < ndiag - 1:
                    ax.set_xticklabels([])
                    ax.set_xlabel('')
                # remove ytick labels for non-leftmost plots
                if l > 0:
                    ax.set_yticklabels([])
                    ax.set_ylabel('')
                    
                # Remove grid lines for clarity
                ax.grid(False)
                # remove the little ticks on the axes
                ax.tick_params(axis='both', which='both', length=0)

        fig.tight_layout()

    def _get_attribute_label(self, attr_name):
        labels = {
            'B0': r'$B$ (T)',
            'phi0': r'$\phi$ (V)',
            'phi_norm0': r'$e\phi/T_e$',
            'ni0': r'$n_{i}$ (m$^{-3}$)',
            'ne0': r'$n_{e}$ (m$^{-3}$)',
            'Ti0': r'$T_{i}$ (eV)',
            'Te0': r'$T_{e}$ (eV)',
            'gamma0': r'$\gamma=\rho_e/\lambda_D$',
            'nioverne': r'$n_{i}/n_{e}$',
            'TioverTe': r'$T_{i}/T_{e}$',
            'x0': r'$x$ (m)',
            'y0': r'$y$ (m)',
            'z0': r'$z$ (m)',
            'tf': 'Time frame',
            't0': 'Time (s)',
            'alphadeg': r'$\alpha$ (deg)'
        }
        return labels.get(attr_name, attr_name)

    def get_attribute_statistics(self, attributes: Optional[List[str]] = None):
        if attributes is None:
            attributes = ['B0','phi0','ni0','ne0','Ti0','Te0','gamma0','nioverne','TioverTe']
        
        if not os.path.exists(self.outfilename):
            print(f"ERROR: Output file {self.outfilename} not found")
            return None
        
        stats = {}
        
        for attr in attributes:
            values = self.collect_attribute(attr)
            if values is not None:
                # Remove NaN values
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    stats[attr] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75)
                    }
                else:
                    print(f"WARNING: No valid data found for attribute {attr}")
        
        return stats

    def print_statistics(self, attributes: Optional[List[str]] = None):
        """
        Print formatted statistical summary of attributes.
        
        Parameters:
        -----------
        attributes : list of str, optional
            List of attribute names. If None, uses default list.
        """
        stats = self.get_attribute_statistics(attributes)
        
        if not stats:
            print("No statistics available")
            return
        
        print("\nAttribute Statistics:")
        print("="*80)
        print(f"{'Attribute':<12} {'Count':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-"*80)
        
        for attr, stat in stats.items():
            print(f"{attr:<12} {stat['count']:<8} {stat['mean']:<12.3e} {stat['std']:<12.3e} "
                  f"{stat['min']:<12.3e} {stat['max']:<12.3e}")