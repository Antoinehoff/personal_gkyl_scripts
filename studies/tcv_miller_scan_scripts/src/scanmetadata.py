"""
ScanMetadata: A class to manage and analyze Gkeyll scan metadata.

This module provides a unified interface for loading, analyzing, and visualizing
simulation scan results from JSON or HDF5 metadata files. All data, including
composite fields (location differences), is pre-computed at initialization for
fast access.

Example usage:
    # Load metadata from HDF5 (produced by gather_metadata.py)
    scan = ScanMetadata('tcv_miller_scan_big_metadata_frame_500_navg_25.h5')
    
    # Load metadata from legacy JSON
    scan = ScanMetadata('tcv_miller_scan_big_metadata_frame_500_navg_25.json')
    scan.info()  # Display available data
    
    # Plot on kappa-delta plane at fixed power (direct, recommended)
    scan.plot_contour_grid(['Ti_core', 'Ti_core_lcfs'],
                          fixed_params={'energy_srcCORE': 1e6})
    
    # Plot on kappa-power plane at fixed delta
    scan.plot_contour_grid(['Ti_core'],
                          fixed_params={'delta': 0.3},
                          cmap='viridis')
    
    # Two-step approach (if you need to reuse data)
    data = scan.extract_field_data(['Ti_core'], fixed_params={'delta': 0.3})
    scan.plot_contour_grid(data, ['Ti_core'])  # Reuse data for multiple plots
    
    # Compare profiles across parameter values
    scan.compare_profiles('kappa', [1.1, 1.2, 1.3, 1.4],
                         fixed_params={'delta': 0.3, 'energy_srcCORE': 1e6})
"""

import pygkyl
import itertools
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import copy

# plt.style.use('seaborn-v0_8-darkgrid')
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({"text.usetex": False, "font.size": 14})


class ScanMetadata:
    """
    A class to manage and analyze scan metadata from Gkeyll simulations.
    
    This class reads a JSON or HDF5 file containing simulation data, pre-loads
    all fields (including composite fields like location differences), and
    provides methods for extracting, plotting, and analyzing field profiles
    across parameter scans. All fields and scan parameters are auto-detected
    from the metadata file.
    
    Attributes:
        metadata_file (Path): Path to the metadata file (JSON or HDF5)
        metadata (List[Dict]): List of simulation metadata dictionaries
        fields (List[str]): Available base field names (e.g., 'Ti', 'Te', 'ne', 'phi')
        locations (List[str]): Available spatial locations (e.g., 'core', 'edge', 'lcfs', 'sol')
        scan_params (Dict[str, List]): Scan parameter arrays detected from data
        all_fields (List[str]): All available fields including composite fields
        field_symbols (Dict[str, str]): LaTeX symbols for fields
        field_units (Dict[str, str]): Units for fields
        data (Dict[str, np.ndarray]): Pre-loaded multi-dimensional arrays for all fields
    """
    
    RAXIS = 0.87
    AMID = 0.24 
    
    # Default field symbols for LaTeX plotting
    DEFAULT_FIELD_SYMBOLS = {
        'Ti': r'T_i', 'Te': r'T_e', 'ne': r'n_e', 'phi': r'\phi',
        'Pi': r'P_i', 'Pe': r'P_e', 'ni': r'n_i', 'upar': r'u_\parallel',
        'hflux_xi' : r'Q_{xi}', 'hflux_xe' : r'Q_{xe}',
        'pflux_xi' : r'\Gamma_{xi}', 'pflux_xe' : r'\Gamma_{xe}',
        'kn': r'k_n', 'kTi': r'k_{T_i}', 'kTe': r'k_{T_e}',
        'avg_dt' : r'\Delta t', 'lambda_q' : r'\lambda_q',
    }
    
    # Default field units
    DEFAULT_FIELD_UNITS = {
        'Ti': r'[eV]', 'Te': r'[eV]', 'ne': r'[$m^{-3}$]', 'phi': r'[V]',
        'Pi': r'[kPa]', 'Pe': r'[kPa]', 'ni': r'[$m^{-3}$]', 'upar': r'[m/s]',
        'hflux_xi': r'[MW/m$^2$]', 'hflux_xe': r'[MW/m$^2$]',
        'pflux_xi': r'[m$^{-2}$ s$^{-1}$]', 'pflux_xe': r'[m$^{-2}$ s$^{-1}$]',
        'T': r'[eV]', 'n': r'[$m^{-3}$]', 'p': r'[V]', 'P': r'[kPa]',
        'Dxi': r'[m$^2$/s]', 'Dxe': r'[m$^2$/s]', 'chixi': r'[m$^2$/s]', 'chixe': r'[m$^2$/s]',
        'kn': r'', 'kTi': r'', 'kTe': r'',
        'avg_dt': r'[ns]', 'lambda_q': r'[mm]',
    }
    
    # Default field scaling factors
    DEFAULT_FIELD_SCALING = {
        'hflux_xi': 1e-6, 'hflux_xe': 1e-6,
        'pflux_xi': 1e-6, 'pflux_xe': 1e-6,
        'avg_dt' : 1e9, 'lambda_q': 1e3,
    }
    
    # Default location symbols (r/a values)
    DEFAULT_LOCATION_VALUES = {
        'core': 0.85, 'edge': 0.9, 'lcfs': 1.0, 'sol': 1.2, 'limlo': 1.2, 'limup': 1.2
    }
    DEFAULT_LOCATION_SYMBOLS = {
        'core': r'\text{core}', 'edge': r'\text{edge}', 'lcfs': r'\text{sep}', 'sol': r'\text{SOL}',
        'limlo': r'\text{lo}', 'limup': r'\text{lu}',
    }
    
    # Known scan parameters to look for
    KNOWN_SCAN_PARAMS = ['kappa', 'delta', 'energy_srcCORE']
    KNOWN_SCAN_PARAM_SYMBOLS = { 'kappa': r'\kappa', 'delta': r'\delta', 'energy_srcCORE': r'P_{\text{in}}' }
    
    # Metadata keys (not scan parameters or fields)
    METADATA_KEYS = {'simdir', 'scanidx', 'tend', 'avg_dt', 'frame', 'navg', 
                     'intmom', 'vol_frac', 'lambda_q'}
    
    def __init__(self, metadata_file: Union[str, Path], bflux_tavg: float = 25.0):
        """
        Initialize ScanMetadata from a JSON or HDF5 file and pre-load all data.
        
        This will read the metadata file, detect all fields and scan parameters,
        and pre-compute all field values including composite fields (location
        differences) into multi-dimensional arrays for fast access.
        
        Parameters:
            metadata_file: Path to JSON (.json) or HDF5 (.h5/.hdf5) metadata file
            bflux_tavg: Averaging window (in mus) for boundary fluxes (default: 25.0)
        """
        self.metadata_file = Path(metadata_file)
        self.bflux_tavg = bflux_tavg
        
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        # Load metadata (supports both JSON and HDF5)
        self._load_metadata()
        
        if not self.metadata or len(self.metadata) == 0:
            raise ValueError("Metadata file is empty")
        
        # Auto-detect fields and locations from metadata
        self._detect_fields_and_locations()
        
        # Extract scan parameters
        self._extract_scan_parameters()
        
        # Setup field properties (symbols, units)
        self._setup_field_properties()
        
        # Generate all composite fields
        self._generate_composite_fields()
        
        # Extract simulation directory pattern
        self._detect_sim_pattern()
        
        # Pre-load all data including composite fields
        self._preload_all_data()
    
    # ------------------------------------------------------------------
    # Metadata loading helpers
    # ------------------------------------------------------------------

    def _load_metadata(self):
        """Dispatch to the correct loader based on file extension."""
        suffix = self.metadata_file.suffix.lower()
        if suffix in ('.h5', '.hdf5'):
            self._load_h5()
        else:
            self._load_json()

    def _load_json(self):
        """Load metadata from a JSON file."""
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)

    def _load_h5(self):
        """Load metadata from an HDF5 file produced by gather_metadata.py.
        
        The HDF5 layout expected is:
          scan_NNNNN/  (one group per simulation)
            attrs: simdir, scanidx, kappa, delta, energy_srcCORE, tend,
                   avg_dt, Ti_core, Te_core, ne_core, phi_core, ...
            bflux/  (optional, sub-group with time-series datasets)
              <flux_name>/
                datasets: time, values
                attrs: tunits, vunits, fluxname
        """
        self.metadata = []
        with h5py.File(self.metadata_file, 'r') as f:
            for grp_name in sorted(f.keys()):
                grp = f[grp_name]
                entry = {}
                # Read all scalar attributes, converting numpy scalars to Python natives
                for key, val in grp.attrs.items():
                    if isinstance(val, bytes):
                        entry[key] = val.decode('utf-8')
                    elif hasattr(val, 'item'):
                        entry[key] = val.item()
                    else:
                        entry[key] = val
                # Read intmom time-series datasets if present
                if 'intmom' in grp:
                    intmom_dict = {}
                    for intmom_name, flux_grp in grp['intmom'].items():
                        # Decode string attrs that h5py may return as bytes
                        def _attr_str(v):
                            if isinstance(v, (bytes, np.bytes_)):
                                return v.decode('utf-8')
                            if hasattr(v, 'item'):
                                v = v.item()
                            return str(v) if not isinstance(v, str) else v
                        try:
                            intmom_dict[intmom_name] = {
                                'time':     np.asarray(flux_grp['time'][:], dtype=float),
                                'values':   np.asarray(flux_grp['values'][:], dtype=float),
                                'tunits':   _attr_str(flux_grp.attrs.get('tunits', 'mus')),
                                'vunits':   _attr_str(flux_grp.attrs.get('vunits', '')),
                                'name': _attr_str(flux_grp.attrs.get('name', intmom_name)),
                            }
                        except Exception as exc:
                            print(f'Warning: could not load intmom dataset {intmom_name}: {exc}')
                    entry['intmom'] = intmom_dict
                self.metadata.append(entry)

    # ------------------------------------------------------------------

    def _detect_fields_and_locations(self):
        """Automatically detect available fields and locations from metadata keys."""
        sample_keys = set(self.metadata[0].keys())
        
        # Find all field_location patterns
        field_location_pairs = []
        for key in sample_keys:
            if key in self.METADATA_KEYS or key in self.KNOWN_SCAN_PARAMS:
                continue
            parts = key.split('_')
            if len(parts) == 2:
                field, location = parts
                field_location_pairs.append((field, location))
            elif len(parts) == 3: # fluxes
                field = f'{parts[0]}_{parts[1]}'
                location = parts[2]
                field_location_pairs.append((field, location))
        
        # Extract unique fields and locations
        self.fields = sorted(list(set(f for f, l in field_location_pairs)))
        self.locations = sorted(list(set(l for f, l in field_location_pairs)))
        
        # Store available field_location keys
        self.available_field_keys = [f'{f}_{l}' for f, l in field_location_pairs]
        
        # Add simulation metadata
        self.available_field_keys.append('avg_dt')
        self.available_field_keys.append('lambda_q')
        self.available_field_keys.append('vol_frac')
    
    def _extract_scan_parameters(self):
        """Extract scan parameter arrays from metadata."""
        self.scan_params = {}
        
        # Check for known scan parameters
        for param in self.KNOWN_SCAN_PARAMS:
            if param in self.metadata[0]:
                values = sorted(list(set(entry[param] for entry in self.metadata)))
                # if len(values) > 1:  # Only include if it varies
                self.scan_params[param] = values
        
        # Create combination list for indexing
        self.scan_keys = list(self.scan_params.keys())
        values = [self.scan_params[k] for k in self.scan_keys]
        self.combinations = list(itertools.product(*values))
    
    def _setup_field_properties(self):
        """Setup field symbols and units, using defaults where available."""
        # Field symbols
        self.field_symbols = {}
        for field in self.fields:
            self.field_symbols[field] = self.DEFAULT_FIELD_SYMBOLS.get(field, field)
        # Add pressure symbols
        self.field_symbols['Pi'] = self.DEFAULT_FIELD_SYMBOLS.get('Pi', r'P_i')
        self.field_symbols['Pe'] = self.DEFAULT_FIELD_SYMBOLS.get('Pe', r'P_e')
        
        # Field units
        self.field_units = dict(self.DEFAULT_FIELD_UNITS)
        for field in self.fields:
            if field not in self.field_units:
                self.field_units[field] = ''
                
        # Field scaling factors (for normalization in plots)
        self.field_scaling = dict(self.DEFAULT_FIELD_SCALING)
        
        # Location symbols
        self.location_symbols = {}
        for loc in self.locations:
            self.location_symbols[loc] = self.DEFAULT_LOCATION_SYMBOLS.get(loc, loc)
        
        # Reference values for normalization
        self.field_refvals = {
            'T': 200,                    # eV
            'n': 1e19,                   # m^-3
            'p': 200,                    # V
            'P': 200 * 1.602e-19 * 1e19, # Pa
        }
    
    def _generate_composite_fields(self):
        """Generate list of all available fields including composite ones."""
        # Basic field_location combinations
        self.all_fields = list(self.available_field_keys)
        
        # Add pressure fields
        for loc in self.locations:
            for pfield in ['Pi', 'Pe']:
                key = f'{pfield}_{loc}'
                if key not in self.all_fields:
                    self.all_fields.append(key)
        
        # Add field differences between locations
        all_base_fields = list(set(self.fields + ['Pi', 'Pe']))
        for field in all_base_fields:
            for i, loc1 in enumerate(self.locations):
                for j, loc2 in enumerate(self.locations):
                    if i >= j:
                        continue
                    key = f'{field}_{loc1}_{loc2}'
                    if key not in self.all_fields:
                        self.all_fields.append(key)
        
        # Add avg dt
        self.all_fields.append('avg_dt')
        
        # Add lambda_q
        self.all_fields.append('lambda_q')
        
        # Build all_field_symbols dict
        self.all_field_symbols = {}
        
        # Basic fields
        for field in self.fields:
            for loc in self.locations:
                key = f'{field}_{loc}'
                fs = self.field_symbols.get(field, field)
                ls = self.location_symbols.get(loc, loc)
                self.all_field_symbols[key] = r'${' + fs + r'}^{' + ls + r'}$'
        
        # Pressure fields
        for loc in self.locations:
            for pfield in ['Pi', 'Pe']:
                key = f'{pfield}_{loc}'
                fs = self.field_symbols.get(pfield, pfield)
                ls = self.location_symbols.get(loc, loc)
                self.all_field_symbols[key] = r'${' + fs + r'}^{' + ls + r'}$'
        
        # Difference fields
        for field in all_base_fields:
            for i, loc1 in enumerate(self.locations):
                key = f'{field}_norm_{loc1}'
                fs = self.field_symbols.get(field, field)
                ls1 = self.location_symbols.get(loc1, loc1)
                self.all_field_symbols[key] = (
                    r'${' + fs + r'}^{' + ls1 + r'} / {' + fs + r'}^{\text{sep}}$'
                )                
                for j, loc2 in enumerate(self.locations):
                    if i >= j:
                        continue
                    key = f'{field}_{loc1}_{loc2}'
                    fs = self.field_symbols.get(field, field)
                    ls1 = self.location_symbols.get(loc1, loc1)
                    ls2 = self.location_symbols.get(loc2, loc2)
                    self.all_field_symbols[key] = (
                        r'${' + fs + r'}^{' + ls1 + r'} - {' + fs + r'}^{' + ls2 + r'}$'
                    )
        
        # Add diffusion coefficients symbols
        for flux in ['hflux_xi', 'hflux_xe', 'pflux_xi', 'pflux_xe']:
            for suffix in ['Dx', 'chix']:
                symbol = 'D' if suffix == 'Dx' else r'\chi'
                key = f'{suffix}{flux[-1]}'
                if key not in self.all_field_symbols:
                    self.all_field_symbols[key] = r'${' + symbol + r'}_{x' + flux[-1] + r'}$'
                    
        # Add metadata fields
        for key in self.METADATA_KEYS:
            self.all_field_symbols[key] = r'${'+self.DEFAULT_FIELD_SYMBOLS.get(key,key)+r'}$'
    
    def _detect_sim_pattern(self):
        """Detect simulation directory pattern from metadata."""
        if 'simdir' in self.metadata[0]:
            simdir = self.metadata[0]['simdir']
            # Extract base pattern (e.g., "tcv_miller_scan_big/tcv_miller_scan_big_")
            parts = simdir.rstrip('/').rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                self.sim_base_dir = parts[0].rsplit('/', 1)[0] if '/' in parts[0] else '.'
                self.sim_prefix = parts[0].rsplit('/', 1)[-1] if '/' in parts[0] else parts[0]
            else:
                self.sim_base_dir = "."
                self.sim_prefix = "sim"
        else:
            self.sim_base_dir = "."
            self.sim_prefix = "sim"
    
    def _preload_all_data(self):
        """Pre-load all data from metadata and compute composite fields."""
        # Get dimensions
        param_shapes = [len(self.scan_params[k]) for k in self.scan_keys]
        
        # Initialize data storage: dict of field_name -> ndarray
        self.data = {}
        
        # Extract all basic fields from metadata
        all_field_keys = self.available_field_keys.copy()
        
        # Add pressure fields to extract
        for loc in self.locations:
            for pfield in ['Pi', 'Pe']:
                key = f'{pfield}_{loc}'
                if key not in all_field_keys:
                    all_field_keys.append(key)
        
        # Pre-allocate arrays for all fields
        for field_key in all_field_keys:
            self.data[field_key] = np.zeros(param_shapes)
        
        # Fill arrays with data from metadata
        for entry in self.metadata:
            # Get indices for this entry
            indices = tuple(self.scan_params[k].index(entry[k]) for k in self.scan_keys)
            
            # Extract field values
            for field_key in self.available_field_keys:
                if field_key in entry:
                    self.data[field_key][indices] = entry[field_key]
            
            # Compute pressure fields
            for loc in self.locations:
                if f'Ti_{loc}' in entry and f'ne_{loc}' in entry:
                    self.data[f'Pi_{loc}'][indices] = (
                        1.5 * entry[f'Ti_{loc}'] * entry[f'ne_{loc}'] * 1.602e-19 * 1e-3
                    )
                if f'Te_{loc}' in entry and f'ne_{loc}' in entry:
                    self.data[f'Pe_{loc}'][indices] = (
                        1.5 * entry[f'Te_{loc}'] * entry[f'ne_{loc}'] * 1.602e-19 * 1e-3
                    )
        
        # Add also the parameters as fields for easy access
        for param in self.scan_keys:
            self.data[param] = np.zeros(param_shapes)
            for entry in self.metadata:
                indices = tuple(self.scan_params[k].index(entry[k]) for k in self.scan_keys)
                self.data[param][indices] = entry[param]
        
        # Compute all composite fields (differences between locations and LCFS normalized)
        all_base_fields = list(set(self.fields + ['Pi', 'Pe']))
        for field in all_base_fields:
            for i, loc1 in enumerate(self.locations):
                key = f'{field}_norm_{loc1}'
                field1 = f'{field}_{loc1}'
                fieldlcfs = f'{field}_lcfs'
                self.data[key] = self.data[field1]/self.data[fieldlcfs]
                
                for j, loc2 in enumerate(self.locations):
                    if i >= j:
                        continue
                    key = f'{field}_{loc1}_{loc2}'
                    field1 = f'{field}_{loc1}'
                    field2 = f'{field}_{loc2}'
                    if field1 in self.data and field2 in self.data:
                        self.data[key] = self.data[field1] - self.data[field2]
        
        # Add ion-electron temperature ratio
        for loc in self.locations:
            key = f'tau_{loc}'
            self.data[key] = self.data[f'Ti_{loc}']/self.data[f'Te_{loc}']
            self.all_field_symbols[key] = r'$(T_i/T_e)^{' + self.location_symbols.get(loc, loc) + r'}$'         
        
        # check if we have fluxes data
        for flux in ['hflux_xi', 'hflux_xe', 'pflux_xi', 'pflux_xe']:
            for loc in self.locations:
                key = f'{flux}_{loc}'
                if key not in self.data:
                    self.data[key] = np.zeros(param_shapes)
                    
        # Compute the diffusivity coefficients based on core and lcfs values
        for flux in ['pflux_xi', 'pflux_xe']:
            s_ = flux[-1]  # Extract the last character (e.g., 'i' or 'e')
            pflux_lcfs = self.data[f'{flux}_lcfs']
            delta_n = self.data[f'ne_edge'] - self.data[f'ne_lcfs']
            delta_x = self.DEFAULT_LOCATION_VALUES['edge'] - self.DEFAULT_LOCATION_VALUES['lcfs']
            gradn = delta_n / (self.AMID*delta_x)
            Dx = -pflux_lcfs / gradn
            self.data[f'Dx{flux[-1]}'] = Dx
            self.all_field_symbols[f'Dx{flux[-1]}'] = r'$D_{x' + flux[-1] + r'}^{\text{sep}}$'
            
        for flux in ['hflux_xi', 'hflux_xe']:
            s_ = flux[-1]  # Extract the last character (e.g., 'i' or 'e')
            hflux_lcfs = self.data[f'{flux}_lcfs']
            pflux_lcfs = self.data[f'pflux_x{s_}_lcfs']
            Tlcfs = self.data[f'T{s_}_lcfs']
            
            delta_n = self.data[f'ne_edge'] - self.data[f'ne_lcfs']
            delta_T = self.data[f'T{s_}_edge'] - self.data[f'T{s_}_lcfs']
            delta_x = self.DEFAULT_LOCATION_VALUES['edge'] - self.DEFAULT_LOCATION_VALUES['lcfs']
            gradn = delta_n / (self.AMID*delta_x)
            gradT = delta_T / (self.AMID*delta_x)
            
            chix = -(hflux_lcfs - 1.5 * Tlcfs * pflux_lcfs) / (gradT *  delta_n)
            
            self.data[f'chix{flux[-1]}'] = chix
            self.all_field_symbols[f'chix{flux[-1]}'] = r'$\chi_{' + flux[-1] + r'}^{\text{sep}}$'
            
        self.data[f'chixe_over_chixi'] = self.data['chixe'] / self.data['chixi']
        self.all_field_symbols[f'chixe_over_chixi'] = r'$\chi_{e}^{\text{sep}} / \chi_{i}^{\text{sep}}$'
        
        self.data[f'De_over_chitot'] = self.data['Dxe'] / (self.data['chixe'] + self.data['chixi'])
        self.all_field_symbols[f'De_over_chitot'] = r'$D_{e}^{\text{sep}} / (\chi_{e}^{\text{sep}} + \chi_{i}^{\text{sep}})$'
        
        self.data[f'dne_rel'] = (self.data['ne_edge'] - self.data['ne_lcfs']) / self.data['ne_edge'] * 100
        self.all_field_symbols[f'dne_rel'] = r'$\Delta n_e / n_{e}$ [%]'
        self.data[f'dTe_rel'] = (self.data['Te_edge'] - self.data['Te_lcfs']) / self.data['Te_edge'] * 100
        self.all_field_symbols[f'dTe_rel'] = r'$\Delta T_e / T_{e}$ [%]'
        self.data[f'dTi_rel'] = (self.data['Ti_edge'] - self.data['Ti_lcfs']) / self.data['Ti_edge'] * 100
        self.all_field_symbols[f'dTi_rel'] = r'$\Delta T_i / T_{i}$ [%]'
        
        Roverdr = self.RAXIS / ((self.DEFAULT_LOCATION_VALUES['lcfs'] - self.DEFAULT_LOCATION_VALUES['edge']) * self.AMID)
        self.data[f'kne'] = (self.data['ne_edge'] - self.data['ne_lcfs']) / self.data['ne_edge'] * Roverdr
        self.all_field_symbols[f'kne'] = r'$R/L_{n_e}$'
        self.data[f'kTe'] = (self.data['Te_edge'] - self.data['Te_lcfs']) / self.data['Te_edge'] * Roverdr
        self.all_field_symbols[f'kTe'] = r'$R/L_{T_e}$'
        self.data[f'kTi'] = (self.data['Ti_edge'] - self.data['Ti_lcfs']) / self.data['Ti_edge'] * Roverdr
        self.all_field_symbols[f'kTi'] = r'$R/L_{T_i}$'
        
        self.data[f'Edens_i'] = 1.5 * self.data['Ti_core'] * self.data['ne_core'] * 1.602e-19
        self.all_field_symbols[f'Edens_i'] = r'$E_{dens,i}$ [J/m$^3$]'
        
        # Pre-load integrated moment time-averaged data
        self._preload_avg_intmom_data()
        
        # Confinement time estimate: \tau_E ~ W / P_sol
        p, w = 0, 0
        for s_ in ['i', 'e']:
            w += self.data[f'W{s_}']
            for l_ in ['x_u', 'z_u', 'z_l']:
                p += self.data[f'bflux_{l_}_H{s_}']
        self.data[f'tau_E'] = w/p
        self.all_field_symbols[f'tau_E'] = r'$\tau_E$ [s]'            
            
    def _preload_avg_intmom_data(self):
        """Average integrated moment time series over the last bflux_tavg microseconds.

        For each int moment name (e.g. 'bflux_x_l_ne'), the mean of values in the
        time window [t_end - bflux_tavg, t_end] is stored in self.data under
        the same key.  Field symbols are built from the bflux name components.

        This is a no-op when no metadata entry contains bflux data.
        """
        # Find first entry that has integrated moment data to get the flux names
        intmom_names = None
        for entry in self.metadata:
            if 'intmom' in entry and entry['intmom']:
                intmom_names = [name for name in entry['intmom'].keys()]
                break
        if intmom_names is None:
            return  # No integrated moment data available

        param_shapes = [len(self.scan_params[k]) for k in self.scan_keys]

        # Pre-allocate
        for name in intmom_names:
            self.data[name] = np.full(param_shapes, np.nan)

        def _find_idx(param, value):
            """Find index using exact match first, then nearest-float fallback."""
            vals = self.scan_params[param]
            try:
                return vals.index(value)
            except ValueError:
                fval = float(value)
                dists = [abs(float(v) - fval) for v in vals]
                best = int(np.argmin(dists))
                if dists[best] / (abs(float(vals[best])) + 1e-300) < 1e-6:
                    return best
                raise

        # Fill from metadata
        for entry in self.metadata:
            if 'intmom' not in entry or not entry['intmom']:
                continue
            try:
                indices = tuple(_find_idx(k, entry[k]) for k in self.scan_keys)
            except (ValueError, KeyError) as exc:
                continue
            for name in intmom_names:
                if name not in entry['intmom']:
                    continue
                t = np.asarray(entry['intmom'][name]['time'], dtype=float)
                v = np.asarray(entry['intmom'][name]['values'], dtype=float)
                if len(t) == 0:
                    continue
                t_end = t[-1]
                mask = t >= (t_end - self.bflux_tavg)
                self.data[name][indices] = np.nanmean(v[mask]) if mask.any() else np.nan     

        # Build human-readable symbols:
        #   bflux_x_l_ne -> $\Gamma_{x,\ell}^{n_e}$  (particle flux)
        #   bflux_x_l_He -> $Q_{x,\ell}^{e}$          (heat flux)
        _species_sym  = {'ne': r'n_e', 'ni': r'n_i', 'He': r'e', 'Hi': r'i'}
        _species_base = {'ne': r'\Gamma', 'ni': r'\Gamma', 'He': r'Q', 'Hi': r'Q'}
        _dir_sym  = {'x': 'x', 'z': 'z'}
        _side_sym = {'l': r'\ell', 'u': 'u'}
        for name in intmom_names:
            parts = name.split('_')  # ['bflux', dir, side, species]
            if len(parts) == 4:
                _, d, s, sp = parts
                base   = _species_base.get(sp, r'\Gamma')
                sp_sym = _species_sym.get(sp, sp)
                d_sym  = _dir_sym.get(d, d)
                s_sym  = _side_sym.get(s, s)
                sym = r'$' + base + r'_{' + d_sym + ',' + s_sym + r'}^{' + sp_sym + r'}$'
            else:
                sym = name
            self.all_field_symbols[name] = sym
            if name not in self.field_units:
                self.field_units[name] = r'[s$^{-1}$]'

        # --- Per-(dir,side) species totals: bflux_{dir}_{side}_tot ---
        # Group flux names by their (dir, side) pair and sum over all species.
        bflux_names = [n for n in intmom_names if 'flux' in n]
        by_dirside = {}
        for name in bflux_names:
            parts = name.split('_')
            if len(parts) == 4:
                _, d, s, _sp = parts
                by_dirside.setdefault((d, s), []).append(name)

        for (d, s), members in by_dirside.items():
            tot_key = f'bflux_{d}_{s}_tot'
            stack = np.stack([self.data[m] for m in members], axis=0)
            self.data[tot_key] = np.nansum(stack, axis=0)
            d_sym = _dir_sym.get(d, d)
            s_sym = _side_sym.get(s, s)
            self.all_field_symbols[tot_key] = (
                r'$(\Gamma+Q)_{' + d_sym + r',' + s_sym + r'}$'
            )
            self.field_units[tot_key] = r'[s$^{-1}$]'
            
        _wall_sides      = {('z', 'l'), ('z', 'u'), ('x', 'u')}
        _particle_species = {'ne', 'ni'}
        _heat_species     = {'He', 'Hi'}
        
        # --- Species aggregates: sum over all species for a given (dir, side) ---
        for dir in _dir_sym:
            for side in _side_sym:
                self.data[f'bflux_{dir}_{side}_n'] = self.data[f'bflux_{dir}_{side}_ne'] + self.data[f'bflux_{dir}_{side}_ni']
                self.data[f'bflux_{dir}_{side}_H'] = self.data[f'bflux_{dir}_{side}_He'] + self.data[f'bflux_{dir}_{side}_Hi']
                self.all_field_symbols[f'bflux_{dir}_{side}_n'] = (
                    r'$\Gamma_{' + _dir_sym.get(dir, dir) + r',' + _side_sym.get(side, side) + r'}$ [s$^{-1}$]'
                )                
                self.all_field_symbols[f'bflux_{dir}_{side}_H'] = (
                    r'$Q_{' + _dir_sym.get(dir, dir) + r',' + _side_sym.get(side, side) + r'}$ [MW]'
                )

        # --- Wall aggregates: sum over wall boundaries (z_l, z_u, x_u) ---
        wall_n_terms = [
            n for n in bflux_names
            if len(n.split('_')) == 4
            and (n.split('_')[1], n.split('_')[2]) in _wall_sides
            and n.split('_')[3] in _particle_species
        ]
        wall_H_terms = [
            n for n in bflux_names
            if len(n.split('_')) == 4
            and (n.split('_')[1], n.split('_')[2]) in _wall_sides
            and n.split('_')[3] in _heat_species
        ]

        if wall_n_terms:
            stack = np.stack([self.data[m] for m in wall_n_terms], axis=0)
            self.data['bflux_wall_n'] = np.nansum(stack, axis=0)
            self.all_field_symbols['bflux_wall_n'] = r'$\Gamma_\mathrm{wall}$'
            self.field_units['bflux_wall_n'] = r'[s$^{-1}$]'

        if wall_H_terms:
            stack = np.stack([self.data[m] for m in wall_H_terms], axis=0)
            self.data['bflux_wall_H'] = np.nansum(stack, axis=0)
            self.all_field_symbols['bflux_wall_H'] = r'$Q_\mathrm{wall}$'
            self.field_units['bflux_wall_H'] = r'[MW]'

    def _base_field(self, key):
        base_field = copy.copy(key)
        for loc in self.DEFAULT_LOCATION_VALUES.keys():
            base_field = base_field.replace(f'_{loc}', '')
        return base_field
    
    def _get_slices(self, power_val):
        return tuple(
            self.scan_params[k].index(power_val) if k == 'energy_srcCORE'
            else slice(None)
            for k in self.scan_keys
        )
    def info(self):
        """Display detailed information about the metadata and available data."""
        print("=" * 80)
        print("ScanMetadata Information")
        print("=" * 80)
        print(f"\nMetadata file: {self.metadata_file}")
        print(f"Total simulations: {len(self.metadata)}")
        print(f"Pre-loaded fields: {len(self.data)} (including composites)")
        
        # Calculate memory usage
        total_bytes = sum(arr.nbytes for arr in self.data.values())
        total_mb = total_bytes / (1024 * 1024)
        print(f"Memory usage: {total_mb:.2f} MB")
        
        print("\n" + "-" * 80)
        print("Detected Fields:")
        print("-" * 80)
        for field in self.fields:
            symbol = self.field_symbols.get(field, field)
            unit = self.field_units.get(field, '')
            print(f"  {field:10s} - {symbol:10s} {unit}")
        
        print("\n" + "-" * 80)
        print("Detected Locations:")
        print("-" * 80)
        for loc in self.locations:
            symbol = self.location_symbols.get(loc, loc)
            print(f"  {loc:10s} - r/a = {symbol}")
        
        print("\n" + "-" * 80)
        print("Scan Parameters:")
        print("-" * 80)
        for param, values in self.scan_params.items():
            print(f"  {param:20s}: {len(values)} values")
            print(f"    Range: [{min(values)}, {max(values)}]")
            if len(values) <= 10:
                print(f"    Values: {values}")
        
        print("\n" + "-" * 80)
        print("Available Field Keys (field_location):")
        print("-" * 80)
        # Group by field
        for field in self.fields:
            keys = [k for k in self.available_field_keys if k.startswith(f'{field}_')]
            print(f"  {field}: {', '.join(keys)}")
        
        print("\n" + "-" * 80)
        print("Composite Fields (differences between locations):")
        print("-" * 80)
        diff_fields = [f for f in self.data.keys() if f.count('_') == 2]
        print(f"  Total: {len(diff_fields)} difference fields pre-computed")
        if diff_fields:
            print(f"  Examples: {', '.join(diff_fields[:5])}")
        
        print("\n" + "-" * 80)
        print("Sample Data Ranges (pre-loaded):")
        print("-" * 80)
        sample_fields = [k for k in self.available_field_keys[:4]]
        for key in sample_fields:
            if key in self.data:
                values = self.data[key]
                print(f"  {key:15s}: [{values.min():.3e}, {values.max():.3e}]")
        
        print("\n" + "-" * 80)
        print("Main Methods:")
        print("-" * 80)
        print("  .info()                                - Display this information")
        print("  .plot_contour_grid(fields, fixed_params) - Plot 2D contours on any parameter plane")
        print("  .compare_profiles(vary_param, vary_vals) - Compare profiles varying 1 parameter")
        print("  .get_scan_data(field, fixed_params)    - Get scan data on any parameter plane")
        print("\n" + "-" * 80)
        print("Advanced Methods:")
        print("-" * 80)
        print("  .extract_field_data(fields, fixed_params) - Extract data (for reuse)")
        print("  .get_sim_index(params)                 - Get simulation index from parameters")
        print("  .setup_simulation(scanidx)             - Setup pygkyl simulation")
        print("  .get_profile(params, field)            - Get 1D profile from simulation")
        print("=" * 80)
    
    def __repr__(self):
        """String representation of ScanMetadata."""
        return (f"ScanMetadata(file='{self.metadata_file.name}', "
                f"n_sims={len(self.metadata)}, "
                f"n_fields={len(self.data)}, "
                f"params={list(self.scan_params.keys())})")
    
    def get_sim_index(self, params: Dict[str, Any]) -> int:
        """
        Get simulation index from parameter dictionary.
        
        Parameters:
            params: Dictionary with scan parameter values
                    (e.g., {'kappa': 1.5, 'delta': 0.3, 'energy_srcCORE': 1e6})
        
        Returns:
            Index of the simulation in the combinations list
        
        Raises:
            ValueError: If parameters not found in scan combinations
        """
        parameters = tuple(params[key] for key in self.scan_keys)
        try:
            return self.combinations.index(parameters)
        except ValueError:
            raise ValueError(f"Parameters {params} not found in scan combinations.\n"
                           f"Available scan keys: {self.scan_keys}")
    
    def extract_field_data(self, field_names: List[str],
                          fixed_params: Optional[Dict[str, Any]] = None,
                          vary_params: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract field data with flexible parameter selection from pre-loaded data.
        
        Parameters:
            field_names: List of field names to extract (e.g., ['Ti_core', 'Ti_core_lcfs'])
            fixed_params: Parameters to hold constant (e.g., {'energy_srcCORE': 1e6})
            vary_params: Parameters to vary (must be exactly 2). If None, auto-detected.
        
        Returns:
            Dictionary containing:
                - 2D arrays for each field
                - 'x_param': name of x-axis parameter
                - 'y_param': name of y-axis parameter
                - 'x_vals': 2D meshgrid of x parameter values
                - 'y_vals': 2D meshgrid of y parameter values
        
        Examples:
            # Kappa-delta plane at fixed power
            data = scan.extract_field_data(['Ti_core'], fixed_params={'energy_srcCORE': 1e6})
            
            # Kappa-power plane at fixed delta with composite field
            data = scan.extract_field_data(['Ti_core_lcfs'], fixed_params={'delta': 0.3})
        """
        if fixed_params is None:
            fixed_params = {}
        
        # Determine varying parameters
        if vary_params is None:
            vary_params = [p for p in self.scan_keys if p not in fixed_params]
        
        if len(vary_params) != 2:
            raise ValueError(f"Must have exactly 2 varying parameters, got {len(vary_params)}: {vary_params}. "
                           f"Available scan params: {self.scan_keys}")
        
        # Prioritize delta as x-axis if present
        if 'delta' in vary_params:
            delta_idx = vary_params.index('delta')
            other_idx = 1 - delta_idx
            x_param = 'delta'
            y_param = vary_params[other_idx]
        else:
            x_param, y_param = vary_params
        
        # Build index slices for fixed parameters
        slices = []
        for param in self.scan_keys:
            if param in fixed_params:
                idx = self.scan_params[param].index(fixed_params[param])
                slices.append(idx)
            else:
                slices.append(slice(None))
        
        dict_out = {}
        
        # Extract fields from pre-loaded data
        for field in field_names:
            if field not in self.data:
                print(f"Warning: Field '{field}' not found in pre-loaded data, skipping")
                continue
            
            # Slice the data
            data_slice = self.data[field][tuple(slices)]
            
            # Determine which axes correspond to x_param and y_param
            x_axis = self.scan_keys.index(x_param)
            y_axis = self.scan_keys.index(y_param)
            
            # Count how many axes were reduced by fixed params
            axes_before_x = sum(1 for i, s in enumerate(slices[:x_axis]) if isinstance(s, int))
            axes_before_y = sum(1 for i, s in enumerate(slices[:y_axis]) if isinstance(s, int))
            
            new_x_axis = x_axis - axes_before_x
            new_y_axis = y_axis - axes_before_y
            
            # Ensure y_axis is first, x_axis is second
            if new_x_axis < new_y_axis:
                data_slice = np.moveaxis(data_slice, [new_x_axis, new_y_axis], [1, 0])
            else:
                data_slice = np.moveaxis(data_slice, [new_y_axis, new_x_axis], [0, 1])
            
            dict_out[field] = data_slice
        
        # Get parameter values
        x_vals = self.scan_params[x_param]
        y_vals = self.scan_params[y_param]
        
        # Create meshgrids
        x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='xy')
        
        # Store parameter information
        dict_out['x_param'] = x_param
        dict_out['y_param'] = y_param
        dict_out['x_vals'] = x_mesh
        dict_out['y_vals'] = y_mesh
        
        # For backward compatibility, also store with parameter names
        dict_out[x_param + 's'] = x_mesh
        dict_out[y_param + 's'] = y_mesh
        
        return dict_out
    
    def plot_contour_grid(self, fields_or_data: Union[List[str], Dict[str, np.ndarray]],
                         fields: Optional[List[str]] = None,
                         fixed_params: Optional[Dict[str, Any]] = None,
                         suptitle: str = '', method: str = 'contourf',
                         cmap: str = 'coolwarm', clim: Optional[Tuple[float, float]] = None,
                         deviation: bool = False, show_fig: bool = True,
                         figfilename: str = None, dpi: int = 300):
        """
        Plot contour grid for multiple fields on any parameter plane.
        
        This method can be called in two ways:
        1. With pre-extracted data: plot_contour_grid(data_dict, fields)
        2. Direct plotting: plot_contour_grid(fields, fixed_params=...)
        
        Parameters:
            fields_or_data: Either a list of field names OR a data dictionary from extract_field_data
            fields: List of field names (only needed if first arg is a data dict)
            fixed_params: Parameters to hold constant (only used if extracting data)
            suptitle: Super title for the figure
            method: Plot method ('contourf', 'imshow', 'scatter', 'pcolormesh')
            cmap: Colormap name
            clim: Color limits (vmin, vmax)
            deviation: If True, plot as percentage deviation from mean
            savefig: If True, save figure to file
            figfilename: Filename for saved figure
            dpi: Resolution of saved figure
        
        Examples:
            # Direct plotting (recommended)
            scan.plot_contour_grid(['Ti_core', 'Te_core'], 
                                  fixed_params={'energy_srcCORE': 1e6})
            
            # Two-step approach (for reusing data)
            data = scan.extract_field_data(['Ti_core'], fixed_params={'delta': 0.3})
            scan.plot_contour_grid(data, ['Ti_core'])
        """
        # Auto-detect calling style
        if isinstance(fields_or_data, list):
            # Called as plot_contour_grid(fields, fixed_params=...)
            field_list = fields_or_data
            data = self.extract_field_data(field_list, fixed_params=fixed_params)
        elif isinstance(fields_or_data, dict):
            # Called as plot_contour_grid(data, fields)
            data = fields_or_data
            if fields is None:
                raise ValueError("When passing a data dict, must provide 'fields' parameter")
            field_list = fields
        else:
            raise TypeError(f"First argument must be list of fields or data dict, got {type(fields_or_data)}")
        
        # Get parameter information from data
        x_param = data.get('x_param', 'delta')
        y_param = data.get('y_param', 'kappa')
        x_vals = data.get('x_vals', data.get('deltas'))
        y_vals = data.get('y_vals', data.get('kappas'))
        
        # Parameter labels for plotting
        param_labels = {
            'kappa': r'$\kappa$',
            'delta': r'$\delta$',
            'energy_srcCORE': r'Power [W] (log scale)',
            'nu': r'$\nu$',
            'beta': r'$\beta$'
        }
        
        xlabel = param_labels.get(x_param, x_param)
        ylabel = param_labels.get(y_param, y_param)
        
        ncols = min(2, len(field_list))
        n_plots = len(field_list)
        nrows = (n_plots + ncols - 1) // ncols
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
        axs = np.atleast_1d(axs).flatten()
        
        for idx, field in enumerate(field_list):
            ax = axs[idx]
            
            # Get field symbol
            fs = self.all_field_symbols.get(field, field)
            
            # Get data to plot
            if field not in data:
                print(f"Warning: Field '{field}' not found in data, skipping")
                ax.set_visible(False)
                continue
            
            toplot = data[field].copy()
            
            # Compute deviation if requested
            if deviation:
                ref_val = np.mean(toplot)
                toplot = (toplot - ref_val) / ref_val * 100
                label = r'$(' + fs + r' - v_0)/v_0$ [%]'
                plot_clim = (-50, 50)
            else:
                # Remove location terminology to get base field name for units and scaling
                base_field = self._base_field(field)
                unit = self.field_units.get(base_field, '')
                label = fs + ' ' + unit
                plot_clim = clim
                scale = self.field_scaling.get(base_field, 1.0)
                toplot *= scale
            
            # Create plot (use pcolormesh for log-scale axes as contourf doesn't work well)
            use_contourf = (x_param == 'energy_srcCORE' or y_param == 'energy_srcCORE')
            
            if method == 'contourf' or use_contourf:
                cf = ax.contourf(x_vals, y_vals, toplot, levels=20, cmap=cmap)
                if idx == 0:
                    ax.scatter(x_vals, y_vals, s=2, c='w', marker='o')
            elif method == 'imshow':
                extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]
                cf = ax.imshow(toplot, extent=extent, origin='lower',
                              aspect='auto', cmap=cmap)
            elif method == 'scatter':
                cf = ax.scatter(x_vals, y_vals, c=toplot, cmap=cmap, s=600, marker='s')
            else:  # pcolormesh (also used for power axes)
                cf = ax.pcolormesh(x_vals, y_vals, toplot, cmap=cmap, shading='auto')
            
            if plot_clim:
                cf.set_clim(plot_clim)
            
            plt.colorbar(cf, ax=ax, label=label)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Set log scale for power axis
            if x_param == 'energy_srcCORE':
                ax.set_xscale('log')
            else:
                ax.set_xticks(np.unique(x_vals))
                ax.tick_params(axis='x', rotation=45)
            
            if y_param == 'energy_srcCORE':
                ax.set_yscale('log')
            else:
                ax.set_yticks(np.unique(y_vals))
        
        # Hide unused subplots
        for idx in range(n_plots, len(axs)):
            axs[idx].set_visible(False)
        
        if suptitle:
            fig.suptitle(suptitle)
        
        plt.tight_layout()
        
        if figfilename is not None:
            fig.savefig(figfilename, dpi=dpi)
            print(f"Figure saved to {figfilename}")
        
        if show_fig:
            plt.show()
        else:
            plt.close(fig)

    def plot_field_vs_field(self, field_x: str, field_y: str,
                            powers: Optional[List[float]] = None,
                            alpha: float = 0.8,
                            cmap: str = 'coolwarm',
                            marker_size: Optional[float] = None,
                            figfilename: Optional[str] = None,
                            dpi: int = 300,
                            xlim = None,
                            ylim = None,
                            axis_equal: bool = False,
                            show_fig: bool = True,
                            annotate: bool = False,
                            lines: list = [],
                            shadows: list = []):
        """
        Scatter plot of one field against another for two or three power values.

        Each scan point (kappa, delta combination) is drawn once per power level:
          - First power value  → **ellipse** marker
          - Second power value → **rectangle** marker
          - Third power value  → **triangle** marker (if provided)

        The marker color encodes triangularity (delta, blue→red via ``cmap``) and
        the aspect ratio (height/width) encodes elongation (kappa / kappa_min)**2,
        so the minimum kappa gives a square/circle and larger kappa values give
        progressively taller markers.

        Parameters
        ----------
        field_x : str
            Field name for the x-axis (e.g. 'ne_core').
        field_y : str
            Field name for the y-axis (e.g. 'Te_core').
        powers : list of 1, 2, or 3 floats
            The ``energy_srcCORE`` values to compare.
            Defaults to first, middle, and last values in the scan (3 levels).
        alpha : float
            Marker transparency (default: 0.8).
        cmap : str
            Matplotlib colormap for delta encoding (default: 'coolwarm').
        marker_size : float, optional
            Marker width in data units. Auto-sized when None.
        figfilename : str, optional
            If provided the figure is saved to this path.
        dpi : int
            Resolution used when saving.
        xlim : tuple, optional
            The x-axis limits.
        ylim : tuple, optional
            The y-axis limits.
        axis_equal : bool
            If True, set equal aspect ratio for x and y axes.
        show_fig : bool
            If False the figure is closed instead of displayed.
        annotate : bool
            If True, annotate each marker with its (delta, kappa) values.
        lines : list of dicts
            Optional list of lines dict {'x': [x1, x2], 'y': [y1, y2], 'kwargs': {...}} to draw reference lines.
        shadows : list of dicts
            Option list of shadow dicts {'x': [x1, x2], 'y': [y1, y2], 'kwargs': {...}} to draw shaded regions.

        Examples
        --------
        scan.plot_field_vs_field('ne_core', 'Te_core', powers=[1e5, 1e6, 5e6])
        """
        from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon
        from matplotlib.colors import Normalize

        for f in [field_x, field_y]:
            if f not in self.data:
                raise ValueError(
                    f"Field '{f}' not found. Available: {sorted(self.data.keys())}"
                )

        # Default to first, middle, and last power values in the scan
        if powers is None:
            powers = self.scan_params.get('energy_srcCORE', [])

        if len(powers) > 4:
            raise ValueError(f"'powers' cannot contain more than 4 values, got {len(powers)}.")

        # Validate that the power values exist in the scan
        all_powers = self.scan_params.get('energy_srcCORE', [])
        for p in powers:
            if p not in all_powers:
                raise ValueError(f"Power value {p} not found in scan. Available: {all_powers}")

        # marker_styles = ['ellipse', 'rectangle', 'triangle_up', 'triangle_down']
        marker_styles = ['triangle_up', 'rectangle', 'hexagon', 'ellipse']

        # delta and kappa parameter arrays (free dimensions)
        delta_vals = np.array(self.scan_params['delta'])
        kappa_vals = np.array(self.scan_params['kappa'])
        kappa_min = kappa_vals.min()

        delta_norm = Normalize(vmin=delta_vals.min(), vmax=delta_vals.max())
        colormap = plt.get_cmap(cmap)

        # Collect all x/y values across all powers to determine axis limits
        all_x, all_y = [], []
        for p in powers:
            sl = self._get_slices(p)
            all_x.append(self.data[field_x][sl].flatten())
            all_y.append(self.data[field_y][sl].flatten())
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        x_range = np.ptp(all_x) if np.ptp(all_x) > 0 else 1.0
        y_range = np.ptp(all_y) if np.ptp(all_y) > 0 else 1.0
        
        if xlim is not None:
            x_range = xlim[1] - xlim[0]
        if ylim is not None:
            y_range = ylim[1] - ylim[0]

        if marker_size is None:
            n_delta = len(delta_vals)
            marker_size = x_range / (n_delta * 3.5)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.set_facecolor('white')
        ax.grid(False)

        free_keys = [k for k in self.scan_keys if k != 'energy_srcCORE']
        free_vals_list = [self.scan_params[k] for k in free_keys]
        free_combos = list(itertools.product(*free_vals_list))
        
        edgecolor = 'None'

        for p_idx, power in enumerate(powers):
            sl = self._get_slices(power)
            flat_x = self.data[field_x][sl].flatten()
            flat_y = self.data[field_y][sl].flatten()
            style = marker_styles[p_idx]

            for i, combo in enumerate(free_combos):
                combo_dict = dict(zip(free_keys, combo))
                kappa = combo_dict['kappa']
                delta = combo_dict['delta']

                color = colormap(delta_norm(delta))
                aspect = (kappa / kappa_min) ** 2
                w = marker_size
                h = marker_size * aspect * (y_range / x_range)
                cx, cy = flat_x[i], flat_y[i]

                if style == 'ellipse':
                    patch = Ellipse(
                        (cx, cy), width=w, height=h,
                        facecolor=color, edgecolor=edgecolor,
                        linewidth=0.5, alpha=alpha, zorder=3
                    )
                elif style == 'rectangle':
                    patch = FancyBboxPatch(
                        (cx - w / 2, cy - h / 2), width=w, height=h,
                        boxstyle='square,pad=0',
                        facecolor=color, edgecolor=edgecolor,
                        linewidth=0.5, alpha=alpha, zorder=3
                    )
                elif style == 'triangle_up':  # triangle (pointing up)
                    verts = np.array([
                        [cx - w / 2, cy - h / 2],
                        [cx + w / 2, cy - h / 2],
                        [cx,         cy + h / 2],
                    ])
                    patch = Polygon(
                        verts, closed=True,
                        facecolor=color, edgecolor=edgecolor,
                        linewidth=0.5, alpha=alpha, zorder=3
                    )
                elif style == 'triangle_down':  # triangle pointing down
                    verts = np.array([
                        [cx - w / 2, cy + h / 2],
                        [cx + w / 2, cy + h / 2],
                        [cx,         cy - h / 2],
                    ])
                    patch = Polygon(
                        verts, closed=True,
                        facecolor=color, edgecolor=edgecolor,
                        linewidth=0.5, alpha=alpha, zorder=3
                    )
                elif style == 'pentagon':  # pentagon marker
                    verts = np.array([
                        [cx,         cy + h / 2],
                        [cx + w / 2, cy + h / 4],
                        [cx + w / 3, cy - h / 2],
                        [cx - w / 3, cy - h / 2],
                        [cx - w / 2, cy + h / 4],
                    ])
                    patch = Polygon(
                        verts, closed=True,
                        facecolor=color, edgecolor=edgecolor,
                        linewidth=0.5, alpha=alpha, zorder=3
                    )
                elif style == 'hexagon':  # hexagon marker
                    verts = np.array([
                        [cx - w / 2, cy],
                        [cx - w / 4, cy + h / 2],
                        [cx + w / 4, cy + h / 2],
                        [cx + w / 2, cy],
                        [cx + w / 4, cy - h / 2],
                        [cx - w / 4, cy - h / 2],
                    ])
                    patch = Polygon(
                        verts, closed=True,
                        facecolor=color, edgecolor=edgecolor,
                        linewidth=0.5, alpha=alpha, zorder=3
                    )
                else:
                    raise ValueError(f"Unknown marker style: {style}")
                ax.add_patch(patch)

                if annotate:
                    ax.annotate(
                        f'{delta:.2f}/{kappa:.2f}',
                        (cx, cy), fontsize=6,
                        ha='center', va='center', zorder=4
                    )

        # Colorbar encoding delta
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=delta_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=r'$\delta$')

        # Legend: one entry per power level using standard markers
        legend_markers = ['^', 's', 'H', 'o']  # triangle_up, square, pentagon, hexagon
        legend_handles = [
            plt.Line2D([0], [0], marker=legend_markers[i], linestyle='None',
                       markerfacecolor='lightgray', markeredgecolor='k',
                       markersize=10, label=f'{p/1e6:.2f} MW')
            for i, p in enumerate(powers)
        ]
        ax.legend(handles=legend_handles, loc='best', fontsize=10)

        # Axis labels
        fx_sym = self.all_field_symbols.get(field_x, field_x)
        fy_sym = self.all_field_symbols.get(field_y, field_y)
        ax.set_xlabel(fx_sym + ' ' + self.field_units.get(self._base_field(field_x), ''))
        ax.set_ylabel(fy_sym + ' ' + self.field_units.get(self._base_field(field_y), ''))

        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(all_x.min() - x_range * 0.1, all_x.max() + x_range * 0.1)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(all_y.min() - y_range * 0.1, all_y.max() + y_range * 0.1)

        if axis_equal:
            ax.set_aspect('equal', adjustable='datalim')
            
        for line in lines:
            ax.plot(line['x'],line['y'], **line['kwargs'])
            
        for shadow in shadows:
            ax.fill_between(shadow['x'], shadow['y'][0], shadow['y'][1], **shadow['kwargs'])
        
        plt.tight_layout()

        if figfilename is not None:
            fig.savefig(figfilename, dpi=dpi)
            print(f"Figure saved to {figfilename}")

        if show_fig:
            plt.show()
        else:
            plt.close(fig)


    def setup_simulation(self, delta=None, kappa=None, energy_srcCORE=None, scanidx=None) -> Tuple:
        """
        Setup simulation paths and find available frames.
        
        Parameters:
            scanidx: Simulation index
        
        Returns:
            Tuple of (simulation, sim_frames, simdir, fileprefix)
        """
        if scanidx is None:
            if delta is None or kappa is None or energy_srcCORE is None:
                raise ValueError("Must provide either scanidx or all of delta, kappa, energy_srcCORE")
            params = {'delta': delta, 'kappa': kappa, 'energy_srcCORE': energy_srcCORE}
            scanidx = self.get_sim_index(params)
        # Use detected pattern or fall back to default
        simdir = f"{self.sim_prefix}/{self.sim_prefix}_{scanidx:05d}/"
        fileprefix = f"{self.sim_prefix}_{scanidx:05d}"
        
        try:
            simulation = pygkyl.load_sim_config('tcv_nt', simDir=simdir, filePrefix=fileprefix)
            simulation.geom_param.kappa = kappa
            simulation.geom_param.delta = delta
            simulation.geom_param.qaxis = 1.2
            simulation.geom_param.qlcfs = 2.6
            simulation.geom_param.qprofile_R = 'quadratic'
            simulation.geom_param.update_geom_params()
            fieldname = 'field'
            sim_frames = pygkyl.file_utils.find_available_frames(simulation, fieldname)
            return simulation, sim_frames, simdir, fileprefix
        except Exception as e:
            raise RuntimeError(f"Failed to setup simulation {scanidx}: {e}")
        
    def plot_2D(self, delta=None, kappa=None, energy_srcCORE=None, field='phi', frame_idx=500,
                cut_coords: Optional[List] = None, cut_dir: str = 'xy', **kwargs):
        simulation, frames, _, _ = self.setup_simulation(delta=delta, kappa=kappa,
                                           energy_srcCORE=energy_srcCORE)
        frame_idx = np.argmin(np.abs(np.array(frames) - frame_idx))
        simulation.plot_2D(field_name=field,cut_coords=cut_coords,cut_dir=cut_dir,
                           frame_idx=frame_idx,**kwargs)
        
    def plot_1D(self, delta=None, kappa=None, energy_srcCORE=None, field='phi', frame_indices=[500],
                cut_dir='x', cut_coords=[0.0,0.0], space_time=False, **kwargs):
        simulation, frames, _, _ = self.setup_simulation(delta=delta, kappa=kappa,
                                           energy_srcCORE=energy_srcCORE)
        simulation.plot_1D_time_evolution(cut_dir=cut_dir, 
                                  cut_coords=cut_coords,
                                  field_name=field,
                                  frame_indices=frame_indices,
                                  space_time=space_time, **kwargs)
        
    def plot_poloidal_projection(self, delta=None, kappa=None, energy_srcCORE=None, 
                                 field_name='phi', frame_idx=None, out_file_name='',
                                 nzInterp=32, colorMap='inferno', colorScale='lin',
                                 showInset=True, showLCFS=True, xlim=[], ylim=[], clim=[],
                                 logScaleFloor=1e-3, figout=[], close_fig=False,
                                 fig_dpi=300, showAxis=True, cutoutLimiter=False):
        simulation, frames, _, _ = self.setup_simulation(delta=delta, kappa=kappa,
                                           energy_srcCORE=energy_srcCORE)
        if frame_idx is None:
            frame_idx = len(frames) - 1
        simulation.plot_poloidal_projection(field_name=field_name,
            frame_idx=frame_idx, out_file_name=out_file_name,
            nzInterp=nzInterp, colorMap=colorMap, colorScale=colorScale,
            showInset=showInset, showLCFS=showLCFS, xlim=xlim, ylim=ylim, clim=clim,
            logScaleFloor=logScaleFloor,figout=figout, close_fig=close_fig,
            fig_dpi=fig_dpi, showAxis=showAxis, cutoutLimiter=cutoutLimiter)
        
    def get_volume_integral(self, delta=None, kappa=None, energy_srcCORE=None, 
                            fieldName='WkinM2', frame_idx=500, jacob_squared=False, average=False,
                            integral_bounds =[None, None, None]):
        """
        Compute the volume integral of a given field at a specific time frame.
        """
        simulation, frames, _, _ = self.setup_simulation(delta=delta, kappa=kappa,
                                           energy_srcCORE=energy_srcCORE)
        frame = simulation.get_frame(fieldName, frame_idx, load=True)
        return frame.compute_volume_integral(jacob_squared=jacob_squared, average=average,
                                             integral_bounds=integral_bounds)
        
    def get_profile(self, params: Dict[str, Any], field: str,
                   frame_idx: int = 500, cut_coords: Optional[List] = None) -> Tuple:
        """
        Get 1D profile from a simulation.
        
        Parameters:
            params: Dictionary with scan parameters
            field: Field name (pygkyl format, e.g., 'phi', 'Ti')
            frame_idx: Frame index to extract
            cut_coords: Cut coordinates for pygkyl (default: ['avg', 0.0])
        
        Returns:
            Tuple of (xdata, ydata, label, scanidx)
        """
        if cut_coords is None:
            cut_coords = ['avg', 0.0]
        
        figout = []
        scanidx = self.get_sim_index(params)
        
        simulation, sim_frames, _, _ = self.setup_simulation(scanidx=scanidx)
        simulation.plot_1D_time_evolution('x', cut_coords, field,
                                         sim_frames[frame_idx],
                                         close_fig=True, figout=figout)
        
        fig = figout[0]
        ax = fig.axes[0]
        lines = ax.get_lines()
        xdata = lines[0].get_xdata()
        ydata = lines[0].get_ydata()
        
        # Create descriptive label
        label_parts = [f"{k}={v}" for k, v in params.items()]
        label = ", ".join(label_parts)
        
        return xdata, ydata, label, scanidx
    
    def compare_profiles(self, vary_param: str,
                        vary_vals: List[Any],
                        fixed_params: Optional[Dict[str, Any]] = None,
                        field: str = 'Ti',
                        frame_idx: int = 500,
                        cut_coords: Optional[List] = None,
                        figname: Optional[str] = None,
                        cmap: str = 'viridis'):
        """
        Compare profiles across a parameter scan (varying 1 parameter).
        
        Parameters:
            vary_param: Name of parameter to vary (e.g., 'kappa', 'energy_srcCORE')
            vary_vals: Values of varying parameter
            fixed_params: Dict with fixed parameters (e.g., {'delta': 0.3, 'energy_srcCORE': 1e5})
            field: Field name to plot (pygkyl format)
            frame_idx: Frame index
            cut_coords: Cut coordinates (default: ['avg', 0.0])
            figname: If provided, save figure to this filename
            cmap: Colormap for parameter variation
        
        Examples:
            # Compare kappa at fixed delta and power
            scan.compare_profiles('kappa', [1.1, 1.2, 1.3],
                                 fixed_params={'delta': 0.3, 'energy_srcCORE': 1e6})
            
            # Compare power at fixed kappa and delta
            scan.compare_profiles('energy_srcCORE', [5e5, 1e6, 1.5e6],
                                 fixed_params={'kappa': 1.5, 'delta': 0.3})
        """
        if cut_coords is None:
            cut_coords = ['avg', 0.0]
        
        if fixed_params is None:
            fixed_params = {}
        
        vmax = np.max(vary_vals)
        vmin = np.min(vary_vals)
        
        # Parameter labels
        param_labels = {
            'kappa': r'\kappa',
            'delta': r'\delta',
            'energy_srcCORE': r'P',
            'nu': r'\nu',
            'beta': r'\beta'
        }
        param_label = param_labels.get(vary_param, vary_param)
        
        fig, ax = plt.subplots(figsize=(5, 3.5))
        
        for val in vary_vals:
            sim = {**fixed_params, vary_param: val}
            field_pgkyl = field.replace('P', 'p')
            x, y, _, scanidx = self.get_profile(sim, field_pgkyl,
                                               frame_idx=frame_idx,
                                               cut_coords=cut_coords)
            
            # Color from colormap
            norm_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            clr = plt.get_cmap(cmap)(norm_val)
            ax.plot(x, y, label=f"{scanidx}", color=clr)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=r'$' + param_label + r'$')
        
        ax.set_xlabel(r'$r/a$')
        
        # Y-label with averaging notation
        base_field = self._base_field(field)
        ylabel = self.field_symbols.get(base_field, base_field)
        
        if cut_coords[0] == 'avg' and cut_coords[1] == 0.0:
            ylabel = r'\langle ' + ylabel + r'\rangle_y'
        elif cut_coords[0] == 'avg' and cut_coords[1] == 'avg':
            ylabel = r'\langle ' + ylabel + r'\rangle_{y,z}'
        
        unit = self.field_units.get(base_field, '')
        ax.set_ylabel(r'$' + ylabel + r'$ ' + unit)
        
        plt.tight_layout()
        
        if figname is not None:
            fig.savefig(figname, dpi=300)
            print(f"Figure saved to {figname}")
        
        plt.show()