import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import time
from ..classes import Frame, Simulation

class GyrazeAttribute:
    def __init__(self, fieldname, label, units, manual=False):
        self.fieldname = fieldname
        self.label = label
        self.units = units
        self.manual = manual
        self.frame = None
        self.v0 = None
        self.tf = None
        
        if fieldname in ['fe', 'fi']:
            self.eval = self.eval5d
            self.filter_negativity = self.filter_negativity_active
        else:
            self.eval = self.eval3d
            self.filter_negativity = self.filter_negativity_disabled
            
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
        
    def check_negativity(self):
        if self.fieldname in ['Te', 'Ti', 'ne', 'ni']:
            if np.any(self.v0 < 0):
                return True
        return False
    
    def filter_negativity_active(self):
        self.v0[self.v0 < 0] = 0.0
        
    def filter_negativity_disabled(self):
        pass
                

class GyrazeDataset:
    def __init__(self):
        self.attributes = {}
        self.attributes['B'] = GyrazeAttribute('Bmag', r'$B$', 'T')
        self.attributes['phi'] = GyrazeAttribute('phi', r'$\phi$', 'V')
        self.attributes['ne'] = GyrazeAttribute('ne', r'$n_e$', 'm$^{-3}$')
        self.attributes['ni'] = GyrazeAttribute('ni', r'$n_i$', 'm$^{-3}$')
        self.attributes['Te'] = GyrazeAttribute('Te', r'$T_e$', 'eV')
        self.attributes['Ti'] = GyrazeAttribute('Ti', r'$T_i$', 'eV')
        self.attributes['gamma'] = GyrazeAttribute('rhoe_lambdaD', r'$\rho_e/\lambda_D$', '')
        self.attributes['fe'] = GyrazeAttribute('fe', r'$f_e$', 'm$^{-6}$s$^3$')
        self.attributes['fi'] = GyrazeAttribute('fi', r'$f_i$', 'm$^{-6}$s$^3$')
        self.attributes['nioverne'] = GyrazeAttribute('nioverne', r'$n_i/n_e$', '', manual=True)
        self.attributes['TioverTe'] = GyrazeAttribute('TioverTe', r'$T_i/T_e$', '', manual=True)
        self.attributes['vpare_norm'] = GyrazeAttribute('vpare_norm', r'$v_{\parallel e}/v_{the}$', '', manual=True)
        self.attributes['mue_norm'] = GyrazeAttribute('mue_norm', r'$\mu_e B_0/T_{e0}$', '', manual=True)
        self.attributes['vpari_norm'] = GyrazeAttribute('vpari_norm', r'$v_{\parallel i}/v_{thi}$', '', manual=True)
        self.attributes['mui_norm'] = GyrazeAttribute('mui_norm', r'$\mu_i B_0/T_{i0}$', '', manual=True)
        self.grids = {}

    def load(self, simulation, tf):
        for attr in self.attributes.values():
            attr.load(simulation, tf)
        self.grids['x'] = self.attributes['B'].frame.new_grids[0]
        self.grids['y'] = self.attributes['B'].frame.new_grids[1]
        self.grids['z'] = self.attributes['B'].frame.new_grids[2]
        self.grids['vpare'] = self.attributes['fe'].frame.new_grids[3]
        self.grids['mue'] = self.attributes['fe'].frame.new_grids[4]
        self.grids['vpari'] = self.attributes['fi'].frame.new_grids[3]
        self.grids['mui'] = self.attributes['fi'].frame.new_grids[4]

    def eval(self, x, y, z):
        for attr in self.attributes.values():
            attr.eval(x, y, z)
        self.attributes['nioverne'].v0 = self.attributes['ni'].v0 / self.attributes['ne'].v0
        self.attributes['TioverTe'].v0 = self.attributes['Ti'].v0 / self.attributes['Te'].v0
        vthe = np.sqrt(self.attributes['Te'].v0 * 1.602e-19)  # electron thermal speed (Te in eV)
        vthi = np.sqrt(self.attributes['Ti'].v0 * 1.602e-19)  # ion thermal speed (Ti in eV)
        mu0e = self.attributes['Te'].v0 * 1.602e-19 / self.attributes['B'].v0  # electron thermal mu (Te in eV)
        mu0i = self.attributes['Ti'].v0 * 1.602e-19 / self.attributes['B'].v0  # ion thermal mu (Ti in eV)
        self.attributes['vpare_norm'].v0 = self.grids['vpare'] / vthe
        self.attributes['mue_norm'].v0 = self.grids['mue'] / mu0e
        self.attributes['vpari_norm'].v0 = self.grids['vpari'] / vthi
        self.attributes['mui_norm'].v0 = self.grids['mui'] / mu0i

    def check_negativity(self):
        for attr in self.attributes.values():
            if attr.check_negativity():
                return True
        return False
    
    def filter_negativity(self):
        for attr in self.attributes.values():
            attr.filter_negativity()
            

class GyrazeInterface:
    def __init__(self, simulation:Simulation, **kwargs):
        self.simulation = simulation
        self.alphadeg : float = kwargs.get('alphadeg', 0.3)
        self.filter_negativity : bool = kwargs.get('filter_negativity', False)
        self.number_datasets : bool = kwargs.get('number_datasets', False)
        self.outfilename : str = kwargs.get('outfilename', 'data.h5')

        self.frames = self.simulation.available_frames['ion']
        self.nspec = len(self.simulation.species)
        self.me = self.simulation.species['elc'].m
        self.mi = self.simulation.species['ion'].m
        self.mioverme = self.mi/self.me
        self.e = np.abs(self.simulation.species['elc'].q)
        self.dataset = GyrazeDataset()

        self.fe_mpe_args_text = None
        self.fe_mpe_text = None
        self.fi_mpi_args_text = None
        self.fi_mpi_text = None
        self.input_physparams_text = None
        self.skip_point = False
        self.nsample = None
        self.nskipped = None
        self.required_attrs = [
            'x0', 'y0', 'z0', 'alphadeg', 'tf', 'B0', 'phi0',
            'ne0', 'ni0', 'Te0', 'Ti0', 'gamma0', 'nioverne',
            'TioverTe', 'mioverme', 'mi', 'me', 'e', 'simprefix'
        ]
        self.required_datasets = [
            'Fe_mpe_args.txt', 'Fe_mpe.txt', 
            'Fi_mpi_args.txt', 'Fi_mpi.txt', 
            'input_physparams.txt'
        ]
        self.text_files = [
            'Fe_mpe_args.txt', 'Fe_mpe.txt', 'Fi_mpi_args.txt', 
            'Fi_mpi.txt', 'input_physparams.txt'
        ]


    def load_frames(self, timeframe):
        self.dataset.load(self.simulation, timeframe)

    def eval_frames(self, x, y, z):
        self.dataset.eval(x, y, z)

        if self.dataset.check_negativity():
            self.skip_point = True
            return
        
        if self.filter_negativity:
            self.dataset.filter_negativity()

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

    def generate_F_mps_content(self, munorm, vpnorm, f0):
        
        # Generate args content
        args_content = ' '.join(map(str, munorm)) + '\n' + ' '.join(map(str, vpnorm))
        
        # Generate f0 content using StringIO to mimic savetxt behavior
        f0_buffer = io.StringIO()
        np.savetxt(f0_buffer, f0.squeeze().T, fmt='%.16e')
        f0_content = f0_buffer.getvalue()
        
        return args_content, f0_content

    def generate_input_physparams_content(self):
        content = (
            '#set type_distfunc_entrance (= ADHOC or other string)\n'
            'GKEYLL\n'
            '#set alphadeg\n'
            f'{self.alphadeg}\n'
            '#set gamma_ref (keep zero to solve only magnetic presheath)\n'
            f'{self.dataset.attributes['gamma'].v0}\n'
            '#set nspec\n'
            f'{self.nspec}\n'
            '#set nioverne\n'
            f'{self.dataset.attributes['nioverne'].v0}\n'
            '#set TioverTe\n'
            f'{self.dataset.attributes['TioverTe'].v0}\n'
            '#set mioverme\n'
            f'{self.mioverme}\n'
            '#set set_current (flag)\n'
            '0\n'
            '#set target_current or phi_wall\n'
            f'{self.dataset.attributes['phi'].v0}\n'
        )
        return content
    
    def generate_input_files(self):
        self.fe_mpe_args_text, self.fe_mpe_text = self.generate_F_mps_content(
            self.dataset.attributes['mue_norm'].v0,
            self.dataset.attributes['vpare_norm'].v0, 
            self.dataset.attributes['fe'].v0)
        self.fi_mpi_args_text, self.fi_mpi_text = self.generate_F_mps_content(
            self.dataset.attributes['mui_norm'].v0, 
            self.dataset.attributes['vpari_norm'].v0, 
            self.dataset.attributes['fi'].v0)
        self.input_physparams_text = self.generate_input_physparams_content()
        
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
        grp.attrs['mi'] = self.mi
        grp.attrs['me'] = self.me
        grp.attrs['e'] = self.e
        grp.attrs['mioverme'] = self.mioverme
        grp.attrs['simprefix'] = self.simulation.data_param.fileprefix
        for attr_name, attr in self.dataset.attributes.items():
            if np.isscalar(attr.v0):
                ext0 = '' if attr.fieldname in ['TioverTe', 'mioverme', 'nioverne'] else '0'
                grp.attrs[attr_name + ext0] = attr.v0

    def generate(self,tf,xmin,xmax,Nxsample,Nysample,alphadeg=None,zplane='both',verbose=False):
        if alphadeg is not None:
            self.alphadeg = alphadeg
            
        self.load_frames(tf)
        xindices, yindices, izplanes = self.get_ranges(xmin, xmax, Nxsample, Nysample, zplane)

        with h5py.File(self.outfilename, 'w') as hf:
            hf.attrs['description'] = 'Gyraze input data files from Gkeyll simulation'
            self.nsample = 0
            self.nskipped = 0
            # Sample points in the (x,y) plane
            for ix in xindices:
                for iy in yindices:
                    for izplane in izplanes:
                        x0 = self.dataset.grids['x'][ix]
                        y0 = self.dataset.grids['y'][iy]
                        z0 = self.dataset.grids['z'][izplane]
                        if verbose: print(f'ix={ix}, x0={x0:.3f}, iy={iy}, y0={y0:.3f}, iz={izplane}, z0={z0:.3f}')

                        self.eval_frames(x0, y0, z0)

                        if self.skip_point:
                            if verbose: print(f'Skipping point due to negativity in Ti, Te, ni, or ne')
                            self.skip_point = False
                            self.nskipped += 1
                            continue                    

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
            return False
            
        try:
            with h5py.File(self.outfilename, 'r') as hf:
                if group_name not in hf:
                    print(f"ERROR: Group {group_name} not found in file")
                    return False
                
                grp = hf[group_name]
                
                if output_dir is None:
                    output_dir = f'gyraze_data_{group_name}'
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Extract each text file
                for filename in self.text_files:
                    if filename in grp:
                        content = grp[filename][()].decode('utf-8')
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, 'w') as f:
                            f.write(content)
                        print(f"Extracted {filepath}")
                    else:
                        print(f"WARNING: {filename} not found in group {group_name}")
                
                # Also write a metadata file
                metadata_file = os.path.join(output_dir, 'metadata.txt')
                with open(metadata_file, 'w') as f:
                    f.write(f"Dataset: {group_name}\n")
                    for attr_name in grp.attrs:
                        f.write(f"{attr_name}: {grp.attrs[attr_name]}\n")
                print(f"Extracted {metadata_file}")
                
                return True
                
        except Exception as e:
            print(f"ERROR: Failed to extract dataset: {e}")
            return False


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
                    for dataset_name in self.required_datasets:
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
            fi_args_h5 = grp['Fi_mpi_args.txt'][()].decode('utf-8')
            if fi_args_h5.strip() != self.fi_mpi_args_text.strip():
                print(f"    ERROR: Fi_mpi_args.txt content mismatch")
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
        