import pygkyl
import itertools
import re
from multiprocessing import Pool, cpu_count
import os
import argparse

scan_arrays={
    "kappa": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    "delta": [-0.6, -0.45, -0.3, -0.15, 0.15, 0.3, 0.45, 0.6],
    "energy_srcCORE": [0.1e6, 0.5e6, 1.0e6, 5.0e6],
}
scandir = 'tcv_miller_scan_big'
# scandir = 'tcv_miller_scan_18x12x12x10x6'
scanname = scandir
frame_idx = 450 # e.g. we take the 100th frame (t=200mus), set to -1 to take the last frame
frame_navg = 10 # number of frames we average the data on.
# Define locations and fields to extract
locations = {'lcfs': 0.04, 'core': 0.02, 'sol': 0.06}
fields = ['Ti', 'Te', 'ne', 'phi']
filter_dict = {
    'Ti': {'min': 0.0, 'max': 2000.0},
    'Te': {'min': 0.0, 'max': 2000.0},
    'ne': {'min': 0.0, 'max': 1e25},
    'phi': {'min': -2000.0, 'max': 2000.0},
}
# ==================== end of user input ====================

def setup_simulation(simdir, fileprefix):
    """Setup simulation with normalization settings."""
    simulation = pygkyl.load_sim_config(configName='tcv_nt', simDir=simdir, filePrefix=fileprefix)
    norm_settings = {
        't': 'mus', 'x': 'minor radius', 'y': 'Larmor radius', 'z': 'pi',
        'fluid velocities': 'thermal velocity', 'temperatures': 'eV',
        'pressures': 'Pa', 'energies': 'MJ', 'current': 'kA',
        'gradients': 'major radius'
    }
    for key, value in norm_settings.items():
        simulation.normalization.set(key, value)
    return simulation

def extract_field_values(simulation, frame_array, locations, fields):
    """Extract average field values at specified locations."""
    results = {f'{field}_{loc}': 0.0 for field in fields for loc in locations}
    ntake = 0
    
    for fidx in frame_array:
        frames = {field: simulation.get_frame(field, fidx) for field in fields}
        ny = frames[fields[0]].values.shape[1]
        
        for iy in range(ny):
            for field in fields:
                for loc_name, x_val in locations.items():
                    key = f'{field}_{loc_name}'
                    results[key] += frames[field].get_value([x_val, iy, 0.0])
            ntake += 1
    
    # Average all values
    for key in results:
        results[key] /= ntake
        
    # Apply filtering
    for field in fields:
        vmin = filter_dict[field]['min']
        vmax = filter_dict[field]['max']
        for loc_name in locations:
            key = f'{field}_{loc_name}'
            if results[key] < vmin or results[key] > vmax:
                results[key] = float('nan')
                    
    return results, frames[fields[0]].time

def get_average_dt(logfile):
    """Calculate average dt from log file."""
    dt_values = []
    try:
        with open(logfile, 'r') as f:
            for line in f:
                if 'dt = ' in line:
                    match = re.search(r'dt = ([\d.eE+-]+)', line)
                    if match:
                        dt_values.append(float(match.group(1)))
        return sum(dt_values) / len(dt_values) if dt_values else 0.0
    except FileNotFoundError:
        return 0.0

filename = f'{scandir}_metadata'
if frame_idx >= 0:
    filename += f'_frame_{frame_idx}'
else:
    filename += '_lastframes'
    
if frame_navg > 1:
    filename += f'_navg_{frame_navg}'
    
values = list(scan_arrays.values())
combinations = list(itertools.product(*values))
nscan = len(combinations)

def process_scan(scanidx):
    """Process a single scan index - designed for parallel execution."""
    simdir = f'{scandir}/{scanname}_{scanidx:05d}/'
    fileprefix = f'{scanname}_{scanidx:05d}'
    
    try:
        simulation = setup_simulation(simdir, fileprefix)
        
        sim_frames = pygkyl.file_utils.find_available_frames(simulation, 'field')
        last_frame = frame_idx if frame_idx >= 0 else sim_frames[-1]
        first_frame = last_frame - frame_navg
        frame_array = list(range(first_frame, last_frame + 1))
        
        field_values, tend = extract_field_values(simulation, frame_array, locations, fields)
        avg_dt = get_average_dt(f'{scandir}/std-{scanname}_{scanidx:05d}.log')
        
        # Gather data
        data = {
            'simdir': simdir,
            'scanidx': scanidx,
            'kappa': combinations[scanidx][0],
            'delta': combinations[scanidx][1],
            'energy_srcCORE': combinations[scanidx][-1],
            'tend': tend,
            'avg_dt': avg_dt,
            **field_values  # Unpack all field values
        }
        
        print("%d/%d: k=%.2f, d=%.2f, P=%.1e, Ti=%.1f, Te=%.1f, ne=%.1e, phi=%.1f, t=%.1f, dt=%.5f" % 
              (scanidx+1, nscan, data['kappa'], data['delta'], data['energy_srcCORE'], 
               field_values['Ti_lcfs'], field_values['Te_lcfs'], field_values['ne_lcfs'], 
               field_values['phi_lcfs'], tend, avg_dt))
        
        return data
    except Exception as e:
        print(f"Error processing scan {scanidx}: {e}")
        return None

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process simulation metadata in parallel')
parser.add_argument('-ncpu', type=int, default=cpu_count(), 
                    help=f'Number of CPU cores to use (default: {cpu_count()}, all available)')
args = parser.parse_args()

# Determine number of parallel processes
num_processes = args.ncpu
print(f"Processing {nscan} simulations using {num_processes} parallel processes...")

# Run parallel processing
if __name__ == '__main__':
    with Pool(processes=num_processes) as pool:
        metadata = pool.map(process_scan, range(nscan))
    
    # Filter out any None results from failed scans
    metadata = [m for m in metadata if m is not None]
    
    # Save the metadata to a json file
    import json
    with open(filename+'.json', 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {filename+'.json'}")
    print(f"Successfully processed {len(metadata)}/{nscan} simulations")