# GyrazeInterface HDF5 File Structure

## Overview
The `GyrazeInterface` class converts Gkeyll simulation data into the format required by the GYRAZE code, storing multiple datasets in a single HDF5 file.

## HDF5 File Structure

```
gkeyll_gyraze_inputs.h5
├── attributes:
│   ├── description: "Gyraze input data files from Gkeyll simulation"
│   └── nsample: <number of datasets>
│
├── Group: "x_1.234_y_5.678_z_0.000_alpha_0.300_tf_50" (or "000001" if numbered)
│   ├── datasets:
│   │   ├── Fe_mpe_args.txt    # Electron velocity/mu grid parameters
│   │   ├── Fe_mpe.txt         # Electron distribution function
│   │   ├── Fi_mpi_args.txt    # Ion velocity/mu grid parameters  
│   │   ├── Fi_mpi.txt         # Ion distribution function
│   │   └── input_physparams.txt # Physical parameters for GYRAZE
│   │
│   └── attributes:
│       ├── x0, y0, z0: spatial coordinates
│       ├── tf: timeframe
│       ├── alphadeg: field line angle
│       ├── B0, phi0: magnetic field and potential
│       ├── ne0, ni0: electron and ion densities
│       ├── Te0, Ti0: electron and ion temperatures
│       ├── gamma0: rhoe_lambdaD parameter
│       ├── nioverne, TioverTe: density and temperature ratios
│       ├── mioverme, mi, me, e: mass and charge constants
│       └── simprefix: simulation identifier
│
├── Group: "x_2.345_y_6.789_z_1.000_alpha_0.300_tf_50"
│   └── ... (same structure)
│
└── ... (additional groups for each sampled point)
```

## Usage

### Creating an HDF5 file
```python
from pygkyl.interfaces import GyrazeInterface

# Initialize interface
interface = GyrazeInterface(simulation, outfilename='gyraze_data.h5')

# Generate datasets
interface.generate(tf=50, xmin=-2.0, xmax=2.0, Nxsample=10, Nysample=5, 
                  alphadeg=0.3, zplane='both', verbose=True)
```

### Verifying data
```python
# Basic verification
interface.verify_h5_data(verbose=True)

# Detailed verification with random sampling
interface.verify_h5_data(verbose=True, Nsamp=5)
```

### Extracting data
```python
# Extract specific dataset to files
interface.extract_dataset_as_files('000001', 'output_dir/')
```