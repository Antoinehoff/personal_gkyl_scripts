# PyGkyl

`pygkyl` is a Python library for loading, analyzing, and visualizing simulation data from gyrokinetic and fluid plasma simulations, especially those produced by the Gkeyll code. It provides tools for plasma physicists and computational scientists to efficiently process high-dimensional simulation outputs. It is mostly made to vizualize results in Jupyter Notebooks.

## Key Features
- **Data Loading:** Import simulation data from Gkeyll and related codes, supporting various file formats and directory structures.
- **Normalization:** Convert simulation units to physical units (e.g., time in microseconds, temperature in eV, pressure in Pascal) using flexible normalization utilities.
- **Visualization:** Advanced plotting and projection tools, including toroidal projections for tokamak geometry, movie generation, and support for custom camera paths and lighting.
- **Analysis:** Extract and analyze physical quantities such as density, temperature, pressure, and energy, with support for fluctuations and averaging.
- **Extensibility:** Modular design allows users to add custom analysis routines and visualizations.

## Example Usage
A typical workflow using `pygkyl`:

```python
import pygkyl

# Set up simulation configuration
simulation = pygkyl.simulation_configs.import_config('tcv_nt', simdir, fileprefix)

# Set normalization for physical units
simulation.normalization.set('t','mus')
simulation.normalization.set('temperatures','eV')
# ...

# Set up toroidal projection and plot
torproj = pygkyl.TorusProjection()
torproj.setup(simulation, Nint_polproj=32, Nint_fsproj=24)
torproj.plot(fieldName='Ti', timeFrame=0)
```

## Applications
- Analysis of gyrokinetic and fluid plasma simulations
- Visualization of 3D toroidal geometries (e.g., tokamaks)
- Generation of movies and publication-quality figures

## Requirements
- Python 3.x
- numpy, matplotlib, and other scientific Python libraries

## Getting Started
To install and use `pygkyl`, clone this repository. Then open `/notebooks/
You can also use the `pygkyl/scripts/pygkyl_install.py -p PATH` script to have an automatic installation at `PATH`.

## License
This project is licensed under the MIT License.
