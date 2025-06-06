



https://github.com/user-attachments/assets/2e82e4ee-6cf7-4699-8920-67bafa266ec3


_Gkeyll TCV turbulence simulation in negative triangularity configuration._

_(see_ `scripts/pygkyl_plottorus.py -h`_)_

# personal_gkyl_scripts

A collection of personal scripts, tools, and Python modules for working with [Gkeyll](https://gkeyll.readthedocs.io/) plasma simulations. This repository is intended to streamline simulation workflows, automate analysis, and provide custom utilities for Gkeyll users.

## Repository Structure

- `pygkyl/` — Python library for loading, analyzing, and visualizing Gkeyll simulation data.
- `scripts/` — Standalone scripts for automation, data processing, and visualization.
- `studies/` — Jupyter notebooks demonstrating analysis and visualization workflows.

## Features

- Automated data loading and normalization for Gkeyll outputs
- Advanced visualization tools, including toroidal projections
- Utilities for extracting and analyzing physical quantities
- Example scripts and notebooks for common analysis tasks

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/personal_gkyl_scripts.git
   cd personal_gkyl_scripts
   ```

2. (Optional) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Use the install script for `pygkyl`:
   ```bash
   python pygkyl/scripts/pygkyl_install.py -p PATH
   ```

## Usage

- Explore the `pygkyl` library for programmatic analysis and visualization.
- Refer to `pygkyl/example.ipynb` for examples of `pygkyl` usage.
- Use scripts in `simulation_scripts` for batch processing or automation.

## Requirements

- Python 3.1x (mostly for postgkyl)
- numpy, matplotlib, and other scientific Python libraries (see `requirements.txt`)

## License

MIT License. See [LICENSE](LICENSE) for details.
