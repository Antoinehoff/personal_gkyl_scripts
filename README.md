
![screenshot](https://github.com/user-attachments/assets/995a9ad0-c647-4266-af4b-7d54395d4897)

# personal_gkyl_scripts

A collection of personal scripts, tools, and Python modules for working with [Gkeyll](https://gkeyll.readthedocs.io/) plasma simulations. This repository is intended to streamline simulation workflows, automate analysis, and provide custom utilities for Gkeyll users.

## Repository Structure

- `pygkyl/` — Python library for loading, analyzing, and visualizing Gkeyll simulation data.
- `scripts/` — Standalone scripts for automation, data processing, and visualization.
- `notebooks/` — Example Jupyter notebooks demonstrating analysis and visualization workflows.
- `tests/` — Unit tests and validation scripts.

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
- Use scripts in the `scripts/` directory for batch processing or automation.
- Refer to `notebooks/` for interactive examples.

## Requirements

- Python 3.x
- numpy, matplotlib, and other scientific Python libraries (see `requirements.txt`)

## License

MIT License. See [LICENSE](LICENSE) for details.
