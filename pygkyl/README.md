# PyGkyl

A Python package for handling and analyzing Gkeyll simulation data. This package provides classes and tools for 
setting up simulations, processing data, and visualizing results.

## Installation

You can install the library using pip:

```sh
pip install /path/to/your/python_utilities
```

Replace `/path/to/your/python_utilities` with the actual path to your `python_utilities` directory.

## Requirements

- numpy
- scipy
- matplotlib

## Usage

Here is an example of how to use the library:

```python
from pygkyl import Simulation, Species, file_utils

# Use the imported classes and functions
simulation = Simulation(dimensionality='3x2v')
# ...rest of your code...
```

Make sure to replace `some_utility_function` with the actual function you want to use from your `utils` module.

## File Summaries

### /pygkyl/utils/plot_utils.py

This module provides various plotting utilities for visualizing Gkeyll simulation data.

### /pygkyl/classes/simulation.py

Manages the setup, parameters, and data for a plasma simulation.

### /pygkyl/classes/frame.py

Manages the loading, slicing, and normalization of simulation data frames.

### /pygkyl/classes/dataparam.py

Manages the setup and configuration of simulation data directories, file prefixes, and data fields.

### /pygkyl/__init__.py

Initializes the pygkyl package and imports core modules and utilities.

## Author

Your Name - [your.email@example.com](mailto:your.email@example.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.