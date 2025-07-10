# Gkyl Parameter Configuration Interface

This is a web-based user interface for configuring parameters in your Gkyl C input files. The interface allows you to visualize, modify, and generate C code with your specified parameters while keeping the rest of the file structure intact.

## Features

- **Interactive Parameter Editing**: Easy-to-use input fields for all configurable parameters
- **Tooltips**: Hover over parameter names to see detailed descriptions
- **Real-time Preview**: See the generated C code update as you modify parameters
- **File Generation**: Download the complete input.c file with your parameters
- **Parameter Validation**: Built-in validation for numeric inputs and constraints
- **Reset Functionality**: Quickly reset all parameters to default values
- **Responsive Design**: Works on desktop and mobile devices

## Parameter Categories

The interface organizes parameters into logical sections:

### 1. TCV #65130 (NT) Discharge Parameters
- `P_exp`: P_sol measured power [W]
- `a_shift`: Parameter in Shafranov shift
- `kappa`: Elongation (=1 for no elongation)
- `delta`: Triangularity (=0 for no triangularity)
- `R_axis`: Magnetic axis major radius [m]
- `R_LCFSmid`: Major radius of the LCFS at the outboard midplane [m]
- `Z_axis`: Magnetic axis height [m]
- `B_axis`: Magnetic field at the magnetic axis [T]

### 2. Reference Values
- `Te0`: Reference electron temperature [eV]
- `Ti0`: Reference ion temperature [eV]
- `n0`: Reference density [1/m³]
- `Bref`: Reference magnetic field [T]

### 3. Numerical Resolutions
- `num_cell_x`, `num_cell_y`, `num_cell_z`: Number of cells in spatial directions
- `num_cell_vpar`, `num_cell_mu`: Number of cells in velocity space
- `x_inner`: Width of the closed flux surface region [m]
- `x_outer`: Width of the SOL region [m]
- `nrhos_y`: Number of rho_s0 in the y-direction
- `nvth_vpar_e/i`: Parallel velocity extent for electrons/ions
- `nvth_mu_e/i`: Magnetic moment extent for electrons/ions
- `nuFrac`: Fraction of the Coulomb collision frequency

### 4. Time Parameters
- `final_time`: Final simulation time [s]
- `num_frames`: Number of output frames

## How to Use

### Method 1: Local Web Server (Recommended)
1. Navigate to the `parameter_ui` directory
2. Start a local web server:
   ```bash
   # Using Python 3
   python -m http.server 8000
   
   # Using Python 2
   python -m SimpleHTTPServer 8000
   
   # Using Node.js (if you have it installed)
   npx http-server
   ```
3. Open your browser and go to `http://localhost:8000`

### Method 2: Direct File Opening
1. Open the `index.html` file directly in your web browser
2. Note: Some features may not work due to browser security restrictions

### Using the Interface
1. **Modify Parameters**: Click on any input field and enter your desired values
2. **Get Help**: Hover over parameter names to see tooltips with descriptions
3. **Preview Code**: The generated C code appears in the preview section at the bottom
4. **Download File**: Click "Download File" to save the complete input.c file
5. **Reset Values**: Use "Reset to Defaults" to restore original values

## File Structure

```
parameter_ui/
├── index.html      # Main HTML interface
├── styles.css      # CSS styling
├── script.js       # JavaScript functionality
└── README.md       # This documentation
```

## Generated Output

The interface generates a complete C file that includes:
- All the original includes and function prototypes
- The q-profile function (unchanged)
- Your customized parameters in the `create_ctx()` function
- All the remaining functions from the original file

The generated file is ready to use as an input file for your Gkyl simulations.

## Browser Compatibility

The interface works with modern web browsers:
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Customization

You can easily customize the interface by:
- Modifying `defaultParameters` in `script.js` to change default values
- Adding new parameters by updating the HTML, CSS, and JavaScript files
- Changing the styling in `styles.css`
- Updating the C code template in `script.js`

## Troubleshooting

### Common Issues:
1. **Interface doesn't load**: Make sure you're using a web server or modern browser
2. **Download doesn't work**: Check if pop-ups are blocked in your browser
3. **Values not updating**: Ensure JavaScript is enabled in your browser

### Tips:
- Use scientific notation for very large or small numbers (e.g., `2.0e19`)
- Check the browser console (F12) for any error messages
- Validate that all required fields have appropriate values

## Support

If you encounter issues or need to add new parameters, you can:
1. Check the browser console for error messages
2. Verify that all input values are valid numbers
3. Ensure the C code template in `script.js` matches your requirements

The interface is designed to be easily extensible - you can add new parameters by following the existing patterns in the HTML, CSS, and JavaScript files.
