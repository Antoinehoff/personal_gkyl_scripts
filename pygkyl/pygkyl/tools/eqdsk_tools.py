# -----------------------------------------------------------------------------
# plot_geqdsk_equilibrium.py
#
# A script to read a GEQDSK file and generate two plots:
#   1. The poloidal cross-section of the magnetic equilibrium, with an
#      optional overlay of an approximate Miller geometry equilibrium.
#   2. The safety factor (q) profile as a function of the major radius.
#
# Based on original script by Manaure Francisquez and T. Bernard.
# Refined with Miller geometry comparison.
# Refined further with least squares fitting for Miller parameters, A. Hoffmann 2025.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from freeqdsk import geqdsk
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, Bounds
import sys

# ============================ USER SETTINGS ==================================

# Path to the g-file you want to analyze
GFILE_PATH = "default/g065125.01298"

# --- Comparison Options ---
PLOT_MILLER_GEOMETRY = True  # Set to True to overlay the Miller geometry
# This is the 'a_shift' parameter in your formula. It controls the Shafranov shift
# effect. A value of 0 gives non-shifted ellipses. Typical values are 0.1-0.5.
MILLER_SHAFRANOV_PARAM = 0.25

# --- Plotting Options ---
SAVE_PLOTS = True
OUTPUT_DIR = "./"
FIGURE_FORMAT = '.png'

# plt.rcParams.update({
#     "font.size": 14,
#     "lines.linewidth": 2.5,
#     "image.cmap": 'viridis',
#     "axes.labelsize": 16,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
#     "legend.fontsize": 14,
# })

# ========================= HELPER FUNCTIONS ================================

def calculate_miller_parameters(gfile_data, method='manual', input_params=None):
    
    rbdry, zbdry = gfile_data["rbdry"], gfile_data["zbdry"]
    R_axis = gfile_data["rmagx"]
    Z_axis = gfile_data["zmagx"]

    ashift = MILLER_SHAFRANOV_PARAM  # This is the user-defined shift parameter

    r_max, r_min = np.max(rbdry), np.min(rbdry)
    
    # Minor radius 'a'
    # amid = (r_max - r_min) / 2.0
    R_LCFSmid = r_max
    amid = R_axis/ashift - np.sqrt(R_axis*(R_axis - 2*ashift*R_LCFSmid + 2*ashift*R_axis))/ashift

    # Elongation 'kappa'
    z_max, z_min = np.max(zbdry), np.min(zbdry)
    kappa = (z_max - z_min) / (r_max - r_min)

    # Triangularity 'delta'
    # Defined as (R_geometric - R_top) / a
    r_geometric_center = (r_max + r_min) / 2.0
    # Find the R value at the highest Z point of the LCFS
    r_at_top_of_lcfs = rbdry[np.argmax(zbdry)]
    delta = (r_geometric_center - r_at_top_of_lcfs) / amid
    
    if method == 'optimization':
        # Use scipy.optimize.minimize instead of curve_fit
        def miller_distance_objective(params):
            """
            Objective function: minimize sum of squared distances from boundary points
            to the Miller parametric curve.
            """
            ashift_opt, kappa_opt, delta_opt = params

            # Recalculate amid and R_LCFSmid for this ashift_opt
            R_LCFSmid_opt = r_max  # This could also be optimized
            amid_opt = R_axis/ashift_opt - np.sqrt(R_axis*(R_axis - 2*ashift_opt*R_LCFSmid_opt + 2*ashift_opt*R_axis))/ashift_opt
            
            total_distance = 0
            
            # For each boundary point, find the closest point on the Miller curve
            for rb, zb in zip(rbdry, zbdry):
                
                # Define the Miller curve as a function of theta
                def miller_curve(theta):
                    r = amid_opt
                    R_miller = R_axis - ashift_opt * r**2 / (2. * R_axis) + r * np.cos(theta + np.arcsin(delta_opt) * np.sin(theta))
                    Z_miller = Z_axis + kappa_opt * r * np.sin(theta)
                    return R_miller, Z_miller
                
                # Find the theta that minimizes distance to this boundary point
                def distance_to_point(theta):
                    R_miller, Z_miller = miller_curve(theta)
                    return (R_miller - rb)**2 + (Z_miller - zb)**2
                
                # Find minimum distance for this boundary point
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(distance_to_point, bounds=(0, 2*np.pi), method='bounded')
                min_distance_squared = result.fun
                
                total_distance += min_distance_squared
            
            return total_distance
        
        # Initial guess using geometric parameters
        initial_guess = [ashift, kappa, delta]
        print(f"Initial guess for optimization: {initial_guess}")
        
        # Set reasonable bounds for the parameters
        bounds = Bounds(
            [0.1, 0.5, -1.0],  # lower bounds: [ashift, kappa, delta] - made bounds tighter
            [1.0, 3.0, 1.0]      # upper bounds
        )
        
        # Minimize the distance
        result = minimize(miller_distance_objective, initial_guess, 
                         method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            ashift, kappa, delta = result.x
            amid = R_axis/ashift - np.sqrt(R_axis*(R_axis - 2*ashift*R_LCFSmid + 2*ashift*R_axis))/ashift
            print(f"Optimization successful. Residual: {result.fun:.6f}")
        else:
            print(f"Optimization failed: {result.message}")
            print("Using geometric parameters as fallback.")
            
    elif method == 'optimization_free_amid':
        # Use scipy.optimize.minimize with amid as a free parameter
        def miller_distance_objective(params):
            """
            Objective function with amid as a free parameter.
            """
            amid_opt, ashift_opt, kappa_opt, delta_opt = params
            
            total_distance = 0
            
            # For each boundary point, find the closest point on the Miller curve
            for rb, zb in zip(rbdry, zbdry):
                
                # Define the Miller curve as a function of theta
                def miller_curve(theta):
                    r = amid_opt
                    R_miller = R_axis - ashift_opt * r**2 / (2. * R_axis) + r * np.cos(theta + np.arcsin(delta_opt) * np.sin(theta))
                    Z_miller = Z_axis + kappa_opt * r * np.sin(theta)
                    return R_miller, Z_miller
                
                # Find the theta that minimizes distance to this boundary point
                def distance_to_point(theta):
                    R_miller, Z_miller = miller_curve(theta)
                    return (R_miller - rb)**2 + (Z_miller - zb)**2
                
                # Find minimum distance for this boundary point
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(distance_to_point, bounds=(0, 2*np.pi), method='bounded')
                min_distance_squared = result.fun
                
                total_distance += min_distance_squared
            
            return total_distance
        
        # Initial guess using geometric parameters
        initial_guess = [amid, ashift, kappa, delta]
        print(f"Initial guess for optimization: {initial_guess}")
        
        # Set reasonable bounds for the parameters
        bounds = Bounds(
            [0.01, 0.01, 0.5, -1.0],  # lower bounds: [amid, ashift, kappa, delta]
            [2.0, 1.0, 3.0, 1.0]      # upper bounds
        )
        
        # Minimize the distance
        result = minimize(miller_distance_objective, initial_guess, 
                        method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            amid, ashift, kappa, delta = result.x
            print(f"Optimization successful. Residual: {result.fun:.6f}")
            print(f"Optimized parameters: amid={amid:.4f}, ashift={ashift:.4f}, kappa={kappa:.4f}, delta={delta:.4f}")
        else:
            print(f"Optimization failed: {result.message}")
            print("Using geometric parameters as fallback.")            
    # Also return axis location for convenience
    amid0 = R_axis/ashift - np.sqrt(R_axis*(R_axis - 2*ashift*r_max + 2*ashift*R_axis))/ashift
    # Compare amid and amid0
    print(f"Calculated amid: {amid:.4f}, Geometric amid: {amid0:.4f}")
    params = {
        'R_axis': gfile_data["rmagx"],
        'Z_axis': gfile_data["zmagx"],
        'R_LCFSmid': r_max,
        'amid': amid,
        'ashift': ashift,
        'kappa': kappa,
        'delta': delta
    }
    if method == 'manual':
        print("Using manual Miller parameters.")
    return params

def generate_miller_lcfs(miller_params, r=None):
    """
    Generates R, Z coordinates for a Miller LCFS using the provided formula.

    Args:
        miller_params (dict): Dictionary with R_axis, Z_axis, a, kappa, delta.
        shafranov_param (float): The 'a_shift' parameter in the formula.

    Returns:
        tuple: (R_miller, Z_miller) coordinate arrays.
    """
    # Create a high-resolution array for the poloidal angle
    theta = np.linspace(0, 2 * np.pi, 200)
    
    # Get parameters from the dictionary for clarity
    R_axis = miller_params['R_axis']
    Z_axis = miller_params['Z_axis']
    amid = miller_params['amid']
    ashift = miller_params['ashift']
    kappa = miller_params['kappa']
    delta = miller_params['delta']
    
    # We are plotting the LCFS, so the minor radius variable 'r' is 'a'
    r = r if r else amid
    
    # Apply the user-provided formulas
    R_miller = R_axis - ashift * r**2 / (2. * R_axis) + r * np.cos(theta + np.arcsin(delta) * np.sin(theta))
    Z_miller = Z_axis + kappa * r * np.sin(theta)
    
    return R_miller, Z_miller

# ========================= CORE ANALYSIS SCRIPT ============================

def plot_equilibrium_and_q_profile(gfile_path, fit_method='geometric', x_surf = [],
                                   title='', outfilename='',
                                   plot_miller_geom=True, save_plots=True, 
                                   show_legend=True, show_colorbar=True):
    """
    Reads a g-file and generates equilibrium and q-profile plots.
    """
    # --- 1. Load Data from G-file ---
    try:
        with open(gfile_path, "r") as f:
            print(f"Reading g-file: {gfile_path}")
            gfile_data = geqdsk.read(f)
    except FileNotFoundError:
        print(f"Error: The file '{gfile_path}' was not found.")
        sys.exit(1)

    psi_RZ = gfile_data["psi"]
    q_profile = gfile_data["qpsi"]
    R_grid = np.linspace(gfile_data["rleft"], gfile_data["rleft"] + gfile_data["rdim"], gfile_data["nx"])
    Z_grid = np.linspace(gfile_data["zmid"] - gfile_data["zdim"]/2, gfile_data["zmid"] + gfile_data["zdim"]/2, gfile_data["ny"])
    RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing='ij')

    # --- 2. Plot the 2D Equilibrium ---
    print("Generating equilibrium plot...")
    fig1, ax1 = plt.subplots(figsize=(3, 5))
    
    levels = np.linspace(psi_RZ.min(), psi_RZ.max(), 30)
    contour = ax1.contourf(RR, ZZ, psi_RZ, levels=levels)
    if show_colorbar: fig1.colorbar(contour, ax=ax1, label=r'$\psi$ (Wb/rad)')
    
    ax1.plot(gfile_data["rlim"], gfile_data["zlim"], 'k-', linewidth=2.5, label='Vessel')
    ax1.plot(gfile_data["rbdry"], gfile_data["zbdry"], 'w-', linewidth=3.0, label='Experimental LCFS')
    ax1.plot(gfile_data["rmagx"], gfile_data["zmagx"], 'wx', markersize=10, mew=2.5, label='Magnetic Axis')

    # --- Miller Geometry Comparison ---
    if plot_miller_geom:
        print("\n--- Miller Geometry Analysis ---")
        # Calculate parameters from the experimental LCFS
        miller_params = calculate_miller_parameters(gfile_data, method=fit_method)
        print(f"Calculated Parameters:")
        print(f"  amid   = {miller_params['amid']:.4f} m")
        print(f"  R_axis = {miller_params['R_axis']:.4f} m")
        print(f"  Z_axis = {miller_params['Z_axis']:.4f} m")
        print(f"  R_LCFSmid = {miller_params['R_LCFSmid']:.4f} m")
        print(f"  ashift = {miller_params['ashift']:.4f}")
        print(f"  κ      = {miller_params['kappa']:.4f}")
        print(f"  δ      = {miller_params['delta']:.4f}")
        if 'optimization' in fit_method:
            print("Using least squares fitting method for Miller parameters.")
        else:
            print(f"Using user-set shafranov_param (a_shift) = {MILLER_SHAFRANOV_PARAM}")

        # Generate the Miller LCFS coordinates
        R_miller, Z_miller = generate_miller_lcfs(miller_params)
        
        # Overlay the Miller LCFS on the plot
        ax1.plot(R_miller, Z_miller, 'r--', linewidth=2.5, label='Miller Optim. LCFS' if fit_method=='optimization' else 'Miller Approx. LCFS')
        
        if x_surf:
            for x in x_surf:
                r = miller_params['amid'] + x
                R_miller_x, Z_miller_x = generate_miller_lcfs(miller_params, r=r)
                ax1.plot(R_miller_x, Z_miller_x, 'r:', linewidth=1.5, label=f'Miller Optim. at x={x:.2f} m')
    
    title = title if title else f"Equilibrium from {gfile_path.split('/')[-1]}"
    ax1.set_title(title)
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_aspect('equal')
    if show_legend: ax1.legend()
    plt.tight_layout()

    if save_plots:
        output_filename = f"{OUTPUT_DIR}equilibrium_plot{FIGURE_FORMAT}"
        if title:
            title = title.replace(" ", "_").replace("\\#", "")
            title = title.replace('(', '').replace(')', '')
            output_filename = f"{OUTPUT_DIR}{title}_equilibrium_plot{FIGURE_FORMAT}"
        if outfilename: output_filename = outfilename
        plt.savefig(output_filename)
        print(f"\nSaved equilibrium plot to {output_filename}")
    plt.show()

    # --- 3. Calculate and Plot the Q-Profile vs. Major Radius (Unchanged) ---
    # ... (The q-profile plotting logic remains the same) ...
    print("\nGenerating q-profile plot...")
    z_axis_idx = np.abs(Z_grid - gfile_data["zmagx"]).argmin()
    psi_outboard_midplane = psi_RZ[:, z_axis_idx]
    r_axis_idx = np.abs(R_grid - gfile_data["rmagx"]).argmin()
    R_of_psi_interpolator = CubicSpline(
        psi_outboard_midplane[r_axis_idx:], 
        R_grid[r_axis_idx:],
        extrapolate=False
    )
    psi_normalized = np.linspace(0, 1, len(q_profile))
    psi_values = gfile_data["simagx"] + psi_normalized * (gfile_data["sibdry"] - gfile_data["simagx"])
    R_for_q_profile = R_of_psi_interpolator(psi_values)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    valid_indices = ~np.isnan(R_for_q_profile)
    ax2.plot(R_for_q_profile[valid_indices], q_profile[valid_indices], 'b-', label='q from g-file')
    ax2.set_title(f"Safety Factor (q) Profile")
    ax2.set_xlabel('Major Radius R (m)')
    ax2.set_ylabel('Safety Factor q')
    ax2.grid(True, linestyle='--')
    if show_legend: ax2.legend()
    plt.tight_layout()

    if save_plots:
        output_filename = f"{OUTPUT_DIR}q_profile_plot{FIGURE_FORMAT}"
        plt.savefig(output_filename)
        print(f"Saved q-profile plot to {output_filename}")
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    plot_equilibrium_and_q_profile(GFILE_PATH, save_plots=SAVE_PLOTS)
    print("\nAnalysis complete.")