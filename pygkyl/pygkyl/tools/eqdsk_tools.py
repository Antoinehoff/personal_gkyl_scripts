# -----------------------------------------------------------------------------
# plot_geqdsk_equilibrium.py
#
# A script to read a GEQDSK file and generate combined plots:
#   1. The poloidal cross-section of the magnetic equilibrium
#   2. The safety factor (q) profile as a function of the major radius
#
# Based on original script by Manaure Francisquez and T. Bernard.
# Refined with Miller geometry comparison and least squares fitting, A. Hoffmann 2025.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from freeqdsk import geqdsk
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, minimize_scalar, Bounds
import sys

# ============================ USER SETTINGS ==================================

# Path to the g-file you want to analyze
GFILE_PATH = "default/g065125.01298"

# --- Comparison Options ---
PLOT_MILLER_GEOMETRY = True  # Set to True to overlay the Miller geometry
MILLER_SHAFRANOV_PARAM = 0.25  # Controls the Shafranov shift effect

# --- Plotting Options ---
SAVE_PLOTS = True
OUTPUT_DIR = "./"
FIGURE_FORMAT = '.png'

# ========================= HELPER FUNCTIONS ================================

def calculate_miller_parameters(gfile_data, method='manual'):
    """Calculate Miller parameters from GEQDSK boundary data."""
    rbdry, zbdry = gfile_data["rbdry"], gfile_data["zbdry"]
    R_axis = gfile_data["rmagx"]
    Z_axis = gfile_data["zmagx"]
    ashift = MILLER_SHAFRANOV_PARAM

    r_max, r_min = np.max(rbdry), np.min(rbdry)
    R_LCFSmid = r_max
    amid = R_axis/ashift - np.sqrt(R_axis*(R_axis - 2*ashift*R_LCFSmid + 2*ashift*R_axis))/ashift

    # Elongation and triangularity
    z_max, z_min = np.max(zbdry), np.min(zbdry)
    kappa = (z_max - z_min) / (r_max - r_min)
    r_geometric_center = (r_max + r_min) / 2.0
    r_at_top_of_lcfs = rbdry[np.argmax(zbdry)]
    delta = (r_geometric_center - r_at_top_of_lcfs) / amid
    
    if 'optimization' in method:
        def miller_distance_objective(params):
            """Objective function: minimize distance from boundary to Miller curve."""
            if method == 'optimization_free_amid':
                amid_opt, ashift_opt, kappa_opt, delta_opt = params
            else:
                ashift_opt, kappa_opt, delta_opt = params
                amid_opt = R_axis/ashift_opt - np.sqrt(R_axis*(R_axis - 2*ashift_opt*R_LCFSmid + 2*ashift_opt*R_axis))/ashift_opt
            
            total_distance = 0
            for rb, zb in zip(rbdry, zbdry):
                def distance_to_point(theta):
                    r = amid_opt
                    R_miller = R_axis - ashift_opt * r**2 / (2. * R_axis) + r * np.cos(theta + np.arcsin(delta_opt) * np.sin(theta))
                    Z_miller = Z_axis + kappa_opt * r * np.sin(theta)
                    return (R_miller - rb)**2 + (Z_miller - zb)**2
                
                result = minimize_scalar(distance_to_point, bounds=(0, 2*np.pi), method='bounded')
                total_distance += result.fun
            
            return total_distance
        
        # Set up optimization
        if method == 'optimization_free_amid':
            initial_guess = [amid, ashift, kappa, delta]
            bounds = Bounds([0.01, 0.01, 0.5, -1.0], [2.0, 1.0, 3.0, 1.0])
        else:
            initial_guess = [ashift, kappa, delta]
            bounds = Bounds([0.1, 0.5, -1.0], [1.0, 3.0, 1.0])
        
        result = minimize(miller_distance_objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            if method == 'optimization_free_amid':
                amid, ashift, kappa, delta = result.x
            else:
                ashift, kappa, delta = result.x
                amid = R_axis/ashift - np.sqrt(R_axis*(R_axis - 2*ashift*R_LCFSmid + 2*ashift*R_axis))/ashift
            print(f"Optimization successful. Residual: {result.fun:.6f}")
        else:
            print(f"Optimization failed: {result.message}")
            print("Using geometric parameters as fallback.")
    
    return {
        'R_axis': R_axis, 'Z_axis': Z_axis, 'R_LCFSmid': r_max,
        'amid': amid, 'ashift': ashift, 'kappa': kappa, 'delta': delta
    }

def generate_miller_lcfs(miller_params, r=None):
    """Generate R, Z coordinates for a Miller LCFS."""
    theta = np.linspace(0, 2 * np.pi, 200)
    
    R_axis = miller_params['R_axis']
    Z_axis = miller_params['Z_axis']
    amid = miller_params['amid']
    ashift = miller_params['ashift']
    kappa = miller_params['kappa']
    delta = miller_params['delta']
    
    r = r if r else amid
    
    R_miller = R_axis - ashift * r**2 / (2. * R_axis) + r * np.cos(theta + np.arcsin(delta) * np.sin(theta))
    Z_miller = Z_axis + kappa * r * np.sin(theta)
    
    return R_miller, Z_miller

# ========================= CORE ANALYSIS SCRIPT ============================

def plot_equilibrium_and_q_profile(gfile_path, fit_method='geometric', x_surf=[],
                                   title='', outfilename='',
                                   plot_miller_geom=True, save_plots=True, 
                                   show_legend=True, show_colorbar=True,
                                   qprofile_fit=[]):
    """
    Reads a g-file and generates combined equilibrium and q-profile plots.
    """
    # Load Data from G-file
    try:
        with open(gfile_path, "r") as f:
            gfile_data = geqdsk.read(f)
    except FileNotFoundError:
        print(f"Error: The file '{gfile_path}' was not found.")
        sys.exit(1)

    psi_RZ = gfile_data["psi"]
    q_profile = gfile_data["qpsi"]
    R_grid = np.linspace(gfile_data["rleft"], gfile_data["rleft"] + gfile_data["rdim"], gfile_data["nx"])
    Z_grid = np.linspace(gfile_data["zmid"] - gfile_data["zdim"]/2, gfile_data["zmid"] + gfile_data["zdim"]/2, gfile_data["ny"])
    RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing='ij')

    # Create combined figure with subplots - back to 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4))
    
    # Plot 1: Equilibrium
    levels = np.linspace(psi_RZ.min(), psi_RZ.max(), 30)
    contour = ax1.contourf(RR, ZZ, psi_RZ, levels=levels)
    if show_colorbar: 
        fig.colorbar(contour, ax=ax1, label=r'$\psi$ (Wb/rad)')
    
    ax1.plot(gfile_data["rlim"], gfile_data["zlim"], 'k-', linewidth=1.5, label='Vessel')
    ax1.plot(gfile_data["rbdry"], gfile_data["zbdry"], 'w-', linewidth=2.0, label='Experimental LCFS')
    ax1.plot(gfile_data["rmagx"], gfile_data["zmagx"], 'wx', markersize=8, mew=2.5, label='Magnetic Axis')

    # Miller Geometry Analysis
    miller_params = None
    if plot_miller_geom:
        print("--- Miller Geometry fit ---")
        miller_params = calculate_miller_parameters(gfile_data, method=fit_method)
        
        # Print parameters in C-style format
        print(f"double R_axis = {miller_params['R_axis']:.6f};")
        print(f"double Z_axis = {miller_params['Z_axis']:.6f};")
        print(f"double amid = {miller_params['amid']:.6f};")
        print(f"double ashift = {miller_params['ashift']:.6f};")
        print(f"double kappa = {miller_params['kappa']:.6f};")
        print(f"double delta = {miller_params['delta']:.6f};")
        print(f"double R_LCFSmid = {miller_params['R_LCFSmid']:.6f};")
        
        # Generate and plot Miller LCFS
        R_miller, Z_miller = generate_miller_lcfs(miller_params)
        label = 'Miller Optim. LCFS' if 'optimization' in fit_method else 'Miller Approx. LCFS'
        ax1.plot(R_miller, Z_miller, 'r--', linewidth=1.5, label=label)
        
        # Plot additional surfaces if requested
        for x in x_surf:
            r = miller_params['amid'] + x
            R_miller_x, Z_miller_x = generate_miller_lcfs(miller_params, r=r)
            ax1.plot(R_miller_x, Z_miller_x, 'r:', linewidth=1.5, label=f'x={x:.2f} m')
    
    plot_title = title if title else f"Equilibrium from {gfile_path.split('/')[-1]}"
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_aspect('equal')
    if show_legend: ax1.legend()

    # Plot 2: Q-Profile
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

    valid_indices = ~np.isnan(R_for_q_profile)
    
    if miller_params:
        # Calculate new cubic fit
        Rfit, qfit = [], []
        for i in range(len(R_for_q_profile)):
            if valid_indices[i]:
                rhoi = (R_for_q_profile[i] - miller_params['R_axis']) / miller_params['amid']
                if 0.5 < rhoi < 0.8:
                    Rfit.append(R_for_q_profile[i])
                    qfit.append(q_profile[i])
        if Rfit:
            new_cubic_fit = np.polyfit(np.array(Rfit), np.array(qfit), 3)
            
            # Print new cubic fit coefficients in list format
            print(f"new_cubic_fit = {new_cubic_fit.tolist()}")
        
        rho = (R_for_q_profile - miller_params['R_axis']) / miller_params['amid']
        ax2.plot(rho[valid_indices], q_profile[valid_indices], 'b-', label='q from g-file')
        
        # Plot vertical lines for x_surf
        for x in x_surf:
            rho_x = (miller_params['amid'] + x) / miller_params['amid']
            ax2.axvline(rho_x, color='r', linestyle=':', label=f'x={x:.2f} m')
        
        # Plot fits if provided
        if qprofile_fit:
            # Print input q-profile fit coefficients
            print(f"qprofile_fit = {qprofile_fit}")
            
            Rmin, Rmax = R_for_q_profile[valid_indices].min(), R_for_q_profile[valid_indices].max()
            for x in x_surf:
                Rmin = min(Rmin, miller_params['amid'] + x + miller_params['R_axis'])
                Rmax = max(Rmax, miller_params['amid'] + x + miller_params['R_axis'])
            R = np.linspace(Rmin, Rmax, 100)
            rho_fit = (R - miller_params['R_axis']) / miller_params['amid']
            ax2.plot(rho_fit, np.polyval(qprofile_fit, R), 'k--', label='q fit')
            # if 'new_cubic_fit' in locals():
                # ax2.plot(rho_fit, np.polyval(new_cubic_fit, R), 'g--', label='New cubic fit')
        
        ax2.set_xlabel(r'$\rho$')
    else:
        ax2.plot(R_for_q_profile[valid_indices], q_profile[valid_indices], 'b-', label='q from g-file')
        ax2.set_xlabel('Major Radius R (m)')
    
    # ax2.set_title("Safety Factor (q) Profile")
    ax2.set_ylabel('Safety Factor q')
    ax2.grid(True, linestyle='--')
    ax2.set_ylim(0, 10)
    if show_legend: ax2.legend()

    # Put the title at the top of the figure
    fig.suptitle(plot_title, fontsize=16, y=0.9, x=0.6)
    plt.tight_layout()

    if save_plots:
        output_filename = outfilename if outfilename else f"{OUTPUT_DIR}combined_equilibrium_q_plot{FIGURE_FORMAT}"
        plt.savefig(output_filename)
    plt.show()
    
    if save_plots:
        print(f"Saved combined plot to {output_filename}")

# --- Main execution block ---
if __name__ == "__main__":
    plot_equilibrium_and_q_profile(GFILE_PATH, save_plots=SAVE_PLOTS)
    print("\nAnalysis complete.")