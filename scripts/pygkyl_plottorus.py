import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Configure plotting
plt.rcParams["figure.figsize"] = (6,4)

# Custom libraries and routines
import pygkyl

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
Plot toroidal projections from Gkeyll simulation data.

This script visualizes simulation results on a toroidal geometry using the pygkyl library.
You can generate either a single snapshot or a movie of the evolution, with full control over
the field, fluctuation type, color scale, camera path, and more.

USAGE EXAMPLES:
---------------
- Plot a snapshot of the fluctuation of Ti at the last frame:
    python pygkyl_plottorus.py --plot_type snapshot --field_name Ti --frame_idx -1 --fluctuation yavg_relative

- Plot a movie of Te using a custom camera path:
    python pygkyl_plottorus.py --plot_type movie --field_name Te --cameras global zoom_lower zoom_lower

- Change color limits and use log scale:
    python pygkyl_plottorus.py --clim 0 100 --log_scale

- Restrict the plotted region:
    python pygkyl_plottorus.py --rho_lim 0 1 --phi_lim 0 3.14

PARAMETERS:
-----------
--sim_dir         Path to the simulation directory.
--file_prefix     Prefix for the simulation files.
--plot_type       'snapshot' for a single frame, 'movie' for a time sequence.
--rho_lim         Radial limits (e.g., --rho_lim 0 1).
--phi_lim         Toroidal angle limits in radians (e.g., --phi_lim 0 4.71).
--nint_polproj    Number of poloidal integration points.
--nint_fsproj     Number of field-line integration points.
--field_name      Field to plot (e.g., Ti, Te, phi, etc.).
--fluctuation     Fluctuation type (e.g., yavg_relative).
--frame_idx       Frame index for snapshot (negative for last frame).
--movie_frame_idx Frame index to start the movie from ('all' or integer index).
--img_size        Image size in pixels (width height).
--clim            Colorbar limits (min max).
--log_scale       Use logarithmic color scale.
--cameras         Camera path for movie (sequence of 'global' or 'zoom_lower').
--device_config   Set the device geometry.
--off_screen      Use off-screen rendering (for non-GUI environments).
--movie_type      Movie file type (e.g., mp4, gif).
-h, --help        Show this help message and exit.

NOTES:
------
- For 'movie', the camera path can be a sequence, e.g., --cameras global zoom_lower.
- For 'snapshot', only the first camera in the path is used.
- The script prints the number of available frames and uses the specified frame(s).
- If you close or move the movie window, the movie generation will be disturbed.

""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--sim_dir', type=str, default='sim_data_dir_example/3x2v_example/gk_tcv_posD_iwl_3x2v_electron_heating/', help='Simulation directory')
    parser.add_argument('--file_prefix', type=str, default='gk_tcv_posD_iwl_3x2v_D02', help='File prefix')
    parser.add_argument('--plot_type', type=str, choices=['snapshot','movie'], default='snapshot', help='Plot type')
    parser.add_argument('--rho_lim', type=float, nargs=2, default=[2,-2], help='Radial limits')
    parser.add_argument('--phi_lim', type=float, nargs=2, default=[0, 3*np.pi/2], help='Toroidal angle limits')
    parser.add_argument('--nint_polproj', type=int, default=32, help='Number of poloidal integration points')
    parser.add_argument('--nint_fsproj', type=int, default=24, help='Number of field-line integration points')
    parser.add_argument('--field_name', type=str, default='Ti', help='Field to plot')
    parser.add_argument('--fluctuation', type=str, default='', help='Fluctuation type')
    parser.add_argument('--frame_idx', type=int, default=-1, help='Frame index for snapshot')
    parser.add_argument('--movie_frame_idx', type=str, default='all', help='Frame indices for movie ("all" or int)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[800,600], help='Image size in pixels')
    parser.add_argument('--clim', type=float, nargs=2, default=[], help='Color limits')
    parser.add_argument('--log_scale', action='store_true', help='Use log scale for color bar')
    parser.add_argument('--cameras', type=str, nargs='+', default=['global', 'global', 'zoom_lower', 'zoom_lower'], help='Camera path for movie')
    parser.add_argument('--device_config', type=str, choices=['tcv_nt','tcv_pt','d3d_nt','d3d_pt','sparc','nstxu'], default='tcv_nt', help='Set the device geometry.')
    parser.add_argument('--off_screen', type=str, default='False', choices=['True','False'], help='Use off-screen rendering (for non-GUI environments)')
    parser.add_argument('--movie_type', type=str, default='gif', help='Movie file type (e.g., mp4, gif)')
    return parser.parse_args()

def main():
    args = parse_args()
    args.off_screen = args.off_screen in ['True', 'T', '1']
  
    # Check if there is simulation results in the sim_dir/file_prefix*
    result_files = [f for f in os.listdir(args.sim_dir) if f.startswith(args.file_prefix)]
    if not result_files:
        print(f"No simulation results found in {args.sim_dir} with prefix '{args.file_prefix}'.")
        sys.exit(1)
    
    # Ensure that the sim_dir is ending with '/'
    if not args.sim_dir.endswith('/'):
        args.sim_dir += '/'
    
    # Setup
    simulation = pygkyl.simulation_configs.import_config(args.device_config, args.sim_dir, args.file_prefix)
    simulation.normalization.set('t','mus')
    simulation.normalization.set('x','minor radius')
    simulation.normalization.set('y','Larmor radius')
    simulation.normalization.set('z','pi')
    simulation.normalization.set('fluid velocities','thermal velocity')
    simulation.normalization.set('temperatures','eV')
    simulation.normalization.set('pressures','Pa')
    simulation.normalization.set('energies','MJ')    

    sim_frames = simulation.available_frames['ion_BiMaxwellianMoments']
    print("%g time frames available (%g to %g)"%(len(sim_frames),sim_frames[0],sim_frames[-1]))

    camera_path = []
    for camera in args.cameras:
        if camera == 'global':
          camera_path.append(simulation.geom_param.camera_global)
        elif camera == 'zoom_lower':
          camera_path.append(simulation.geom_param.camera_zoom_lower)
        elif camera == 'zoom_obmp':
          camera_path.append(simulation.geom_param.camera_zoom_obmp)
        else:
            print(f"Invalid camera movement option: {camera}. Choose from 'global', 'zoom_lower'.")
            sys.exit(1)
            

    torproj = pygkyl.TorusProjection()
    torproj.setup(simulation, Nint_polproj=args.nint_polproj, Nint_fsproj=args.nint_fsproj,
                  phiLim=args.phi_lim, rhoLim=args.rho_lim, timeFrame=sim_frames[0])

    if args.plot_type == 'snapshot':
        timeFrame = sim_frames[args.frame_idx]
        torproj.plot(fieldName=args.field_name, timeFrame=timeFrame, fluctuation=args.fluctuation, 
                     clim=args.clim, logScale=args.log_scale, vessel=True, filePrefix=args.file_prefix, 
                     imgSize=tuple(args.img_size), jupyter_backend='none', colorbar=True, cameraSettings=camera_path[0],
                     off_screen=args.off_screen)
    elif args.plot_type == 'movie':
        if args.movie_frame_idx == 'all':
            timeFrames = sim_frames
        else:
            try:
                idx = int(args.movie_frame_idx)
                timeFrames = sim_frames[idx:]
            except Exception:
                print("movie_frame_idx must be 'all' or an integer index.")
                sys.exit(1)
                
        print("You will see the generation of the movie in a window. DO NOT MOVE IT OR CLOSE IT.")

        torproj.movie(fieldName=args.field_name, timeFrames=timeFrames, fluctuation=args.fluctuation, 
                      filePrefix=args.file_prefix, imgSize=tuple(args.img_size), colorbar=True,
                      cameraPath=camera_path, logScale=args.log_scale, clim=args.clim,
                      off_screen=args.off_screen, movie_type=args.movie_type)

if __name__ == "__main__":
    main()