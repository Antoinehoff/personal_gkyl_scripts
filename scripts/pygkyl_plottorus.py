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
    python pygkyl_plottorus.py --plottype snapshot --fieldName Ti --frameidx -1 --fluctuation yavg_relative

- Plot a movie of Te using a custom camera path:
    python pygkyl_plottorus.py --plottype movie --fieldName Te --cameras global zoom_lower zoom_lower

- Change color limits and use log scale:
    python pygkyl_plottorus.py --clim 0 100 --logScale

- Restrict the plotted region:
    python pygkyl_plottorus.py --rhoLim 0 1 --phiLim 0 3.14

PARAMETERS:
-----------
--simdir         Path to the simulation directory.
--fileprefix     Prefix for the simulation files.
--plottype       'snapshot' for a single frame, 'movie' for a time sequence.
--rhoLim         Radial limits (e.g., --rhoLim 0 1).
--phiLim         Toroidal angle limits in radians (e.g., --phiLim 0 4.71).
--Nint_polproj   Number of poloidal integration points.
--Nint_fsproj    Number of field-line integration points.
--fieldName      Field to plot (e.g., Ti, Te, phi, etc.).
--fluctuation    Fluctuation type (e.g., yavg_relative).
--frameidx       Frame index for snapshot (negative for last frame).
--movieframeidx  Frame index to start the movie from ('all' or integer index).
--imgSize        Image size in pixels (width height).
--clim           Colorbar limits (min max).
--logScale       Use logarithmic color scale.
--cameras        Camera path for movie (sequence of 'global' or 'zoom_lower').
-h, --help       Show this help message and exit.

NOTES:
------
- For 'movie', the camera path can be a sequence, e.g., --cameras global zoom_lower.
- For 'snapshot', only the first camera in the path is used.
- The script prints the number of available frames and uses the specified frame(s).
- If you close or move the movie window, the movie generation will be disturbed.

""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--simdir', type=str, default='sim_data_dir_example/3x2v_example/gk_tcv_posD_iwl_3x2v_electron_heating/', help='Simulation directory')
    parser.add_argument('--fileprefix', type=str, default='gk_tcv_posD_iwl_3x2v_D02', help='File prefix')
    parser.add_argument('--plottype', type=str, choices=['snapshot','movie'], default='snapshot', help='Plot type')
    parser.add_argument('--rhoLim', type=float, nargs=2, default=[2,-2], help='Radial limits')
    parser.add_argument('--phiLim', type=float, nargs=2, default=[0, 3*np.pi/2], help='Toroidal angle limits')
    parser.add_argument('--Nint_polproj', type=int, default=32, help='Number of poloidal integration points')
    parser.add_argument('--Nint_fsproj', type=int, default=24, help='Number of field-line integration points')
    parser.add_argument('--fieldName', type=str, default='Ti', help='Field to plot')
    parser.add_argument('--fluctuation', type=str, default='', help='Fluctuation type')
    parser.add_argument('--frameidx', type=int, default=-1, help='Frame index for snapshot')
    parser.add_argument('--movieframeidx', type=str, default='all', help='Frame indices for movie ("all" or int)')
    parser.add_argument('--imgSize', type=int, nargs=2, default=[800,600], help='Image size in pixels')
    parser.add_argument('--clim', type=float, nargs=2, default=[], help='Color limits')
    parser.add_argument('--logScale', action='store_true', help='Use log scale for color bar')
    parser.add_argument('--cameras', type=str, nargs='+', default=['global', 'global', 'zoom_lower', 'zoom_lower'], help='Camera path for movie')
    parser.add_argument('--deviceconfig', type=str, choices=['tcv_nt','tcv_pt','d3d_nt','d3d_pt'], default='tcv_nt', help='Set the device geometry.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Check if there is simulation results in the simdir/fileprefix*
    result_files = [f for f in os.listdir(args.simdir) if f.startswith(args.fileprefix)]
    if not result_files:
        print(f"No simulation results found in {args.simdir} with prefix '{args.fileprefix}'.")
        sys.exit(1)
    
    # Ensure that the simdir is ending with '/'
    if not args.simdir.endswith('/'):
        args.simdir += '/'
    
    # Setup
    simulation = pygkyl.simulation_configs.import_config(args.deviceconfig, args.simdir, args.fileprefix)
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
        else:
            print(f"Invalid camera movement option: {camera}. Choose from 'global', 'zoom_lower'.")
            sys.exit(1)

    
    torproj = pygkyl.TorusProjection()
    torproj.setup(simulation, Nint_polproj=args.Nint_polproj, Nint_fsproj=args.Nint_fsproj,
                  phiLim=args.phiLim, rhoLim=args.rhoLim, timeFrame=sim_frames[0])

    if args.plottype == 'snapshot':
        timeFrame = sim_frames[args.frameidx]
        torproj.plot(fieldName=args.fieldName, timeFrame=timeFrame, fluctuation=args.fluctuation, clim=args.clim, logScale=args.logScale,
                 vessel=True, filePrefix=args.fileprefix, imgSize=tuple(args.imgSize), jupyter_backend='none', colorbar=True, cameraSettings=camera_path[0])
    elif args.plottype == 'movie':
        if args.movieframeidx == 'all':
            timeFrames = sim_frames
        else:
            try:
                idx = int(args.movieframeidx)
                timeFrames = sim_frames[idx:]
            except Exception:
                print("movieframeidx must be 'all' or an integer index.")
                sys.exit(1)
                
        print("You will see the generation of the movie in a window. DO NOT MOVE IT OR CLOSE IT.")

        torproj.movie(fieldName=args.fieldName, timeFrames=timeFrames, fluctuation=args.fluctuation, filePrefix=args.fileprefix, imgSize=tuple(args.imgSize), colorbar=True,
                  cameraPath=camera_path, logScale=args.logScale, clim=args.clim)

if __name__ == "__main__":
    main()