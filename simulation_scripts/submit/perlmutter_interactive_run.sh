#!/bin/bash

# Display help message
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [GKYL_ARGS]

Run Gkeyll simulation on Perlmutter interactive nodes (GPU or CPU).

Options:
  -d, --dim DIM          Dimensionality: 2d/2D/2 or 3d/3D/3 (default: 3d)
  -e, --exec NAME        Executable name (default: gkeyll)
  -n, --ntasks NUM       Number of tasks (GPUs or CPU ranks) to use (default: 4)
  -l, --log LOGNAME      Log file name (default: gkyl_run.log)
  -c, --cpu              Run on CPU (omits the -g flag passed to gkeyll)
  -p, --print            Print the srun command instead of executing it
  -h, --help             Display this help message
  The remaining arguments are passed directly to the Gkeyll executable.

Example:
  $(basename "$0") -n 8 -d 2 -r 40 [To restart at frame 40 on GPU]
  $(basename "$0") -c -n 32 --dim 3d --log my_simulation.log [CPU run]
  $(basename "$0") --ntasks 16 --dim 3d --exec custom_gkeyll --log my_simulation.log

EOF
}

ntasks=4  # Default value

# Default decomposition direction
decomp_dir="-e"

# Default executable name
executable="gkeyll"

# Default log file name
logname="gkyl_run.log"

# Default: GPU mode
use_gpu=true

# Default: execute (not just print)
print_only=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dim)
            if [[ "$2" == "2d" || "$2" == "2D" || "$2" == "2" ]]; then
                decomp_dir="-d"
            elif [[ "$2" == "3d" || "$2" == "3D" || "$2" == "3" ]]; then
                decomp_dir="-e"
            else
                echo "Error: Invalid dimension '$2'. Use '2d' or '3d'."
                exit 1
            fi
            shift 2
            ;;
        -e|--exec)
            executable="$2"
            shift 2
            ;;
        -n|--ntasks)
            ntasks="$2"
            shift 2
            ;;
        -l|--log)
            logname="$2"
            shift 2
            ;;
        -c|--cpu)
            use_gpu=false
            shift
            ;;
        -p|--print)
            print_only=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

gpu_flag=$( [[ "$use_gpu" == true ]] && echo "-g" || echo "" )
CMD="srun -u -n $ntasks ./$executable $gpu_flag -M $decomp_dir $ntasks $* | tee $logname"
if [[ "$print_only" == true ]]; then
    echo $CMD
else
    echo "Executing command: $CMD"
    eval $CMD
fi