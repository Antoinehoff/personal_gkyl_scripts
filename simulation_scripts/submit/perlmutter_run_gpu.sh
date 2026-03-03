#!/bin/bash

# Display help message
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [GKYL_ARGS]

Run Gkeyll simulation on Perlmutter GPU nodes.

Options:
  -d, --dim DIM          Dimensionality: 2d/2D/2 or 3d/3D/3 (default: 3d)
  -e, --exec NAME        Executable name (default: gkeyll)
  -n, --ngpu NUM         Number of GPUs to use (default: 4)
  -l, --log LOGNAME      Log file name (default: gkyl_run_gpu.log)
  -h, --help             Display this help message
  The remaining arguments are passed directly to the Gkeyll executable.

Example:
  $(basename "$0") -n 8 -d 2d -r 40 [To restart at frame 40]
  $(basename "$0") --ngpu 16 --dim 3d --exec custom_gkeyll --log my_simulation.log

EOF
}

ngpu=4  # Default value

# Assuming 4 GPUs per node, calculate the number of nodes required.
gpus_per_node=4
nnodes=$(( (ngpu + gpus_per_node - 1) / gpus_per_node ))

# Default decomposition direction
decomp_dir="-e"

# Default executable name
executable="gkeyll"

# Default log file name
logname="gkyl_run_gpu.log"

# Parse arguments for dimensionality option
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dim)
            if [[ "$2" == "2d" || "$2" == "2D" || "$2" == "2" ]]; then
                decomp="-d"
            elif [[ "$2" == "3d" || "$2" == "3D" || "$2" == "3" ]]; then
                decomp="-e"
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
        -n|--ngpu)
            ngpu="$2"
            shift 2
            ;;
        -l|--log)
            logname="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Use -N to specify the calculated number of nodes.
# --ntasks is set to the number of GPUs, assuming one task per GPU.
CMD="srun -N $nnodes --ntasks=$ngpu --gpus-per-task=1 --gpu-bind=closest -u ./$executable -g -M $decomp_dir $ngpu $* | tee $logname"
echo "Executing command: $CMD"
eval $CMD