#!/bin/bash

# Check if the number of arguments is less than 1 or if -h or --help is provided
if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <ngpu> [--dim 2d|3d] [additional gkyl options]"
    echo "Run the 'gkeyll' executable with the specified number of GPUs using srun."
    echo "This function automatically calculates the number of nodes needed."
    echo "Options:"
    echo "  --dim 2d|3d    Set dimensionality (default: 3d)"
    echo "Example: $0 8 --dim 2d your_input_file.lua"
    echo "Example: $0 8 your_input_file.lua"
    exit 1
fi

ngpu=$1
shift  # Shift the arguments to remove the first one (ngpu)

# Assuming 4 GPUs per node, calculate the number of nodes required.
gpus_per_node=4
nnodes=$(( (ngpu + gpus_per_node - 1) / gpus_per_node ))

# Default decomposition option
decomp="-e $ngpu"

# Parse arguments for dimensionality option
while [[ $# -gt 0 ]]; do
    case $1 in
        --dim)
            if [[ "$2" == "2d" || "$2" == "2D" || "$2" == "2" ]]; then
                decomp="-d $ngpu"
            elif [[ "$2" == "3d" || "$2" == "3D" || "$2" == "3" ]]; then
                decomp="-e $ngpu"
            else
                echo "Error: Invalid dimension '$2'. Use '2d' or '3d'."
                exit 1
            fi
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Use -N to specify the calculated number of nodes.
# --ntasks is set to the number of GPUs, assuming one task per GPU.
CMD="srun -N $nnodes --ntasks=$ngpu --gpus-per-task=1 --gpu-bind=closest -u ./gkeyll -g -M $decomp $* | tee gkyl_run_gpu.log"
echo "Executing command: $CMD"
eval $CMD