#!/bin/bash

# Display help message
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [GKYL_ARGS]

Run Gkeyll simulation on a Perlmutter GPU interactive session.
The script automatically detects free GPUs and creates a temporary slurm script to use them. 
This allows to run multiple simulations in parallel on the same node without interfering with each other.

Options:
  -d, --dim DIM          Dimensionality: 1d/1D/1, 2d/2D/2 or 3d/3D/3 (default: 3d)
  -e, --exec NAME        Executable name (default: gkeyll)
  -n, --ngpu NUM         Number of GPUs to use (default: 4)
  -G, --gpus IDX[,IDX]   Comma-separated GPU indices to use (skips auto-detection)
  -l, --log LOGNAME      Log file name (default: gkyl_run_gpu.log)
  -h, --help             Display this help message
  The remaining arguments are passed directly to the Gkeyll executable.

Example:
  $(basename "$0") -n 1 -d 2 input.lua          [auto-pick 1 free GPU, 2d decomp]
  $(basename "$0") -n 1 -G 2                    [force GPU 2]
  $(basename "$0") -n 2 -G 2,3 -e exec          [force GPUs 2 and 3]
  $(basename "$0") -n 4 -d 2 -r 40              [restart gkeyll at frame 40]
  $(basename "$0") --ngpu 4 --dim 3d --exec custom_gkeyll --log my_simulation.log [run a compiled .c input file]

EOF
}

# Find $needed free GPUs by checking nvidia-smi for compute processes.
# Prints a comma-separated list of free GPU indices, or exits with an error.
get_free_gpus() {
    local needed=$1
    local free_gpus=()
    local n_gpus
    n_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

    for i in $(seq 0 $((n_gpus - 1))); do
        local procs
        procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" 2>/dev/null)
        if [[ -z "$procs" ]]; then
            free_gpus+=("$i")
        fi
    done

    if [[ ${#free_gpus[@]} -lt $needed ]]; then
        echo "Error: Need $needed free GPU(s), but only ${#free_gpus[@]} are free." >&2
        nvidia-smi --query-gpu=index,gpu_name,memory.used --format=csv >&2
        return 1
    fi

    local selected=("${free_gpus[@]:0:$needed}")
    local IFS=','
    echo "${selected[*]}"
}

ngpu=4 # Default number of GPUs
gpus_per_node=4 # Perlmutter: 4 GPUs per node
decomp_dir="-e" # Default 3d decomposition
executable="gkeyll" # Default executable name
logname="gkyl_run_gpu.log" # Default log file name
manual_gpus=""  # Empty means auto-detect

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dim)
            if [[ "$2" == "1D" || "$2" == "1d" || "$2" == "1" ]]; then
                decomp_dir="-c"
            elif [[ "$2" == "2d" || "$2" == "2D" || "$2" == "2" ]]; then
                decomp_dir="-d"
            elif [[ "$2" == "3d" || "$2" == "3D" || "$2" == "3" ]]; then
                decomp_dir="-e"
            else
                echo "Error: Invalid dimension '$2'. Use '1d', '2d' or '3d'."
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
        -G|--gpus)
            manual_gpus="$2"
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

# Compute node count after ngpu is known
nnodes=$(( (ngpu + gpus_per_node - 1) / gpus_per_node ))

# Determine which GPUs to use
if [[ -n "$manual_gpus" ]]; then
    cuda_devices="$manual_gpus"
    echo "Using GPUs (manual): $cuda_devices"
else
    cuda_devices=$(get_free_gpus "$ngpu") || exit 1
    echo "Using GPUs (auto): $cuda_devices"
fi

# Write a temporary wrapper that re-applies our GPU selection inside the srun step,
# after Cray SLURM overwrites CUDA_VISIBLE_DEVICES in the step environment.
tmpwrapper=$(mktemp /tmp/gkyl_gpu_wrap_XXXXXX.sh)
trap "rm -f $tmpwrapper" EXIT
printf '#!/bin/bash\nexport CUDA_VISIBLE_DEVICES=%s\nexec "$@"\n' "$cuda_devices" > "$tmpwrapper"
chmod +x "$tmpwrapper"

CMD="srun --overlap -u -n $ngpu $tmpwrapper ./$executable -g -M $decomp_dir $ngpu $* | tee $logname"
echo "Using GPUs: $cuda_devices"
echo "Executing command: $CMD"
eval $CMD