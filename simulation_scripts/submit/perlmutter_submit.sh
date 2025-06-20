#!/bin/bash
set -e
#.SLURM PARAMETERS (user oriented)
#.Number of nodes to request.
NODES=1
#.Specify the QOS
QOS="regular"
#.Specify GPUs per node (Perlmutter has 4 GPUs per node).
GPU_PER_NODE=4
#.Request wall time.
TIME="22:00:00"  # HH:MM:SS
#.Mail is sent to you when the job starts and when it terminates or aborts.
EMAIL="ahoffman@pppl.gov"
#.Module to load.
MODULES="PrgEnv-gnu/8.5.0 craype-accel-nvidia80 cray-mpich/8.1.28 cudatoolkit/12.4 nccl/2.18.3-cu12"
#.Set the account.
ACCOUNT="m4564"
#.Default value to check a possible restart.
LAST_FRAME=-1 
#.Get the job name from the directory name
JOB_NAME=$(basename $(pwd))

# help function
show_help() {
    echo "Usage: $0 [-q QOS] [-n JOB_NAME] [-t TIME] [-h] [-d job_id]"
    echo "Submit a job to Perlmutter"
    echo "  -q QOS      QOS to use (default: regular)"
    echo "  -n JOB_NAME Name of the job (default: $JOB_NAME)"
    echo "  -t TIME     Wall time for the job (default: $TIME)"
    echo "  -N numnodes Number of nodes required (default: $NODES)"
    echo "  -r frame    Restart from frame (default: $LAST_FRAME)"
    echo "  -d job_id   Job ID to create a dependency (default: none)"
    echo "  -h          Display this help and exit"
    echo "  -a ACCOUNT  Account to use (default: $ACCOUNT)"
}
# check the following options : -q -n -t -h -N -r -d -a
while getopts ":q:n:t:r:N:d:a:h" opt; do
    case ${opt} in
        q )
            QOS=$OPTARG
            #if its debug add _dbg to jobname
            if [ "$QOS" == "debug" ]; then
                JOB_NAME="${JOB_NAME}_dbg"
                TIME="00:30:00"
            fi
            ;;
        n )
            JOB_NAME=$OPTARG
            ;;
        N )
            NODES=$OPTARG
            ;;
        t )
            TIME=$OPTARG
            ;;
        r )
            LAST_FRAME=$OPTARG
            ;;
        d )
            DEPENDENCY_JOB_ID=$OPTARG
            ;;
        a )
            ACCOUNT=$OPTARG
            ;;
        h )
            show_help
            exit 0
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done


#.AUXILIARY SLURM VARIABLES
#.Total number of cores/tasks/MPI processes.
NTASKS_PER_NODE=$GPU_PER_NODE
#.Calculate total number of GPUs (nodes * GPUs per node)
TOTAL_GPUS=$(( NODES * GPU_PER_NODE ))

#.------- DETECT 2D vs 3D RUN
# Check input.c file for cdim value to determine run type
if [[ -f "input.c" ]]; then
    # Extract cdim value from input.c
    CDIM=$(grep -E 'int\s+cdim\s*=\s*[0-9]+' input.c | sed -E 's/.*int\s+cdim\s*=\s*([0-9]+).*/\1/' | head -1)
    
    if [[ -z "$CDIM" ]]; then
        # Try alternative patterns for cdim declaration
        CDIM=$(grep -E '\.cdim\s*=\s*[0-9]+' input.c | sed -E 's/.*\.cdim\s*=\s*([0-9]+).*/\1/' | head -1)
    fi
    
    if [[ "$CDIM" == "2" ]]; then
        echo "Detected 2D run (cdim = 2)"
        RUN_TYPE="2D"
        GPU_OPTS="-c 1 -d $TOTAL_GPUS"
    elif [[ "$CDIM" == "3" ]]; then
        echo "Detected 3D run (cdim = 3)"
        RUN_TYPE="3D"
        GPU_OPTS="-c 1 -d 1 -e $TOTAL_GPUS"
    else
        echo "Warning: Could not determine cdim from input.c, defaulting to 3D"
        RUN_TYPE="3D"
        GPU_OPTS="-c 1 -d 1 -e $TOTAL_GPUS"
    fi
else
    echo "Warning: input.c not found, defaulting to 3D"
    RUN_TYPE="3D"
    GPU_OPTS="-c 1 -d 1 -e $TOTAL_GPUS"
fi

#.------- RESTART PREPARATION
# Only use LAST_FRAME if explicitly set via -r option
if (( LAST_FRAME >= 0 )); then
    echo "Using specified restart frame: $LAST_FRAME"
    RESTART_OPT="-r $LAST_FRAME"
    FRAME_SUFFIX="_$LAST_FRAME"
else
    echo "Will auto-detect last frame at runtime"
    RESTART_OPT=""
    # Use timestamp for unique naming to avoid overwrites
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    FRAME_SUFFIX="_$TIMESTAMP"
    LAST_FRAME="detect"
fi

#.Name format of output and error files.
if [[ "$LAST_FRAME" == "detect" ]]; then
    # Use timestamp initially, will create frame-specific links at runtime
    OUTPUT="output_$FRAME_SUFFIX.out"
    ERROR="error_$FRAME_SUFFIX.out"
else
    # We can use the specified LAST_FRAME for output names
    OUTPUT="../history/output_sf$FRAME_SUFFIX.out"
    ERROR="../history/error_sf$FRAME_SUFFIX.out"
fi

#.Run command - will be modified at runtime if auto-detecting
RUNCMD="srun -u -n $TOTAL_GPUS ./g0 -g -M $GPU_OPTS $RESTART_OPT"
SCRIPTNAME="slurm_script_sf$FRAME_SUFFIX.sh"
# Generate the SLURM script
cat <<EOT > $SCRIPTNAME
#!/bin/bash -l
#SBATCH --job-name $JOB_NAME
#SBATCH --qos $QOS
#SBATCH --nodes $NODES
#SBATCH --tasks-per-node=$NTASKS_PER_NODE
#SBATCH --time $TIME
#SBATCH --constraint gpu
#SBATCH --gpus $TOTAL_GPUS
#SBATCH --mail-user=$EMAIL
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --output $OUTPUT
#SBATCH --error $ERROR
#SBATCH --account $ACCOUNT
EOT

if [ -n "$DEPENDENCY_JOB_ID" ]; then
    echo "#SBATCH --dependency=afterany:$DEPENDENCY_JOB_ID" >> $SCRIPTNAME
fi

# Add module load and environment setup
cat <<EOT >> $SCRIPTNAME
# Load necessary modules
module load $MODULES
export MPICH_GPU_SUPPORT_ENABLED=0
export DVS_MAXNODES=32_
export MPICH_MPIIO_DVS_MAXNODES=32

# Run type: $RUN_TYPE
EOT

# Add runtime frame detection logic if needed
if [[ "$LAST_FRAME" == "detect" ]]; then
cat <<EOT >> $SCRIPTNAME
# Auto-detect last frame at runtime using utility script
SCRIPT_DIR=\$(dirname "\$0")
UTIL_SCRIPT="\$HOME/personal_gkyl_scripts/simulation_scripts/utilities/gkyl_find_last_frame.sh"

if [[ ! -f "\$UTIL_SCRIPT" ]]; then
    echo "Error: gkyl_find_last_frame.sh utility script not found at \$UTIL_SCRIPT"
    exit 1
fi

LAST_FRAME=\$(sh \$UTIL_SCRIPT .)

# Create frame-specific output file links after detecting frame
if (( LAST_FRAME >= 0 )); then
    # Create symbolic links with frame-specific names
    FRAME_OUTPUT="../history/output_sf_\$LAST_FRAME.out"
    FRAME_ERROR="../history/error_sf_\$LAST_FRAME.out"
    
    # Create links to the actual SLURM output files
    ln -sf "output_sf$FRAME_SUFFIX.out" "\$FRAME_OUTPUT"
    ln -sf "error_sf$FRAME_SUFFIX.out" "\$FRAME_ERROR"
    
    echo "Created output links: \$FRAME_OUTPUT and \$FRAME_ERROR"
fi

# Set restart options based on detected frame
if (( LAST_FRAME > 0 )); then
    if compgen -G "*_\$LAST_FRAME.gkyl" > /dev/null; then
        echo "Runtime: Restart from frame \$LAST_FRAME"
        RESTART_OPT="-r \$LAST_FRAME"
    else
        echo "Runtime: Frame \$LAST_FRAME not found"
        exit 1
    fi
else
    echo "Runtime: Start simulation from 0"
    RESTART_OPT=""
fi

srun -u -n $TOTAL_GPUS ./g0 -g -M $GPU_OPTS \$RESTART_OPT
exit 0
EOT
else
cat <<EOT >> $SCRIPTNAME
$RUNCMD
exit 0
EOT
fi

echo "---------------------------------"
echo "The script generated is"
echo "---------------------------------"
cat $SCRIPTNAME
echo "---------------------------------"

# Ask the user if they want to proceed
read -p "Proceed and submit the job? ((y)/n) " proceed

if [[ "$proceed" == "" || "$proceed" == "y" ]]; then
    module load $MODULES
    # make only if there is no job dependency
    if [ -z "$DEPENDENCY_JOB_ID" ]; then
        make
    fi
    mkdir -p history
    cp input.c history/input_sf$FRAME_SUFFIX.c
    cp $SCRIPTNAME wk/.
    mv $SCRIPTNAME history/.
    cd wk
    sbatch $SCRIPTNAME
else
    echo "Operation canceled."
fi
