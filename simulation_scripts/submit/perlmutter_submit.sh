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
TIME="06:00:00"  # HH:MM:SS
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
#.Executable name
INPUT_FILE="input.c"

# help function
show_help() {
    echo "Usage: $0 [-q QOS] [-n JOB_NAME] [-t TIME] [-h] [-d job_id] [-j] [-p]"
    echo "Submit a job to Perlmutter"
    echo "  -q QOS      QOS to use (default: regular)"
    echo "  -n JOB_NAME Name of the job (default: $JOB_NAME)"
    echo "  -t TIME     Wall time for the job (default: $TIME)"
    echo "  -N numnodes Number of nodes required (default: $NODES)"
    echo "  -r frame    Restart from frame (default: $LAST_FRAME)"
    echo "  -d job_id   Job ID to create a dependency (default: none)"
    echo "  -j          Auto-detect and use most recent job ID for dependency"
    echo "  -p          Print script and exit (do not submit)"
    echo "  -h          Display this help and exit"
    echo "  -a ACCOUNT  Account to use (default: $ACCOUNT)"
    echo "  -i INPUTC   C Input file name (default: input.c)"
}
# check the following options : -q -n -t -h -N -r -d -a -j -p
while getopts ":q:n:t:r:N:d:a:i:jph" opt; do
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
        j )
            # Auto-detect job dependency
            if [ -d "history" ]; then
                LATEST_JOB_FILE=$(ls -t history/job_id_*.txt 2>/dev/null | head -1)
                if [ -n "$LATEST_JOB_FILE" ] && [ -f "$LATEST_JOB_FILE" ]; then
                    AUTO_DEPENDENCY=$(cat "$LATEST_JOB_FILE" 2>/dev/null | grep -o '[0-9]\+' | head -1)
                    if [ -n "$AUTO_DEPENDENCY" ]; then
                        DEPENDENCY_JOB_ID="$AUTO_DEPENDENCY"
                        echo "Auto-detected job dependency: $DEPENDENCY_JOB_ID"
                    else
                        echo "Warning: Could not extract job ID from $LATEST_JOB_FILE"
                    fi
                else
                    echo "Warning: No job_id files found in history directory"
                fi
            else
                echo "Warning: history directory not found"
            fi
            ;;
        p )
            PRINT_ONLY=1
            ;;
        a )
            ACCOUNT=$OPTARG
            ;;
        i )
            INPUT_FILE=$OPTARG
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

#.Get the executable name from the input file (strip .c)
EXEC_NAME=$(basename "$INPUT_FILE" .c)
GKYLEXE="gkeyll"

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
        # echo "Detected 2D run (cdim = 2)"
        RUN_TYPE="2D"
        GPU_OPTS="-c 1 -d $TOTAL_GPUS"
    elif [[ "$CDIM" == "3" ]]; then
        # echo "Detected 3D run (cdim = 3)"
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
else
    echo "Will auto-detect last frame at runtime"
    RESTART_OPT=""
    LAST_FRAME="detect"
fi

# Use timestamp for unique naming to avoid overwrites
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FRAME_SUFFIX="$TIMESTAMP"

# Use timestamp initially, will create frame-specific links at runtime
INPUT="input_$FRAME_SUFFIX.c"
SLURM_INPUT="slurm_script_$FRAME_SUFFIX.sh"
OUTPUT="output_$FRAME_SUFFIX.out"
ERROR="error_$FRAME_SUFFIX.out"

#.Run time option.
# get time in seconds
IFS=':' read -r h m s <<< "$TIME"
seconds=$((10#$h * 3600 + 10#$m * 60 + 10#$s))
RUNTIME_OPT="-o max_run_time=$seconds"

#.Run command - will be modified at runtime if auto-detecting
RUNCMD="srun -u -n $TOTAL_GPUS ./$GKYLEXE -g -M $GPU_OPTS $RESTART_OPT"
SCRIPTNAME="slurm_script_$FRAME_SUFFIX.sh"
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

# Create links to the actual SLURM output files
ln -sf "../wk/output_$FRAME_SUFFIX.out" "../history/output_sf_\$LAST_FRAME.out"
ln -sf "../wk/error_$FRAME_SUFFIX.out" "../history/error_sf_\$LAST_FRAME.out"
ln -sf "../wk/input_$FRAME_SUFFIX.c" "../history/input_sf_\$LAST_FRAME.c"
ln -sf "../wk/slurm_script_$FRAME_SUFFIX.sh" "../history/slurm_script_sf_\$LAST_FRAME.sh"

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

SRUNCMD="srun -u -n $TOTAL_GPUS ./$GKYLEXE -g -M $GPU_OPTS \$RESTART_OPT $RUNTIME_OPT"

echo "Running command: \$SRUNCMD"
eval \$SRUNCMD
exit 0
EOT
else
cat <<EOT >> $SCRIPTNAME
$RUNCMD
exit 0
EOT
fi

# Exit early if print-only mode
if [ "$PRINT_ONLY" = "1" ]; then
    echo "---------------------------------"
    echo "The script generated is"
    echo "---------------------------------"
    cat $SCRIPTNAME
    echo "---------------------------------"
    exit 0
fi

# Ask the user if they want to proceed
read -p "Proceed and submit the job? ((y)/n) " proceed

if [[ "$proceed" == "" || "$proceed" == "y" ]]; then
    module load $MODULES
    # make only if there is no job dependency
    if [ -z "$DEPENDENCY_JOB_ID" ]; then
        make
    fi
    mkdir -p history
    mkdir -p wk
    cp $INPUT_FILE wk/input_$FRAME_SUFFIX.c
    mv $SCRIPTNAME wk/.
    cp $EXEC_NAME wk/$GKYLEXE
    cd wk
    
    # Submit job and capture the job ID
    SUBMIT_OUTPUT=$(sbatch $SCRIPTNAME)
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -o '[0-9]\+')
    
    echo "$SUBMIT_OUTPUT"
    
    # Save job ID to file for reference
    echo "$JOB_ID" > "../history/job_id_${JOB_ID}.txt"
    echo "Job ID saved to: job_id_${JOB_ID}.txt"
else
    echo "Operation canceled."
fi
