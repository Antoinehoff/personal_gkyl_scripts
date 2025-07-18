#!/bin/bash
set -e
#.SLURM PARAMETERS (user oriented)
#.Request the queue. See stellar docs online.
QOS="pppl-short"
#.Number of nodes to request (Stellar has 96 cores and 2 GPUs per node).
NODES=1
#.Specify GPUs per node (stellar-amd has 2):
GPU_PER_NODE=2
#.Request wall time
TIME="22:00:00"  # HH:MM:SS
#.Mail is sent to you when the job starts and when it terminates or aborts.
EMAIL="ahoffman@pppl.gov"
#.Module to load
MODULES="cudatoolkit/12.0 openmpi/cuda-11.1/gcc/4.1.1"

#.AUXILIARY SLURM VARIABLES
#.Total number of cores/tasks/MPI processes.
NTASKS_PER_NODE=$GPU_PER_NODE
#.Calculate total number of GPUs (nodes * GPUs per node)
TOTAL_GPUS=$(( NODES * GPU_PER_NODE ))
# default value to check a possible restart
LAST_FRAME=-1 
#.Get the job name from the directory name
JOB_NAME=$(basename $(pwd))

# help function
show_help() {
    echo "Usage: $0 [-q QOS] [-n JOB_NAME] [-t TIME] [-h]"
    echo "Submit a job to Perlmutter"
    echo "  -q QOS      QOS to use (default: regular)"
    echo "  -n JOB_NAME Name of the job (default: $JOB_NAME)"
    echo "  -t TIME     Wall time for the job (default: $TIME)"
    echo "  -N numnodes Number of nodes required (default: $NODES)"
    echo "  -h          Display this help and exit"
}
# check the following options : -q -n -t -h
while getopts ":q:n:t:r:N:h" opt; do
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

if (( LAST_FRAME < 0 )); then
    #.Find the most recent frame for restart using utility script
    UTIL_SCRIPT="$HOME/personal_gkyl_scripts/simulation_scripts/utilities/gkyl_find_last_frame.sh"
    
    if [[ -f "$UTIL_SCRIPT" ]]; then
        LAST_FRAME=$($UTIL_SCRIPT wk)
    else
        echo "Warning: gkyl_find_last_frame.sh utility script not found, using fallback method"
        # Fallback to original method
        for file in wk/*.gkyl; do
            # Check if the file exists to avoid errors when no .gkyl files are present
            if [[ -e "$file" ]]; then
                # Extract the numeric part before .gkyl (assuming it's the last part of the filename)
                num=$(basename "$file" .gkyl)   # Remove the .gkyl extension
                num=${num##*_}                  # Extract the part after the last underscore
                # Check if the extracted part is a number and compare it
                if [[ $num =~ ^[0-9]+$ ]]; then
                    if (( num > LAST_FRAME )); then
                        LAST_FRAME=$num
                    fi
                fi
            fi
        done
    fi
fi

#.If a frame has been found, set a restart
if (( LAST_FRAME > 0 )); then
    if compgen -G "wk/*_$LAST_FRAME.gkyl" > /dev/null; then
        echo "Restart from frame $LAST_FRAME"
        RESTART_OPT="-r $LAST_FRAME"
    else
        echo "Frame $LAST_FRAME not found"
        exit 1
    fi
else
    echo "Start simulation from 0"
    LAST_FRAME=0
    RESTART_OPT=
fi

#.Name format of output and error files.
OUTPUT="../history/output_sf_$LAST_FRAME.out"
ERROR="../history/error_sf_$LAST_FRAME.out"

#.Run command
RUNCMD="mpirun -np $TOTAL_GPUS ./g0 -g -M -c 1 -d 1 -e $TOTAL_GPUS $RESTART_OPT"
SCRIPTNAME="slurm_script_sf_$LAST_FRAME.sh"
# Generate the SLURM scripta
cat <<EOT > $SCRIPTNAME
#!/bin/bash -l
#SBATCH --job-name $JOB_NAME
#SBATCH --qos $QOS
#SBATCH --nodes $NODES
#SBATCH --tasks-per-node=$NTASKS_PER_NODE
#SBATCH --time $TIME
#SBATCH --gres=gpu:$TOTAL_GPUS
#SBATCH --mail-user=$EMAIL
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --output $OUTPUT
#SBATCH --error $ERROR
module load $MODULES
$RUNCMD
exit 0
EOT
echo "---------------------------------"
echo "The script generated is"
echo "---------------------------------"
cat $SCRIPTNAME
echo "---------------------------------"

# Ask the user if they want to proceed
read -p "Proceed and submit the job? ((y)/n) " proceed

if [[ "$proceed" == "" || "$proceed" == "y" ]]; then
    module load $MODULES
    make
    mkdir -p history
    cp input.c history/input_sf_$LAST_FRAME.c
    cp $SCRIPTNAME wk/.
    mv $SCRIPTNAME history/.
    cd wk
    sbatch $SCRIPTNAME
else
    echo "Operation canceled."
fi
