#!/bin/bash
set -e
#.SLURM PARAMETERS (user oriented)
#.Declare a name for this job, preferably with 16 or fewer characters.
JOB_NAME="gk_tcv_3x2v_AH"
#.Request the queue. See stellar docs online.
QOS="pppl-short"
#.Number of nodes to request (Stellar has 96 cores and 2 GPUs per node).
NODES=2
#.Specify GPUs per node (stellar-amd has 2):
GPU_PER_NODE=2
#.Request wall time
TIME="10:00:00"  # HH:MM:SS
#.Mail is sent to you when the job starts and when it terminates or aborts.
EMAIL="ahoffman@pppl.gov"
#.Module to load
MODULES="cudatoolkit/12.4 openmpi/cuda-11.1/gcc/4.1.1"

#.AUXILIARY SLURM VARIABLES
#.Total number of cores/tasks/MPI processes.
NTASKS_PER_NODE=$GPU_PER_NODE
#.Calculate total number of GPUs (nodes * GPUs per node)
TOTAL_GPUS=$(( NODES * GPU_PER_NODE ))

#.------- RESTART HELPER
#.Default value
LAST_FRAME=-1
# Function to display help
show_help() {
    echo "Usage: $0 [-r N] [-h]"
    echo ""
    echo "Options:"
    echo "  -r N      Restart the simulation from frame N (optional)."
    echo "  -h        Display this help message."
}
# Parse command-line options
while getopts ":r:h" opt; do
    case ${opt} in
        r )
            LAST_FRAME=$OPTARG
            ;;
        h )
            show_help
            exit 0
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
    esac
done
if (( LAST_FRAME < 0 )); then
    #.Find the most recent frame for restart
    # Loop through all files ending with .gkyl in the current directory
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