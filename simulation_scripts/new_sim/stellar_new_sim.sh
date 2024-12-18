#!/bin/bash

# Function to handle name conflicts by appending a suffix
handle_name_conflict() {
    local base_name="$1"
    local suffix=1
    local new_name="$base_name"

    # Loop until we find a directory name that doesn't exist
    while [ -d "$new_name" ]; do
        new_name="${base_name}_${suffix}"
        suffix=$((suffix + 1))
    done

    echo "$new_name"
}

# Check if a parameter (simulation name) is provided
if [ -n "$1" ]; then
    # Use the provided name for the simulation directory
    new_dir="$1"
else
    # Initialize the highest number found to zero
    highest_num=0
    # Loop through all directories matching the pattern sim_XX
    for dir in sim_*; do
        if [[ -d "$dir" ]]; then
            # Extract the numeric part (XX) after sim_ from the directory name
            num=${dir#sim_}
            # Check if the extracted part is a valid number
            if [[ $num =~ ^[0-9]+$ ]]; then
                # Compare it with the highest number found so far
                if (( num > highest_num )); then
                    highest_num=$num
                fi
            fi
        fi
    done
    # Calculate the next number by adding 1 to the highest number
    next_num=$((highest_num + 1))
    # Format the number to ensure it has leading zeros (e.g., sim_01, sim_02)
    formatted_next_num=$(printf "%02d" "$next_num")
    # Set the new directory name as sim_XX
    new_dir="sim_$formatted_next_num"
fi

# Handle the case where the directory already exists
new_dir=$(handle_name_conflict "$new_dir")

# Create the new simulation directory
mkdir "$new_dir"
echo "Created directory: $new_dir"

# Add a README
cat <<EOT > $new_dir/README.txt
-- This folder is a framework to run a simulation with g0 --
- The submit.sh script is central and allows compile and run the code on the cluster.
        .The submit.sh script will, by default, look for the latest frame
         output in the wk/ directory and will restart the simulation from that frame.
        .The user can also specify a frame to restart from using -r N option (N the frame number)
        .One can use submit.sh -h to see the potential input parameters.
        .Finally, the submit.sh script names the simulation input and output according to the
         starting frame number, hence the creation of *_sf_##* file.
- The simulation parameters are to be set in the input.c file.
- The Makefile is already adapted to make a g0 executable from the input.c file.
- the wk/ directory is where the executable will run and output data.
How to run a simulation:
a) set up the simulation parameters in input.c
b) set up the parallelization and slurm job parameters in submit.sh
c) run submit.sh (sh submit.sh)
d) verify the slurm script and proceed with the job
e) submit.sh automatically make, copy the executable in /wk, go to /wk, submit the job in /wk
f) notes can (and should) be taken in the sim_log.txt to keep trace of the restart changes
EOT
cp $HOME/personal_gkyl_scripts/simulation_scripts/input_c/TCV_NT_tq_300eV_48x32x16x12x6_nuvg.c $new_dir/input.c
cp $HOME/personal_gkyl_scripts/simulation_scripts/Makefile $new_dir
cp $HOME/personal_gkyl_scripts/simulation_scripts/submit/stellar_submit.sh $new_dir
cp $HOME/personal_gkyl_scripts/example.ipynb $new_dir/$new_dir.ipynb
mkdir $new_dir/wk
mkdir $new_dir/history
echo "# This is a simulation logbook to keep track on restarts and changes" > $new_dir/sim_log.txt