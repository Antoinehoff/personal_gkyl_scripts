#!/bin/bash
#this script calculates the total runtime of all the slurm scripts in the specified folder
#it multiplies the runtime by the number of nodes specified in the slurm script

# Check if a folder was provided as an argument, otherwise use the current directory
folder=${1:-"."}

# Initialize total time in seconds
total_seconds=0

# Loop through all slurm_script_sf_*.sh files in the specified folder
for file in "$folder"/history/slurm_script_sf_*.sh; do
    # Extract the time (HH:MM:SS) from the #SBATCH --time line
    time=$(grep "#SBATCH --time" "$file" | awk '{print $3}')
    
    # Extract the number of nodes from the #SBATCH --nodes line
    nodes=$(grep "#SBATCH --nodes" "$file" | awk '{print $3}')
    
    # Default to 1 node if not specified
    nodes=${nodes:-1}
    
    # Check if time was found
    if [[ -n "$time" ]]; then
        # Split the time into hours, minutes, and seconds
        IFS=: read -r hours minutes seconds <<< "$time"
        
        # Convert the time to total seconds
        seconds_in_file=$((hours * 3600 + minutes * 60 + seconds))
        
        # Multiply the time by the number of nodes
        total_seconds_in_file=$((seconds_in_file * nodes))
        
        # Add to the total time
        total_seconds=$((total_seconds + total_seconds_in_file))
    fi
done

# Convert the total time in seconds back to HH:MM:SS format
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))
seconds=$((total_seconds % 60))

# Print the total time
printf "Total time (multiplied by nodes): %02d:%02d:%02d\n" $hours $minutes $seconds
