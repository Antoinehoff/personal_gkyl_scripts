#!/bin/bash
#this script lists all the available frames in the specified folder
#it extracts the numbers just before .gkyl in the filenames
#it is useful for checking which frames are available in a simulation

# Check if a folder was provided as an argument, otherwise use the current directory
folder=${1:-"."}

# Use a more robust method similar to gkyl_find_last_frame.sh
frames=()

# Loop through all files matching the pattern
for file in "${folder}"/wk/*-ion_[0-9]*.gkyl; do
    # Check if the file exists
    if [[ -e "$file" ]]; then
        # Extract the numeric part before .gkyl
        num=$(basename "$file" .gkyl)
        num=${num##*_}
        # Check if the extracted part is a number
        if [[ $num =~ ^[0-9]+$ ]]; then
            frames+=($num)
        fi
    fi
done

# Sort and display the frames
if [ ${#frames[@]} -gt 0 ]; then
    printf '%s\n' "${frames[@]}" | sort -n | paste -sd ' ' -
else
    echo "No frames found"
fi