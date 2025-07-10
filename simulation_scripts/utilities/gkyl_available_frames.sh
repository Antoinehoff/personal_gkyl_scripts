#!/bin/bash
# This script lists all the available frames in the specified folder.
# It extracts the numbers just before .gkyl in the filenames.
# It is useful for checking which frames are available in a simulation.

# Check if a folder was provided as an argument
if [ -z "$1" ]; then
    echo "Error: No folder path provided."
    echo "Usage: $0 <folder_path>"
    exit 1
fi

folder="$1"
frames=()

# Loop through all files matching the pattern
for file in "${folder}"/*-ion_[0-9]*.gkyl; do
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