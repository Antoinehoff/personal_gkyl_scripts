#!/bin/bash
# This script lists all the available frames in the specified folder.
# It extracts the numbers just before .gkyl in the filenames.
# It is useful for checking which frames are available in a simulation.

# Check for help flag or set default directory
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [folder_path]"
    echo ""
    echo "Lists all available restart simulation frames in the specified folder."
    echo "Looks for files matching *-ion_<N>.gkyl and extracts the frame numbers."
    echo ""
    echo "Arguments:"
    echo "  folder_path   Path to the simulation folder (default: current directory)"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message and exit"
    exit 0
fi

folder="${1:-.}"
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