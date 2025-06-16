#!/bin/bash

# Script to find the last frame number from .gkyl files
# Usage: gkyl_find_last_frame.sh [directory]
# If no directory is provided, uses current directory
# Returns the highest frame number found, or -1 if no frames are found

show_help() {
    echo "Usage: $0 [directory]"
    echo "Find the last frame number from *-ion_*.gkyl files"
    echo "  directory   Directory to search (default: current directory)"
    echo "  -h          Display this help and exit"
    echo ""
    echo "Returns: Highest frame number found, or -1 if no frames found"
    exit 0
}

# Check if -h option is provided
if [[ "$1" == "-h" ]]; then
    show_help
fi

# Set directory to search (default to current directory)
SEARCH_DIR=${1:-.}

# Initialize last frame to -1
LAST_FRAME=-1

# Loop through all files matching the pattern
for file in "$SEARCH_DIR"/*-ion_[0-9]*.gkyl; do
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

# Output the result
echo $LAST_FRAME
