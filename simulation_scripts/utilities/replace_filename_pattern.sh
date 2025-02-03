#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: sh replace_pattern_filename.sh <old_string> <new_string>"
    exit 1
fi

old_string="$1"
new_string="$2"

# Loop through all files in the current directory
for file in *"$old_string"*; do
    # Generate the new filename by replacing the old string with the new string
    new_file="${file//$old_string/$new_string}"
    
    # Rename the file
    mv "$file" "$new_file" && echo "Renamed: $file -> $new_file"
done
