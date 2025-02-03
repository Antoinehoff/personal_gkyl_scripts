#!/bin/bash
# Replace a pattern in all filenames in the current directory
# Usage: ./replace_pattern_filename.sh <pattern> <replacement>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <pattern> <replacement>"
    exit 1
fi

pattern=$1
replacement=$2

for file in *"$pattern"*; do
    new_file=$(echo "$file" | sed "s/$pattern/$replacement/g")
    mv "$file" "$new_file"
done