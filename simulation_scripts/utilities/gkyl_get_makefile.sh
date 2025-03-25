#!/bin/bash

# Check if the user provided the path to the gkylsoft directory
if [ -z "$1" ]; then
    echo "Usage: ./gkyl_adapt_makefile.sh /path/to/gkylsoft"
    exit 1
fi

# Define the INPUT as the Makefile located in the gkylsoft directory
INPUT="$1/gkylzero/share/Makefile"
OUTPUT="Makefile"

# Find the line number where 'all:' starts in Makefile.in
LINE_NUM=$(grep -n '^all:' "$INPUT" | cut -d: -f1 | head -n 1)

# If 'all:' is not found, exit with an error
if [ -z "$LINE_NUM" ]; then
    echo "Error: 'all:' target not found in $INPUT"
    exit 1
fi

# Write the unchanged part to the output
head -n $((LINE_NUM - 1)) "$INPUT" > "$OUTPUT"

# Manually append the modified content
cat <<EOF >> "$OUTPUT"
all:
	# Custom build rules go here
	echo "Building project..."
EOF

echo "The Makefile has been successfully created."