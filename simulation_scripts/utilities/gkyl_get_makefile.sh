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
cat <<'EOF' >> "$OUTPUT"
all: input setup

input: input.c
	${CC} ${CFLAGS} ${INCLUDES} input.c -o g0 -L${G0_LIB_DIR} ${G0_RPATH} ${G0_LIBS} ${LIB_DIRS} ${EXT_LIBS}

setup: g0
	mkdir -p wk
	cp g0 wk

clean:
	rm -rf g0 g0.dSYM wk/g0
EOF

echo "The Makefile has been successfully created."