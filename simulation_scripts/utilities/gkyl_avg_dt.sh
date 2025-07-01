#!/bin/bash

# Check if a file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file="$1"

# Extract dt values, sum them, and count them
awk '
/Taking time-step at t = .* dt = [0-9.eE+-]+ mus/ {
    for(i=1;i<=NF;i++) {
        if ($i == "dt") {
            dt=$(i+2)
            sum += dt
            count++
        }
    }
}
END {
    if (count > 0) {
        print "Average dt:", sum/count, "mus"
    } else {
        print "No dt values found."
    }
}
' "$input_file"