#!/bin/bash
#this script lists all the available frames in the specified folder
#it extracts the numbers just before .gkyl in the filenames
#it is useful for checking which frames are available in a simulation

# Check if a folder was provided as an argument, otherwise use the current directory
folder=${1:-"."}

ls ${folder}/wk/*field*.gkyl > ls.tmp

# Extract all numbers just before .gkyl
grep '\.gkyl$' "ls.tmp" | sed -n 's/.*_\([0-9]\+\)\.gkyl$/\1/p' | sort -n | paste -sd ' ' -

rm ls.tmp