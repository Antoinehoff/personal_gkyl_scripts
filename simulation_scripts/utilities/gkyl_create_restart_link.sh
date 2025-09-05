#!/bin/bash

# filepath: /path/to/script.sh

# Check if a folder path is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path-to-folder>"
  exit 1
fi

folder_path="$1"

# Use gkyl_find_last_frame.sh to find the last frame for ion files
UTIL_SCRIPT="$(dirname "$0")/gkyl_find_last_frame.sh"

if [[ ! -f "$UTIL_SCRIPT" ]]; then
  echo "Error: gkyl_find_last_frame.sh utility script not found at $UTIL_SCRIPT"
  exit 1
fi

last_frame=$(bash "$UTIL_SCRIPT" "$folder_path")

# Check if a valid frame was found
if [[ "$last_frame" == "-1" ]]; then
  echo "No matching ion frames found in the folder."
fi

# Resolve the full filenames for ion and elc files
ion_file=$(find "$folder_path" -name "*ion_$last_frame.gkyl" | head -n 1)
elc_file=$(find "$folder_path" -name "*elc_$last_frame.gkyl" | head -n 1)

# Check if the files exist
if [[ -z "$ion_file" || -z "$elc_file" ]]; then
  echo "Could not find matching ion or elc files for frame $last_frame."
else
  # Create symbolic links for the last frame
  ln -s "$ion_file" restart-ion.gkyl
  ln -s "$elc_file" restart-elc.gkyl

  echo "Symbolic links created:"
  echo "restart-ion.gkyl -> $ion_file"
  echo "restart-elc.gkyl -> $elc_file"
fi

# Treat also the jacobtot_inv.gkyl file
jacobtot_file=$(find "$folder_path" -name "*jacobtot_inv.gkyl" | head -n 1)
if [[ -n "$jacobtot_file" ]]; then
  ln -s "$jacobtot_file" restart-jacobtot_inv.gkyl
  echo "restart-jacobtot_inv.gkyl -> $jacobtot_file"
else
  echo "No matching jacobtot_inv file found."
fi