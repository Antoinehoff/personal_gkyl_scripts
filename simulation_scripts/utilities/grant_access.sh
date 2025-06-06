#!/bin/bash
# This script grants read and execute permissions to a user on a specific path if the author username is in the path.
# The script takes three arguments:
# 1. The username of the author
# 2. The username of the user to grant access to
# 3. The path to grant access to
# Authors: A.C.D. Hoffmann & T.N. Bernard, 2025

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
  echo "Usage: grant_access.sh <username_author> <username_receiver> <path>"
  exit 1
fi

# Assign arguments to variables
USERNAME_AUTHOR=$1
USERNAME_RECEIVER=$2
TARGET_PATH=$(realpath "$3")  # Normalize the full path

# Initialize path and split into components
current_path=""
IFS='/' read -ra DIRS <<< "$TARGET_PATH"

for dir in "${DIRS[@]}"; do
  if [ -n "$dir" ]; then
    current_path="$current_path/$dir"

    # Check if author username is in the current path
    if [[ "$current_path" == *"$USERNAME_AUTHOR"* ]]; then
      normalized_current=$(realpath "$current_path" 2>/dev/null)

      if [[ "$normalized_current" == "$TARGET_PATH" ]]; then
        setfacl -R -m u:"$USERNAME_RECEIVER":rx "$current_path"
        echo "setfacl -R -m u:$USERNAME_RECEIVER:rx $current_path"
      else
        setfacl -m u:"$USERNAME_RECEIVER":rx "$current_path"
        echo "setfacl -m u:$USERNAME_RECEIVER:rx $current_path"
      fi
    fi
  fi
done