#!/bin/bash
# This script grants read and execute permissions to a user on a specific path if the author username is in the path.
# The script takes three arguments:
# 1. The username of the author
# 2. The username of the user to grant access to
# 3. The path to grant access to

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
  echo "Usage: grant_access.sh <username_author> <username_receiver> <path>"
  exit 1
fi

# Assign arguments to variables
USERNAME_AUTHOR=$1
USERNAME_RECEIVER=$2
TARGET_PATH=$3

# Split the TARGET_PATH and grant access to each level if author username is in the path
current_path=""
for dir in $(echo "$TARGET_PATH" | tr '/' ' '); do
  current_path="$current_path/$dir"
  
  # Check if author username is in the current path
  if [[ "$current_path" == *$USERNAME_AUTHOR* ]]; then
    setfacl -m u:"$USERNAME_RECEIVER":rx "$current_path"
    echo setfacl -m u:"$USERNAME_RECEIVER":rx "$current_path"
  fi
done
