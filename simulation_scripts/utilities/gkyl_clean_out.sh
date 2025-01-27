#!/bin/bash
# this script deletes all files matching the pattern '*I1_XX.gkyl' where XX is between I2 and I3
# it is useful for cleaning up the output files from a simulation

# Function to display the help message
show_help() {
  echo "Usage: $0 I1 I2 I3"
  echo ""
  echo "Deletes all files matching the pattern '*I1_XX.gkyl' where XX is between I2 and I3."
  echo ""
  echo "Arguments:"
  echo "  I1    Prefix of the files (e.g., D02 in 'D02_XX.gkyl')"
  echo "  I2    Start of the range (e.g., 10)"
  echo "  I3    End of the range (e.g., 20)"
  echo ""
  echo "Example:"
  echo "  $0 D02 10 20"
  echo "  This will delete all files matching the pattern '*D02_XX.gkyl' where XX is between 10 and 20."
  exit 0
}

# Check if -h option is provided
if [[ "$1" == "-h" ]]; then
  show_help
fi

# Check if the required inputs are provided
if [ "$#" -ne 3 ]; then
  echo "Error: Incorrect number of arguments."
  echo "Use -h for help."
  exit 1
fi

# Assign input arguments to variables
I1=$1
I2=$2
I3=$3

# Ensure I2 and I3 are numbers
if ! [[ "$I2" =~ ^[0-9]+$ ]] || ! [[ "$I3" =~ ^[0-9]+$ ]]; then
  echo "Error: I2 and I3 must be numbers."
  exit 1
fi

# Loop over the range from I2 to I3
for ((i=I2; i<=I3; i++)); do
  # Format the file pattern
  file_pattern="*${I1}_${i}.gkyl"

  # Check if any files match the pattern
  if ls $file_pattern 1> /dev/null 2>&1; then
    echo "Deleting files matching: $file_pattern"
    rm $file_pattern
  else
    echo "No files found for pattern: $file_pattern"
  fi
done