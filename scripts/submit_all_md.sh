#!/bin/bash

# Check if input file base name is provided
if [ $# -ne 1 ]; then
    echo "Usage: ./submit_all_md.sh <base_xyz_filename>"
    echo "Example: ./submit_all_md.sh your_structures"
    exit 1
fi

BASE_NAME=$1
DELAY=3  # Delay in seconds between submissions

# Submit jobs with delay
for i in {0..11}; do
    echo "Submitting job for ${BASE_NAME}_part${i}.xyz"
    sbatch submit_md.sh "${BASE_NAME}_part${i}.xyz"
    echo "Waiting ${DELAY} seconds before next submission..."
    sleep $DELAY
done

echo "All jobs submitted!"
