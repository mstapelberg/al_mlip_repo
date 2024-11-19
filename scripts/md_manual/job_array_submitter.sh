#!/bin/bash

# Get number of xyz files
NUM_FILES=$(ls ../../data/zr-w-v-ti-cr/gen_5_2024-11-17/analysis_output/structures/*.xyz | wc -l)
NUM_JOBS=$((NUM_FILES - 1))

# Submit the array job with max 12 concurrent tasks
sbatch --array=0-${NUM_JOBS}%12 submit_parallel_md.sh