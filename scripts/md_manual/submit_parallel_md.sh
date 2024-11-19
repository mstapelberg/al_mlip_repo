#!/bin/bash
#
#SBATCH --job-name=run_md_gen
#SBATCH --output=md_gen_%A_%a.out
#SBATCH --error=md_gen_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=5-06:00:00
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh
# Activate the 'allegro' environment
conda activate mace-11.7

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=10

# Directory containing input structures
INPUT_DIR="../../data/zr-w-v-ti-cr/gen_5_2024-11-17/analysis_output/structures"
# Directory for output trajectories
OUTPUT_DIR="../../data/zr-w-v-ti-cr/gen_5_2024-11-17/md_output"
# Model directory
MODEL_DIR="../../Models/zr-w-v-ti-cr/gen_4_2024-11-15/"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Get array of xyz files
mapfile -t XYZ_FILES < <(ls ${INPUT_DIR}/*.xyz)
NUM_FILES=${#XYZ_FILES[@]}

# Get the file for this array task
INPUT_FILE="${XYZ_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Processing file: $INPUT_FILE"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Total number of files: $NUM_FILES"

# Run the Python script
python run_parallel_md.py \
    --input "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_dir "$MODEL_DIR" \
    --task_id "$SLURM_ARRAY_TASK_ID"