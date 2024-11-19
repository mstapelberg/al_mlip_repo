#!/bin/bash 
#
#SBATCH --job-name=run_aa
#SBATCH --output=aa_gen_%j.out
#SBATCH --error=aa_gen_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --time=5-06:00:00
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh

# Activate the 'allegro' environment
conda activate mace-11.7

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=10


# Run the script
python run_aa.py
