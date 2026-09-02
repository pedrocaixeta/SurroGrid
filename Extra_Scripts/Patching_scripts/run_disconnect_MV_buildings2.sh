#!/bin/bash

#SBATCH -J disconnect_bldgs2
#SBATCH --output=logs/normal/%j_disconnect2_output.log
#SBATCH --error=logs/errors/%j_disconnect2_error.log

#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --mem-per-cpu=6200M

# Ensure the log directories exist locally before starting tasks (sbatch may need this)
mkdir -p logs/normal logs/errors

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate grid_alloc

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

srun python3 disconnect_MV_buildings2.py
wait

echo "Done!"
