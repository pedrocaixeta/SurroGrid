#!/bin/bash

#SBATCH -J extract_peaks
#SBATCH --output=logs/normal/%j_extract_peaks.log
#SBATCH --error=logs/errors/%j_extract_peaks.log

#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-01:30:00
#SBATCH --mem-per-cpu=4000M

# Ensure log directories exist
mkdir -p logs/normal logs/errors

# Load Miniforge conda environment
module load miniforge3
module list

# Hook conda into bash and activate the environment
eval "$(conda shell.bash hook)"
conda activate pwrflw-hpc

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Extract script started at: $start_time"

python3 extract_aggregated_peaks.py

echo "Finished at: $(date +'%Y-%m-%d %H:%M:%S')"
