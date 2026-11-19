#!/bin/bash

#SBATCH -J disconnect_bldgs
#SBATCH --output=logs/normal/%j_disconnect_output.log
#SBATCH --error=logs/errors/%j_disconnect_error.log

#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --mem-per-cpu=6200M

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate pwrflw-hpc

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

srun python3 disconnect_MV_buildings.py
wait

echo "Done!"
