#!/bin/bash

#SBATCH -J inf_direct
#SBATCH --output=logs/normal/%j_output.log
#SBATCH --error=logs/errors/%j_error.log

#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:20:00
#SBATCH --mem=6G

set -euo pipefail

mkdir -p logs/normal logs/errors

module load miniforge3
module list

eval "$(conda shell.bash hook)"
conda activate inference
conda env list

# Prevent HDF5 file-locking issues on DSS and avoid writing __pycache__ files.
# Per-stage OMP/MKL/OPENBLAS thread counts are set cleanly inside main.py:
#   - Steps 1 & 2: 1 BLAS thread per worker (avoids multiprocessing oversubscription)
#   - Step 3:      SLURM_CPUS_PER_TASK threads for CPU PyTorch forward pass
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONDONTWRITEBYTECODE=1

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

INDEX="$1"
echo "Fileindex: $INDEX"

srun python3 main.py "$INDEX" --n_cpu "${SLURM_CPUS_PER_TASK:-1}"

end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script finished at: $end_time"

# Delete error log file at end of run if it is empty; otherwise append the file index
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ERR_FILE="logs/errors/${SLURM_JOB_ID}_error.log"
  if [ -f "$ERR_FILE" ]; then
    if [ ! -s "$ERR_FILE" ]; then
      rm "$ERR_FILE"
    else
      echo "Fileindex: $INDEX" >> "$ERR_FILE"
    fi
  fi
fi
