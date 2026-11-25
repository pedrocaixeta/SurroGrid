#!/bin/bash

#SBATCH -J inf_dmd_alloc
#SBATCH --output=logs/normal/%j_output.log
#SBATCH --error=logs/errors/%j_error.log

# Default: cm4_inter (8h max walltime limit; 1 grid takes ~2-10 min on 4 CPUs)
# Alternative if using serial cluster: --clusters=serial --partition=serial_std
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:40:00
#SBATCH --mem=8G

set -euo pipefail

mkdir -p logs/normal logs/errors

module load miniforge3
module list

eval "$(conda shell.bash hook)"
conda activate inference
conda env list

# Prevent nested BLAS/OpenMP threads from oversubscribing multiprocessing workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

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
