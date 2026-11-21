#!/bin/bash

#SBATCH -J tune_pmax
#SBATCH --output=logs/normal/%j_output.log
#SBATCH --error=logs/errors/%j_error.log

# Default: cm4_inter (5/6 nodes idle, 8h max limit; expected runtime ~15-25 min on 24 CPUs)
# Alternative if using serial cluster: --clusters=serial --partition=serial_std
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --mem=32G

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

srun python3 main.py --n_cpu "${SLURM_CPUS_PER_TASK:-4}" "$@"

end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script finished at: $end_time"

# Delete error log file at end of run if it is empty
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ERR_FILE="logs/errors/${SLURM_JOB_ID}_error.log"
  if [ -f "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ]; then
    rm "$ERR_FILE"
  fi
fi
