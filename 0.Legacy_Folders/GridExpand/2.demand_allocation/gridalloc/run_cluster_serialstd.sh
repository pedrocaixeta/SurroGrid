#!/bin/bash

#SBATCH -J test_run
#SBATCH --output=logs/normal/%j_output.log
#SBATCH --error=logs/errors/%j_error.log

#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-09:00:00
#SBATCH --mem=2G

module load miniforge3 # loads python into the cluster
module list # presents the dependencies in the environment

eval "$(conda shell.bash hook)" # Is in the local run: creates the local environment
conda activate grid_alloc # Is in the local run: activates the local environment
conda env list # prints the dependencies on the screen

# prints the start time in the screen
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

INDEX="$1"
echo "Fileindex: $INDEX"

srun python3 main.py $INDEX --n_cpu $SLURM_CPUS_PER_TASK # Is in the local run: runs main.py with INDEX cpus. When no file for a single grid is passed, all the disponible files get processed
wait

### Delete error log file at end of run if it is empty, otherwise append the file index
ERR_FILE="logs/errors/${SLURM_JOB_ID}_error.log"
if [ -f "$ERR_FILE" ]; then
  if [ ! -s "$ERR_FILE" ]; then
    rm "$ERR_FILE"
  else
    echo "Fileindex: $INDEX" >> "$ERR_FILE"
  fi
fi