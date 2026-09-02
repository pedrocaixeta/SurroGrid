#!/bin/bash

# ==============================================================================
# Slurm Job Submission Script to Generate Thesis Figures on HPC Compute Nodes
# ==============================================================================
#
# This script submits a batch job to run the python plotting script in a safe,
# allocated compute node on the LRZ cluster.
#
# Usage (run this inside GridForecast/0_preprocessing on the HPC cluster):
#   sbatch run_plots.sh 4.2a     <-- Plot only Figure 4.2 a)
#   sbatch run_plots.sh 4.2b     <-- Plot only Figure 4.2 b)
#   sbatch run_plots.sh 4.2b_filtered <-- Plot Figure 4.2 b) filtering out building buses
#   sbatch run_plots.sh 4.2      <-- Plot both Figures 4.2 a) and b)
#   sbatch run_plots.sh 4.1      <-- Plot only Figure 4.1
#   sbatch run_plots.sh all      <-- Plot all figures (default)
#   sbatch run_plots.sh          <-- Plot all figures (default)
# ==============================================================================

# --- Slurm Configuration Headers ---
#SBATCH -J plot_figures                 # Job name shown in squeue
#SBATCH --output=logs/normal/%j_output.log # Log file for normal output (%j inserts job ID)
#SBATCH --error=logs/errors/%j_error.log   # Log file for error prints

# --- LRZ Cluster Specific Resources ---
#SBATCH --clusters=serial               # Use the serial cluster
#SBATCH --partition=serial_std          # Submit to standard serial queue
#SBATCH --ntasks=1                      # Run on a single task/process
#SBATCH --cpus-per-task=4               # Allocate 4 CPUs for calculations
#SBATCH --time=0-00:15:00               # Time limit (15 minutes max)
#SBATCH --mem-per-cpu=4000M             # Request 4GB of RAM per CPU

# Make sure our log directories exist on the HPC file system
mkdir -p logs/normal logs/errors

# --- Load Environment & Packages ---
# Load Miniforge module (gives us access to conda env manager on LRZ)
module load miniforge3

# Initialize shell interface for conda environment activation
eval "$(conda shell.bash hook)"

# Activate the conda environment created for preprocessing/plotting
conda activate preprocessing

# --- Parse Terminal Argument ---
# Read the first argument passed to this script ($1).
# If no argument is provided, default to 'all'.
# Terminal commands:
#   sbatch run_plots.sh 4.2b_filtered  # Plot Figure 4.2 b) filtering out building buses
#   sbatch run_plots.sh 4.2    # Plot only Figure 4.2 b)
#   sbatch run_plots.sh 4.1    # Plot only Figure 4.1
#   sbatch run_plots.sh all    # Plot all figures (default)
FIG_ARG="${1:-all}"
EXTRA_ARGS="${@:2}"

# Print execution settings to log file for verification
echo "========================================="
echo "Starting Slurm Plotting Job"
echo "========================================="
echo "Job ID           : $SLURM_JOB_ID"
echo "Active Cluster   : $SLURM_CLUSTER_NAME"
echo "Allocated CPUs   : $SLURM_CPUS_PER_TASK"
echo "Figure target    : $FIG_ARG"
echo "Extra arguments  : $EXTRA_ARGS"
echo "=========================================\n"

# Run the python script on the allocated compute node using 'srun'
srun python3 plot_thesis_figures.py --fig "$FIG_ARG" $EXTRA_ARGS

echo "\n========================================="
echo "Slurm Job finished successfully!"
echo "========================================="
