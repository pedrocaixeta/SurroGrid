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
#   sbatch run_plots.sh all      <-- Plot all thesis figures (default)
#   sbatch run_plots.sh vv       <-- Run plot_vv_incidence.py
#   sbatch run_plots.sh vm_multi <-- Run plot_vm_timeseries_multipple_grids.py
#   sbatch run_plots.sh vm_regio <-- Run plot_vm_timeseries_by_regiostar.py
#   sbatch run_plots.sh submit_all <-- Submits 4 separate jobs for thesis, vv, vm_multi, and vm_regio
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
#SBATCH --time=0-00:35:00               # Time limit
#SBATCH --mem-per-cpu=4000M             # Request 4GB of RAM per CPU

# Make sure our log directories exist on the HPC file system
mkdir -p logs/normal logs/errors

# --- Load Environment & Packages ---
# Load Miniforge module (gives us access to conda env manager on LRZ)
module load miniforge3

# Initialize shell interface for conda environment activation
eval "$(conda shell.bash hook)"

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

# If user wants to submit all jobs concurrently
if [ "$FIG_ARG" = "submit_all" ]; then
    echo "Submitting a separate job for each plotting script..."
    sbatch -J plot_thesis "$0" all
    sbatch -J plot_vv "$0" vv
    sbatch -J plot_vm_multi "$0" vm_multi
    sbatch -J plot_vm_regio "$0" vm_regio
    echo "All 4 jobs have been submitted to Slurm!"
    exit 0
fi

# Activate the conda environment created for preprocessing/plotting
if [ "$FIG_ARG" = "vv" ] || [ "$FIG_ARG" = "vm_multi" ] || [ "$FIG_ARG" = "vm_regio" ]; then
    conda activate pwrflw-hpc
else
    conda activate preprocessing
fi

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

# Run the appropriate python script on the allocated compute node using 'srun'
if [ "$FIG_ARG" = "vv" ]; then
    srun python3 plot_vv_incidence.py $EXTRA_ARGS
elif [ "$FIG_ARG" = "vm_multi" ]; then
    srun python3 plot_vm_timeseries_multipple_grids.py $EXTRA_ARGS
elif [ "$FIG_ARG" = "vm_regio" ]; then
    srun python3 plot_vm_timeseries_by_regiostar.py $EXTRA_ARGS
else
    srun python3 plot_thesis_figures.py --fig "$FIG_ARG" $EXTRA_ARGS
fi

echo -e "\n========================================="
echo "Slurm Job finished successfully!"
echo "========================================="
