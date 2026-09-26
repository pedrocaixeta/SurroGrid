#!/bin/bash
# ==============================================================================
# Slurm Job Submission Script to Generate Voltage Violation Incidence Plot
# ==============================================================================
#
# Usage (run this inside Extra_Scripts/Plotting on the HPC cluster):
#   sbatch run_plot_vv_incidence.sh
# ==============================================================================

# --- Slurm Configuration Headers ---
#SBATCH -J plot_vv_incidence            # Job name shown in squeue
#SBATCH --output=logs/normal/%j_output.log
#SBATCH --error=logs/errors/%j_error.log

# --- LRZ Cluster Specific Resources ---
# Using cm2 cluster / cm2_tiny partition to avoid the serial partition queue
#SBATCH --clusters=cm2
#SBATCH --partition=cm2_tiny
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:30:00
#SBATCH --mem-per-cpu=4000M

mkdir -p logs/normal logs/errors

# Load Miniforge module (gives us access to conda env manager on LRZ)
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate grid_alloc

echo "========================================="
echo "Starting Slurm Plotting Job for Voltage Violations"
echo "========================================="
echo "Job ID           : $SLURM_JOB_ID"
echo "Active Cluster   : $SLURM_CLUSTER_NAME"
echo "Active Partition : $SLURM_JOB_PARTITION"
echo "Allocated CPUs   : $SLURM_CPUS_PER_TASK"
echo "========================================="

srun python plot_vv_incidence.py

echo "========================================="
echo "Slurm Job finished successfully!"
echo "========================================="
