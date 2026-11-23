#!/bin/bash

#SBATCH -J export_all_buildings
#SBATCH --output=logs/normal/%j_export_all_output.log
#SBATCH --error=logs/errors/%j_export_all_error.log

#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:15:00
#SBATCH --mem-per-cpu=1024M

# Ensure log directories exist
mkdir -p logs/normal logs/errors

# Load Miniforge conda environment
module load miniforge3
module list

# Hook conda into bash and activate the environment
eval "$(conda shell.bash hook)"
conda activate pwrflw-hpc

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Export ALL script started at: $start_time"

# Case 1: ALL_building_Elias_PostUrbs.xlsx
#srun python3 export_all_buildings.py \
#    --input-folder "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/EliasH/PostUrbs/" \
#    --output-file "result/ALL_building_Elias_PostUrbs.xlsx"

# Case 2: ALL_buildings_Pedro_PostUrbs.xlsx
srun python3 export_all_buildings.py \
    --input-folder "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/4th_RUN/4.Power_Flown/" \
    --output-file "output/ALL_buildings_with_installed_HP_Capacity.xlsx"

# Case 3: verification_2dmd_alloc.xlsx
#srun python3 export_all_buildings.py \
#    --input-folder "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/Runs_on_Elias_grids/" \
#    --output-file "result/ALL_buildings_Pedro_PostUrbs.xlsx"
wait

echo "Finished at: $(date +'%Y-%m-%d %H:%M:%S')"
