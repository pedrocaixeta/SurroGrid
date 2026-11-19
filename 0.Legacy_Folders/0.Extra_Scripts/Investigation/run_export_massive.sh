#!/bin/bash

#SBATCH -J export_massive
#SBATCH --output=logs/normal/%j_export_output.log
#SBATCH --error=logs/errors/%j_export_error.log

#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:35:00
#SBATCH --mem-per-cpu=6200M

# Ensure log directories exist
mkdir -p logs/normal logs/errors

# Load Miniforge conda environment
module load miniforge3
module list

# Hook conda into bash and activate the environment
eval "$(conda shell.bash hook)"
conda activate pwrflw-hpc

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Export script started at: $start_time"

# Case 1: buildings_Elias_PostUrbs.xlsx (in PLZ List)
srun python3 export_massive_buildings.py \
    --input-folder "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/EliasH/PostUrbs/" \
    --output-file "Output/buildings_Elias_PostUrbs.xlsx" \
    --filter-type "plz"

# Case 2: buildings_Pedro_PostUrbs.xlsx (in PLZ List)
srun python3 export_massive_buildings.py \
    --input-folder "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/result3.urbs__input4/" \
    --output-file "Output/buildings_Pedro_PostUrbs.xlsx" \
    --filter-type "plz"

# Case 3: buildings_overloadedindexes_PostUrbs.xlsx (in Index list)
srun python3 export_massive_buildings.py \
    --input-folder "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/result3.urbs__input4/" \
    --output-file "Output/buildings_overloadedindexes_PostUrbs.xlsx" \
    --filter-type "index"

# Case 4: buildings_successfulindexes_PostUrbs.xlsx (NOT in Index list)
srun python3 export_massive_buildings.py \
    --input-folder "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/result3.urbs__input4/" \
    --output-file "Output/buildings_successfulindexes_PostUrbs.xlsx" \
    --filter-type "not_index"

wait

echo "Finished at: $(date +'%Y-%m-%d %H:%M:%S')"
