#!/bin/bash
#SBATCH --job-name=train_combined
#SBATCH --output=logs/out/%x-%j.out
#SBATCH --error=logs/error/%x-%j.err
#SBATCH --time=01:10:00
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-v100x2,lrz-dgx-1-v100x8 #These last two partitions don't work with the 25.06 container. If you want to use them (they are usually more disponible, but slower) you have to change point to the 24.12 container below in the code
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

mkdir -p logs/out logs/error models

srun --container-mounts=/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/3_transformer/:/workspace,\
/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/3_transformer/data:/data \
     --container-image='/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/4th_RUN/containers/nvidia+pytorch+24.12-py3.sqsh' \
     #lrz-v100x2,lrz-dgx-1-v100x8 partitions only work with this one (but slows down the processing)

     #--container-image='/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/4th_RUN/containers/nvidia+pytorch+25.06-py3.sqsh' \
     #This container is best suited for the lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8 partitions, but it doesn't work with the lrz-v100x2,lrz-dgx-1-v100x8 partitions.
     python train_combined_data.py
