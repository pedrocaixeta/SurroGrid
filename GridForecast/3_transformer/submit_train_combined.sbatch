#!/bin/bash
#SBATCH --job-name=train_combined
#SBATCH --output=logs/out/%x-%j.out
#SBATCH --error=logs/error/%x-%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

mkdir -p logs/out logs/error models

srun --container-mounts=/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/3_transformer/:/workspace,\
/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/3_transformer/data:/data \
     --container-image='/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/4th_RUN/containers/nvidia+pytorch+25.06-py3.sqsh' \
     python train_combined_data.py
