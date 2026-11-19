#!/bin/bash
#SBATCH --job-name=mlp_hpo
#SBATCH --output=logs/out/%x-%j.out
#SBATCH --error=logs/error/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=lrz-dgx-a100-80x8,lrz-hgx-a100-80x4,lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

##SBATCH --job-name=mlp_hpo
##SBATCH --output=logs/out/%x-%j.out
##SBATCH --error=logs/error/%x-%j.err
##SBATCH --time=00:30:00
##SBATCH --partition=lrz-dgx-a100-80x8,lrz-hgx-a100-80x4
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=18
##SBATCH --gres=gpu:1
##SBATCH --mem=128G


srun	--container-mounts=/dss/dsshome1/05/go49cer2/SurroGrid_4thRUN/GridForecast/2_mlp/:/workspace \
	    --container-image='/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/3rd_batch/4th_RUN/containers/nvidia+pytorch+25.06-py3.sqsh' \
	    python run_mlp_models_HPO.py