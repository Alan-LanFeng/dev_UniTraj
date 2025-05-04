#!/bin/bash
#SBATCH --nodes=1
#SBATCH --chdir=/work/vita/lanfeng/dev_UniTraj/unitraj
#SBATCH --account=vita
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --partition=h100
#SBATCH --mem=90GB
#SBATCH --output=/home/lfeng/task_logs/%j.log

export HYDRA_FULL_ERROR=1
module load gcc cuda
srun python evaluation.py method=transfuser