#!/bin/bash
#
#SBATCH --job-name=ivae-gpu
#SBATCH --output=slurm_log/ivae-gpu.%A_%a.out
#SBATCH --error=slurm_log/ivae-gpu.%A_%a.err
#
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH --time=0-6:00
#
#SBATCH --gres=gpu:1

module add nvidia/9.0

source ~/.bashrc
conda activate deep

python main.py $(sed -n ${SLURM_ARRAY_TASK_ID}p args_gpu_seeded.txt)
