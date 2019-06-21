#!/bin/bash
#
#SBATCH --job-name=ivae-cpu
#SBATCH --output=slurm_log/ivae-cpu.%A_%a.out
#SBATCH --error=slurm_log/ivae-cpu.%A_%a.err
#
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=8G
#SBATCH --time=0-6:00

source ~/.bashrc
conda activate deep

python main.py $(sed -n ${SLURM_ARRAY_TASK_ID}p args_cpu_seeded.txt)
