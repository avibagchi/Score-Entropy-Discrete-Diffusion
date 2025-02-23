#!/bin/bash
#SBATCH --job-name=sedd_gpu_job       # Job name
#SBATCH --output=output.txt           # Output log file
#SBATCH --error=error.txt             # Error log file
#SBATCH --partition=gpu               # Request a GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1                    # Number of tasks (1 per GPU job)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=48G                     # Memory per node
#SBATCH --time=02:00:00               # Time limit
#SBATCH --constraint="a100|a40|l40s" # Request GPUs with compute capability 8.0 or higher

module load cuda/11.8.0

module load miniconda/22.11.1-1


echo "Conda environment: $(which python)"

python run_train.py # changed this 