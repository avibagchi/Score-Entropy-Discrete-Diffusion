#!/bin/bash
#SBATCH --job-name=sedd_gpu_job       # Job name
#SBATCH --output=output.txt           # Output log file
#SBATCH --error=error.txt             # Error log file
#SBATCH --partition=gpu               # Request a GPU partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1                    # Number of tasks (1 per GPU job)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=16G                     # Memory per node
#SBATCH --time=02:00:00               # Time limit
#SBATCH --constraint="a100|a40|l40s" # Request GPUs with compute capability 8.0 or higher

# Load CUDA module
module load cuda/11.8.0

# Load Miniconda module
module load miniconda/22.11.1-1

# Activate your Conda environment
source activate sedd_env

# Optionally, check if Conda environment is active
echo "Conda environment: $(which python)"

# Run your Python script
python run_sample.py --model_path louaaron/sedd-small --steps 1000
