#!/bin/bash
#SBATCH --job-name=sedd_h200_job       # Job name
#SBATCH --output=output.txt            # Output log file
#SBATCH --error=error.txt              # Error log file
#SBATCH --partition=gpuH200x8          # Partition with 8× H200 GPUs
#SBATCH --account=bemc-delta-gpu         # Your valid Slurm account
#SBATCH --gres=gpu:2                   # Request 2 GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # One task (you can adjust for multi-GPU)
#SBATCH --cpus-per-task=16             # 16 cores per GPU is safe
#SBATCH --mem=96G                      # Memory for the job
#SBATCH --time=04:00:00                # Time limit

# Load correct CUDA for H200
# module purge
module load python/3.10.13
module load cuda/12.3.0

# Activate your Python environment
source /work/nvme/bemc/python_envs/sedd_env_3/bin/activate

# Debug info
echo "Running on $HOSTNAME"
echo "GPUs allocated:"
nvidia-smi

# Launch training
python train.py