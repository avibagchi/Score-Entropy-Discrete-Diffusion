#!/bin/bash
#SBATCH --job-name=sedd_h100_job       # Job name
#SBATCH --output=output.txt            # Output log file
#SBATCH --error=error.txt              # Error log file
#SBATCH --partition=ghx4               # Match the interactive partition
#SBATCH --account=bemc-dtai-gh         # Your Slurm account
#SBATCH --gres=gpu:h100:2                   # Request 2 GPUs
#SBATCH --cpus-per-gpu=72              # 72 CPUs per GPU (like interactive)
#SBATCH --mem=0                        # Let Slurm auto-assign full memory
#SBATCH --time=48:00:00                # Time limit (48 hours)
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks=1                     # Single task

# Load CUDA for H100
# module purge
module load cuda/12.2.0

# Activate your Python environment
source /work/nvme/bemc/python_envs/sedd_env/bin/activate

# Debug info
echo "Running on $HOSTNAME"
echo "GPUs allocated:"
nvidia-smi

# Launch training
srun python train.py
