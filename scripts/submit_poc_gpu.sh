#!/bin/bash
#SBATCH --job-name=swe_poc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/poc_%j.log
#SBATCH --error=logs/poc_%j.err

echo "=== Pegasus GPU Job Started ==="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Compute Node: $SLURMD_NODENAME"
echo "Date: $(date)"

# Load modules and conda environment
module load miniconda/23.11.0-2
eval "$(/c1/apps/miniconda/23.11.0-2/bin/conda shell.bash hook)"
conda activate praxis-env

# Show GPU specifications
nvidia-smi

# Execute the 5-task refactoring inference
cd ~/praxis
python scripts/run_5_tasks_poc.py

echo "=== Pegasus GPU Job Finished ==="