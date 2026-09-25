#!/bin/bash
#SBATCH --job-name=swe_orchestrator
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/orchestrator_%j.log
#SBATCH --error=logs/orchestrator_%j.err

echo "=== Pegasus GPU Job Started ==="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Compute Node: $SLURMD_NODENAME"
echo "Date: $(date)"

# Load Cluster Modules
module load miniconda/23.11.0-2 jdk/1.11.0.7 maven/3.9.6 cuda/12.4
eval "$(/c1/apps/miniconda/23.11.0-2/bin/conda shell.bash hook)"
conda activate praxis-env

# Show GPU specifications
nvidia-smi

# Execute the Multi-Agent Reflection Orchestrator
cd /gpfs/automountdir/gpfs/homes/SEAS/home/g44758203/praxis
python scripts/run_reflection_orchestrator.py

echo "=== Pegasus GPU Job Finished ==="
