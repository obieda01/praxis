#!/bin/bash
#SBATCH --job-name=praxis_n100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/orchestrator_n100_%j.log
#SBATCH --error=logs/orchestrator_n100_%j.err

echo "=== Pegasus GPU Unified Orchestrator Job Started (100 Tasks) ==="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Compute Node: $SLURMD_NODENAME"
echo "Date: $(date)"

module load miniconda/23.11.0-2 jdk/1.11.0.7 maven/3.9.6 cuda/12.4
eval "$(/c1/apps/miniconda/23.11.0-2/bin/conda shell.bash hook)"
conda activate praxis-env

nvidia-smi

cd /gpfs/automountdir/gpfs/homes/SEAS/home/g44758203/praxis

MODEL_NAME=${1:-"deepseek-ai/deepseek-coder-6.7b-instruct"}
echo "Evaluating Model: $MODEL_NAME"

python scripts/run_100_tasks_orchestrator.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "data/poc_100_tasks.json" \
    --output_file "results/orchestrator_100_tasks_${SLURM_JOB_ID}.json" \
    --evolution_loops 3

echo "=== Pegasus GPU Unified Orchestrator Job Finished ==="
