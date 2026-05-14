#!/bin/bash

#SBATCH --job-name=nlp_adaptive_rag

#SBATCH --output=/work/qgi899/Final_project/phase2/logs/pipeline_%j.out

#SBATCH --error=/work/qgi899/Final_project/phase2/logs/pipeline_%j.err

#SBATCH --time=24:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16

#SBATCH --partition=gpu2v100

#SBATCH --gres=gpu:1



echo "Job ID: $SLURM_JOB_ID"

echo "Node: $SLURMD_NODENAME"

echo "Start: $(date)"



WORKDIR=/work/qgi899/Final_project/phase2

cd $WORKDIR

mkdir -p logs



module load miniconda/24.4.0



ENV_NAME=nlp_rag

if ! conda env list | grep -q "^$ENV_NAME "; then

    conda create -n $ENV_NAME python=3.11 -y

fi



source activate $ENV_NAME



pip install --quiet openai python-dotenv sentence-transformers numpy



echo "Checking servers..."

curl -s --max-time 10 http://10.246.100.230/v1/models > /dev/null && echo "Llama: OK" || echo "Llama: UNREACHABLE"

curl -s --max-time 10 http://10.100.1.213:8888/v1/models > /dev/null && echo "Qwen: OK" || echo "Qwen: UNREACHABLE"



echo "Starting pipeline at $(date)..."

python adaptive_Rag_CoN.py



echo "Done at $(date)"

ls -lh pipeline_results_v2_prompt2.json pipeline_summary_v2_prompt2.txt 2>/dev/null

