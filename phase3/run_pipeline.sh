#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# run_pipeline.sh — Adaptive RAG Phase 3, batch-mode SLURM submission
#
# USAGE — submit 2-3 jobs covering different index ranges in parallel:
#
#   sbatch --export=START_IDX=0,END_IDX=2000    run_pipeline.sh
#   sbatch --export=START_IDX=2000,END_IDX=4000  run_pipeline.sh
#   sbatch --export=START_IDX=4000,END_IDX=6000  run_pipeline.sh
#
# Or set a default range here and submit once:
#   sbatch run_pipeline.sh
#
# Output files are named with the index range, e.g.:
#   phase3_results_0-2000.json
#   phase3_summary_0-2000.txt
# ════════════════════════════════════════════════════════════════════

#SBATCH --job-name=nlp_p3_%j
#SBATCH --output=/work/qgi899/Final_project/phase3/logs/pipeline_%j.out
#SBATCH --error=/work/qgi899/Final_project/phase3/logs/pipeline_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu2v100
#SBATCH --gres=gpu:1

# ── Default index range (overridden by --export on sbatch call) ────
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-2000}

echo "════════════════════════════════════════════"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Start    : $(date)"
echo "Batch    : [$START_IDX - $END_IDX]"
echo "════════════════════════════════════════════"

WORKDIR=/work/qgi899/Final_project/phase3
cd $WORKDIR
mkdir -p logs

module load miniconda/24.4.0

ENV_NAME=nlp_rag
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda env..."
    conda create -n $ENV_NAME python=3.11 -y
fi

source activate $ENV_NAME

echo "Installing dependencies..."
pip install --quiet openai python-dotenv sentence-transformers numpy

echo ""
echo "Checking LLM servers..."
curl -s --max-time 10 http://10.246.100.230/v1/models > /dev/null \
    && echo "  Llama  : OK" \
    || echo "  Llama  : UNREACHABLE — job may fail"
curl -s --max-time 10 http://10.100.1.213:8888/v1/models > /dev/null \
    && echo "  Qwen   : OK" \
    || echo "  Qwen   : UNREACHABLE — job may fail"

echo ""
echo "Starting pipeline at $(date) ..."
python adaptive_rag_pipeline.py --start $START_IDX --end $END_IDX

echo ""
echo "Done at $(date)"
echo ""

# Show output files for this batch
BATCH_LABEL="${START_IDX}-${END_IDX}"
echo "── Output files ──"
ls -lh \
    "phase3_results_${BATCH_LABEL}.json" \
    "phase3_summary_${BATCH_LABEL}.txt" \
    2>/dev/null || echo "(no output files found — check logs above)"