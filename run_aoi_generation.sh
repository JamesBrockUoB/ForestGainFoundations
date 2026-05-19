#!/bin/bash
#SBATCH --job-name=aoi-gen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=72:00:00
#SBATCH --output=logs/aoi_%j.log
#SBATCH --error=logs/aoi_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com

cd ./DataCollection/

source ~/.bashrc
conda activate forest-growth-venv

export NUM_WORKERS=7  # Leave 1 CPU for main thread + writer
export BATCH_SIZE=500
export OUTPUT_DIR="data/"

python generate_aois.py

# Email results summary
tail -20 logs/aoi_$SLURM_JOB_ID.log | mail -s "AOI generation complete (Job $SLURM_JOB_ID)" james.brock@bristol.ac.uk