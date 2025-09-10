#!/bin/bash
#SBATCH --job-name=ablation_baseline
#SBATCH --output=endo_ablation_080825/ablation_baseline_%j.out
#SBATCH --error=endo_ablation_080825/ablation_baseline_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --partition=jamesz

# Load any required modules
module load python/3.9
module load cuda/11.7

# Activate your conda environment if needed
source /scratch/users/salber/packages/miniconda3/etc/profile.d/conda.sh
conda activate CellVoyager

# Change to the tests directory
cd /home/groups/jamesz/salber/CellVoyager/tests

# Run the single ablation test
python single_ablation_test.py \
    --h5ad-path "/scratch/groups/jamesz/scAgent/endo_data.h5ad" \
    --manuscript-path "/home/groups/jamesz/salber/scAgent/paper_summaries/endo.txt" \
    --test-name "baseline" \
    --num-analyses 25 \
    --max-iterations 8 \
    --output-dir "endo_ablation_080825" \
    --use-self-critique \
    --use-vlm

echo "✅ Ablation test baseline completed!"
