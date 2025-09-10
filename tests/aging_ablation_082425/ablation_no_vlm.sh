#!/bin/bash
#SBATCH --job-name=ablation_no_vlm
#SBATCH --output=aging_ablation_082425/ablation_no_vlm_%j.out
#SBATCH --error=aging_ablation_082425/ablation_no_vlm_%j.err
#SBATCH --time=24:00:00
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
    --h5ad-path "/scratch/users/salber/aging_final.h5ad" \
    --manuscript-path "../caseStudies_paperSummaries/aging_brain_eric.txt" \
    --test-name "no_vlm" \
    --num-analyses 15 \
    --max-iterations 8 \
    --output-dir "aging_ablation_082425" \
    --use-self-critique \
    --no-vlm \
    --use-documentation

echo "✅ Ablation test no_vlm completed!"
