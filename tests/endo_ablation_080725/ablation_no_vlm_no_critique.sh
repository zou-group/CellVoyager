#!/bin/bash
#SBATCH --job-name=ablation_no_vlm_no_critique
#SBATCH --output=endo_ablation_080725/ablation_no_vlm_no_critique_%j.out
#SBATCH --error=endo_ablation_080725/ablation_no_vlm_no_critique_%j.err
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
    --test-name "no_vlm_no_critique" \
    --num-analyses 25 \
    --max-iterations 8 \
    --output-dir "endo_ablation_080725" \
    --no-self-critique \
    --no-vlm

echo "✅ Ablation test no_vlm_no_critique completed!"
