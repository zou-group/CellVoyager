#!/bin/bash
#SBATCH --job-name=ablation_baseline
#SBATCH --output=covid_ablation_080125/ablation_baseline_%j.out
#SBATCH --error=covid_ablation_080125/ablation_baseline_%j.err
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
    --h5ad-path "../example/covid19.h5ad" \
    --manuscript-path "../example/covid19_summary.txt" \
    --test-name "baseline" \
    --num-analyses 8 \
    --max-iterations 6 \
    --output-dir "covid_ablation_080125" \
    --use-self-critique \
    --use-vlm

echo "✅ Ablation test baseline completed!"
