#!/bin/bash
#SBATCH --job-name=ablation_tests
#SBATCH --output=jobs_files/ablation_%j.out
#SBATCH --error=jobs_files/ablation_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=1
#SBATCH --partition=jamesz

# Load any required modules
module load python/3.9
module load cuda/11.7

# Activate your conda environment if needed
source /scratch/users/salber/packages/miniconda3/etc/profile.d/conda.sh
conda activate CellVoyager

#python run_ablation_tests.py ../example/covid19.h5ad ../example/covid19_summary.txt \
#        --num-analyses 25 --max-iterations 8 --output-dir covid_ablation_080725

python run_ablation_tests.py /scratch/groups/jamesz/scAgent/endo_data.h5ad \
        /home/groups/jamesz/salber/scAgent/paper_summaries/endo.txt \
        --num-analyses 25 --max-iterations 8 --output-dir endo_ablation_080725
