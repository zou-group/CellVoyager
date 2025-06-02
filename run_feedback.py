import os
import json
import openai
from agent import AnalysisAgent
from notebook_generator import generate_notebook

# Initialize the agent
agent = AnalysisAgent(
    h5ad_path="/scratch/groups/jamesz/scAgent/final_aging_brain_eric.h5ad",
    paper_summary_path="/home/groups/jamesz/salber/scAgent/paper_summaries/aging_brain_eric.txt",
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="o3-mini",
    analysis_name="aging",
    num_analyses=1
)
#feedback = """

######### ENDO #########
#feedback = "From the set of 40 ligand-receptor correlations you found, plot the three high absolute pearson correlation coefficients (all on the same row). Make sure you use pearson and not spearman"
#agent.improve_notebook("endo_feedback_2.ipynb", feedback, output_path="endo_feedback_3.ipynb")



######## COVID #####
#feedback = """
#For the following analysis conducted in the notebook, redo it but using pseudobulk by donor:
#This code subsets the pre-loaded anndata object by key immune cell types, generates violin plots for apoptosis and pyroptosis scores, and uses helper functions to plot ligand or receptor gene expression with reduced redundancy and enhanced readability. It also performs Mann–Whitney U tests using the 'two-sided' alternative after confirming sufficient cell numbers in both COVID-19 and healthy groups.
#"""
#agent.improve_notebook("outputs/covid_20250517_152612/covid_analysis_7.ipynb", feedback, output_path="covid_feedback.ipynb")


####### AGING #######
#feedback_1 = "Plot a UMAP of the Leiden clusters, colored by transcriptional noise. Use data_astro as the anndata object (contains just astrocytes)"
#agent.improve_notebook("outputs/aging_analysis_1.ipynb", feedback_1, output_path="aging_feedback_1.ipynb")

#feedback_2 = """
#Although there is minimal change in Astrocyte_qNSC 
#transcriptional noise with age, are there more substantial 
#changes in transcriptional noise with age in other cell types
#(e.g. aNSC_NPC, Neuroblast)? Look at 2-4 celltypes apart from Astrocyte_qNSC. 
#"""
#agent.improve_notebook("outputs/aging_analysis_1.ipynb", feedback_2, output_path="aging_feedback_2.ipynb")

#feedback_3 = """
#For the top plot showing transcriptional noise with age in G2M cells, there might need to be some more analysis to interpret the 
#negative correlation (typically trans. noise increases with age) -- it could be that old cells are less "G2M" than young cells which 
#could be interesting or might be an artifact from the oldest animal which seems to drive most of the negative correlation. 
#"""

#feedback_4  = """
#Use cells that are in the set ['Astrocyte_qNSC', 'Neuroblast_1', 'Neuroblast_2'].
#Plot every pairing of the following: noise_metric, age, and G2M.Score, split by their cell cycle phase (in .obs['Phase']).
#In other words, plot a 3x3 grid where the row is the cell cycle phase and the column is on the above pairings.
#For all pairings involving age, age must be on the x-axis.
#For each plot the line of best fit, including the r and p values.
#Calculate noise_metric as done previously in the code
#For noise_metric, remove genes used for calculating G2M.Score: these are in /home/groups/jamesz/salber/scAgent_v2/regev_lab_cell_cycle_genes.txt
#When checking if genes are in the adata, ensure all the gene names are capitalized in the adata
#"""

feedback_5 = """
For every celltype, calcualte the transcriptional noise among young cells and among old cells (splitting on the median).
Plot a boxplot showing the difference in distribution for each celltype and calculate a p-value for each celltype.
Color the young boxplots as blue and the old boxplots as red.
Calculate noise_metric as done previously in the code.
Put everything on the same plot. Put stars for significant boxplots.

Make sure to put a legend on the plot and label each pair of boxplots by just their celltype.
"""
#Use cells that are in the set ['Astrocyte_qNSC', 'Neuroblast_1', 'Neuroblast_2'].
#Calculate noise_metric as done previously in the code
agent.improve_notebook("outputs/aging_analysis_1.ipynb", feedback_5, output_path="aging_feedback_6.ipynb")

