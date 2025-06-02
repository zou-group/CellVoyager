import os
import json
import openai
from agent import AnalysisAgent
from notebook_generator import generate_notebook

# Initialize the agent
agent = AnalysisAgent(
    h5ad_path = os.path.join(os.getcwd(), "example/covid19.h5ad"),
    paper_summary_path = os.path.join(os.getcwd(), "example/covid19_summary.txt"),
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="o3-mini",
    analysis_name="covid19",
    num_analyses=1
)
feedback = "Extend the analysis to more celltypes"
agent.improve_notebook("outputs/covid19_analysis_1.ipynb", feedback, output_path="outputs/covid19_analysis_1_improved.ipynb")

