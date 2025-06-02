import os
import json
import openai
from agent import AnalysisAgent
from notebook_generator import generate_notebook

# Initialize the agent
agent = AnalysisAgent(
    h5ad_path="example/covid19.h5ad",
    paper_summary_path="example/covid19_summary.txt",
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="o3-mini",
    analysis_name="covid19",
    num_analyses=3
)

# Run the analysis
agent.run()  # This will run all the analyses the agent decides to attempt.

# Generate Jupyter notebook after analysis is complete
generate_notebook(agent.completed_analyses, agent.output_dir)
