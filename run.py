import os
import json
import openai
from agent import AnalysisAgent
from notebook_generator import generate_notebook

# Initialize the agent
agent = AnalysisAgent(
    h5ad_path="/scratch/users/salber/endo_data.h5ad",
    paper_summary_path="/home/groups/jamesz/salber/scAgent/paper_summaries/endo.txt",
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="o3-mini",
    analysis_name="endo",
    num_analyses=8
)

# Run the analysis
agent.run()  # This will run all the analyses the agent decides to attempt.

# Generate Jupyter notebook after analysis is complete
generate_notebook(agent.completed_analyses, agent.output_dir)

def collect_feedback():
    # Ask for feedback from the user
    print("Please provide feedback on the analysis (e.g., improvements or ideas for new analysis).")
    feedback = input("Your feedback: ")
    return feedback

# Only after all analyses are completed, ask for feedback
#feedback = collect_feedback()

# Example feedback handling:
#print("\nUser Feedback: ", feedback)
# Here, you can send the feedback to the agent or use it to tweak future analyses.
