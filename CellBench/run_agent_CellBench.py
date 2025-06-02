import os
import pandas as pd
import json
import openai
from dotenv import load_dotenv
import re
import pickle
import numpy as np
import sys
from multiprocessing import Pool, cpu_count

NUM_RUN = 3
import sys
MODEL = sys.argv[1]
print(f'USING {MODEL}')


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

input_dir = "inputs"
output_dir = "outputs_check"
results_dir = "results_check"

agent_output_path = os.path.join(output_dir, f"agent_analyses_unconditioned_{MODEL.replace('-', '')}_{{index}}.json")
data_files = ["gemini_2.5_pro_qa_unconditioned.csv", "gemini_2.5_pro_qa_unconditioned_p2.csv", "gemini_2.5_pro_qa_unconditioned_p3.csv"]

dfs = []
for file_name in data_files:
    path = os.path.join(input_dir, file_name)
    df = pd.read_csv(path)
    df['analyses_full'] = df['analyses_full'].apply(eval)
    dfs.append(df)

# Concatenate and reset index
df = pd.concat(dfs, ignore_index=True)

analyses_full = df['analyses_full'].tolist()
home_dir = "/home/groups/jamesz/salber/scAgent_v2"

import json
import re

def fix_malformed_json(json_str):
    """Fix common JSON formatting issues"""
    # Clean basic formatting
    clean_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
    clean_str = clean_str.replace('```json', '').replace('```python', '').replace('```', '').strip()
    
    # Fix the specific issue: missing comma and field name between strings
    # This pattern looks for: "some text"    "more text" and fixes it
    pattern = r'(")\s+("(?:None of the ground truth|The closest ground truth).*?")'
    replacement = r'\1,\n    "comparison": \2'
    clean_str = re.sub(pattern, replacement, clean_str)
    
    return clean_str

def parse_response(response):
    try:
        if isinstance(response, str):
            clean_response = fix_malformed_json(response)
            try:
                return json.loads(clean_response)
            except json.JSONDecodeError as e:
                print(f"JSON Error: {e}")
                print(f"Error at position: {e.pos}")
                print(f"Fixed string: {clean_response}")
                return {"error": "Invalid JSON", "text": clean_response}
        elif isinstance(response, list):
            parsed = []
            for i, r in enumerate(response):
                if isinstance(r, str):
                    try:
                        clean_r = fix_malformed_json(r)
                        parsed.append(json.loads(clean_r))
                    except json.JSONDecodeError as e:
                        print(f"JSON Error at index {i}: {e}")
                        print(f"Fixed string: {clean_r}")
                        parsed.append({"error": "Invalid JSON", "text": clean_r})
                else:
                    parsed.append(r)
            return parsed
        else:
            return response
    except Exception as e:
        print(f"Exception in parse_response: {str(e)}")
        return {"error": str(e)}

def get_response(prompt, system_prompt, use_json=False):
    if use_json:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        return json.loads(result)
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
        )
        result = response.choices[0].message.content
        return result

analyses_overview = open(os.path.join(home_dir, "prompts", "DeepResearch_Analyses.txt")).read()

first_draft_prompt = f"""
You will be provided the background/introduction from a research paper.
The computational analyses done in the paper are hidden from you.

Your role is to propose a computational analysis that you think was most likely done in the paper.

Ensure that your output is in the specified JSON format.

For the analysis plan, think of the analysis plan as a scientific workflow:
    1. Start with exploratory data analysis that is broad and tests many things
    2. Then, focus on the more promising results from the exploratory phase by creating more focused analyses
    3. Include statistical validation of your results where appropiate
Do not number the analysis plan.
Each step in the analysis plan should be distinct from one another and could involve loading the data, conducting a statistical analysis, printing information about the AnnData object, etc.
Use however many steps is appropiate, but go for at least 5 steps. 


Ensure that your analyses solely use data explicitly mentioned in the paper. For example, if only RNA-seq data is mentioned, do NOT suggest spatial.
Likewise, if no spliced/unspliced RNA counts are mentioned, do NOT suggest RNA velocity.

Here are the previous analyses attempted:
{{past_analyses}}

Here is the background information from the paper:
{{paper_txt}}

For the rest of the prompt, we have examples of potential analyses:
{analyses_overview}
"""



system_prompt = f"""
You are a creative and skilled expert in single-cell transcriptomics computational analysis.

Output your response in the following JSON format (do not number the analysis steps, just list them):
{{
    "hypothesis": "...",
    "analysis_plan": ["First step", "Second step", ...],
    "summary": "A string describing the analysis in a detailed paragraph outlining how you will conduct the analysis"
}}
"""


critic_prompt = f"""
You will be given a hypothesis, analysis plan, and a summary of the analysis plan.
This analysis was generated by being given the background/introduction from a research paper (shown below).
The computational analyses done in the paper are hidden and the goal is to propose a computational analysis that is most likely to be in that hidden set.

Your role is to provide feedback for the analysis based on these goals.

Ensure that the analyses solely uses data explicitly mentioned in the paper. For example, if only RNA-seq data is mentioned, the analysis should NOT involve spatial analyses.
Likewise, if no spliced/unspliced RNA counts are mentioend, the analysis should NOT suggest RNA velocity.

Analysis Summary:
{{summary}}

Analysis Hypothesis:
{{hypothesis}}

Analysis Plan:
{{analysis_plan}}

Here is the background information from the paper:
{{paper_txt}}

Previous Analysis Attempted:
{{past_analyses}}

For the rest of the prompt, we have examples of potential analyses:
{analyses_overview}
"""

incorporate_prompt = f"""
You will be given a hypothesis, analysis plan, and a summary of the analysis plan.
This analysis was generated by being given the background/introduction from a research paper (shown below).
The computational analyses done in the paper are hidden and the goal is to propose a computational analysis that is most likely to be in that hidden set.

You will also be given feedback for the analysis in order so that it achieves these goals. 
Your role is to incorporate that feedback and update the analysis components (summary, hypothesis, analysis plan)

Analysis Summary:
{{summary}}

Analysis Hypothesis:
{{hypothesis}}

Analysis Plan:
{{analysis_plan}}

Feedback:
{{feedback}}

Here are the previous analyses attempted:
{{past_analyses}}

Here is the background information from the paper:
{{paper_txt}}

For the rest of the prompt, we have examples of potential analyses:
{analyses_overview}
"""

def run_single_iteration(run_index):
    """Run a single iteration of agent execution and evaluation"""
    print(f"Starting run {run_index+1}/{NUM_RUN}")
    index = run_index + 1
    agent_output_path_run = agent_output_path.format(index=index)
    
    # Run agent
    all_agent_analyses = []
    for i, row in df.iterrows():
        context = row['context']
        analyses_full_paper = row['analyses_full']

        past_analyses = ""

        agent_analyses = []
        print(f"Running {len(analyses_full_paper)} analyses for paper {i+1}")
        for j in range(len(analyses_full_paper)):
            print(f"Running {j+1}/{len(analyses_full_paper)}")
            first_draft_prompt_filled = first_draft_prompt.format(past_analyses=past_analyses, paper_txt=context)
            first_draft = get_response(first_draft_prompt_filled, system_prompt, use_json=True)

            hypothesis, analysis_plan, summary = first_draft["hypothesis"], first_draft["analysis_plan"], first_draft["summary"]
            critic_prompt_filled = critic_prompt.format(summary=summary, hypothesis=hypothesis, analysis_plan=analysis_plan, 
                                                    paper_txt=context, past_analyses=past_analyses)
            feedback = get_response(critic_prompt_filled, "You are a single-cell bioinformatics expert providing feedback on code and analysis plan.", use_json=False)

            incorporate_prompt_filled = incorporate_prompt.format(summary=summary, hypothesis=hypothesis, analysis_plan=analysis_plan, 
                                                                    feedback=feedback, past_analyses=past_analyses, paper_txt=context)
            final_analysis = get_response(incorporate_prompt_filled, system_prompt, use_json=True)

            analysis = final_analysis['summary']
            agent_analyses.append(analysis)
            past_analyses += f"{analysis}\n\n"

        all_agent_analyses.append(agent_analyses)

        # Save intermediate result after each iteration
        with open(agent_output_path_run, "w") as f:
            json.dump(all_agent_analyses, f, indent=2)

    # Run evaluation
    judge_responses_path = os.path.join(output_dir, f"judge_responses_unconditioned_{MODEL.replace('-', '')}_{index}.json")
    if os.path.exists(judge_responses_path):
        with open(judge_responses_path, "r") as f:
            all_judge_responses = json.load(f)
    else:
        all_judge_responses = []
        for i, agent_analysis_list in enumerate(all_agent_analyses):
            print(f"Processing paper {i+1}/{len(all_agent_analyses)} for run {index}")
            judge_responses = []
            for j, agent_analysis in enumerate(agent_analysis_list):
                prompt = f"""
                You are given a proposed analysis for single-cell transcriptomics and a set of ground truth analyses. Your task is to determine whether the proposed analysis matches at least one analysis from the set of ground truth analyses.

                Proposed:
                {agent_analysis}

                Ground truth:
                {analyses_full[i]}

                Give your answer in the following JSON format:
                {{
                    "match": true/false,
                    "reason": "explanation of the match or mismatch and if match, which analysis it matches"
                }}
                """
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                response = response.choices[0].message.content
                judge_responses.append(response)
            all_judge_responses.append(judge_responses)
        # Save judge responses to file
        with open(judge_responses_path, "w") as f:
            json.dump(all_judge_responses, f, indent=2)

    df_eval = pd.DataFrame({'raw_judge_response': all_judge_responses})
    df_eval['parsed_judge_response'] = df_eval['raw_judge_response'].apply(parse_response)
    
    def validate_judge_response(response):
        if isinstance(response, dict) and 'error' in response:
            return 0
        if not isinstance(response, list):
            return 0
        return sum(1 for r in response if isinstance(r, dict) and r.get('match', False))
    
    df_eval['num_match'] = df_eval['parsed_judge_response'].apply(validate_judge_response)
    num_per_paper = df_eval['parsed_judge_response'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    micro_avg = (df_eval['num_match'] / num_per_paper).mean() if num_per_paper.sum() > 0 else 0
    macro_avg = df_eval['num_match'].sum() / num_per_paper.sum() if num_per_paper.sum() > 0 else 0

    print(f"Run {index} - Micro average: {micro_avg}, Macro average: {macro_avg}")
    return micro_avg, macro_avg

if __name__ == '__main__':
    # Use number of CPU cores minus 1 to leave one core free for system processes
    num_processes = max(1, cpu_count() - 1)
    print(f"Running {NUM_RUN} iterations in parallel using {num_processes} processes")
    
    with Pool(num_processes) as pool:
        results = pool.map(run_single_iteration, range(NUM_RUN))
    
    # Collect results
    micro_averages = [r[0] for r in results]
    macro_averages = [r[1] for r in results]

    #### SAVE RESULTS #####
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"agent_grades_unconditioned_{MODEL}.pkl"), "wb") as f:
        results_dict = {
            "micro_averages": micro_averages,
            "macro_averages": macro_averages,
            "micro_avg": sum(micro_averages) / len(micro_averages) if micro_averages else 0,
            "micro_std": np.std(micro_averages) if micro_averages else 0,
            "macro_avg": sum(macro_averages) / len(macro_averages) if macro_averages else 0,
            "macro_std": np.std(macro_averages) if macro_averages else 0
        }
        print(results_dict)
        pickle.dump(results_dict, f)
