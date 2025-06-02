import pandas as pd
import os
import json
from tqdm import tqdm
import openai

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def parse_response(response):
    try:
        if isinstance(response, str):
            response = response.replace('```json', '').replace('```', '').strip()
            parsed = json.loads(response)
        elif isinstance(response, list):
            parsed = []
            for r in response:
                r = r.replace('```json', '').replace('```', '').strip()
                r = json.loads(r)
                parsed.append(r)
    except json.JSONDecodeError:
        print(f"Failed to parse response: {response}")
        return response
    return parsed

def parse_matches(response):
    try:
        return sum(map(lambda x: int(x['match']), response))
    except Exception as e:
        print(f"{e}: Failed to parse matches: {response}")
        return 0

def run_judge(run_name, gt_col):
    df = pd.read_csv(f'responses/{run_name}.csv')
    df['parsed_response'] = df['parsed_response'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    df[gt_col] = df[gt_col].apply(eval)

    if os.path.isfile(f'judged/{run_name}_judged.csv'):
        print(f'Judged file already exists for {run_name}.')
        df = pd.read_csv(f'judged/{run_name}_judged.csv')
    else:
        df['raw_judge_response'] = None
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # labels = row['analyses_full']
            labels = row[gt_col]
            responses = row['parsed_response']
            judge_responses = []
            for response in responses:
                prompt = f"""
    You are given a proposed analysis for single-cell transcriptomics and a set of ground truth analyses. Your task is to determine whether the proposed analysis matches at least one analysis from the set of ground truth analyses.

    Proposed:
    {response}

    Ground truth:
    {labels}

    Give your answer in the following format:
    {{
        "match": true/false,
        "reason": "explanation of the match or mismatch and if match, which analysis it matches"
    }}
    """

                response = client.responses.create(
                    model="gpt-4o",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                },
                            ]
                        }
                    ]
                )

                response = response.output_text
                judge_responses.append(response)
            df.at[idx, 'raw_judge_response'] = judge_responses

    df['parsed_judge_response'] = df['raw_judge_response'].apply(parse_response)
    df['num_match'] = df['parsed_judge_response'].apply(parse_matches)
    num_per_paper = df['parsed_judge_response'].apply(len)
    micro_avg = (df['num_match'] / num_per_paper).mean()
    macro_avg = df['num_match'].sum() / num_per_paper.sum()
    df.to_csv(f'judged/{run_name}_judged.csv', index=False)
    return micro_avg, macro_avg

def main():
    gt_col = 'analyses_full'
    os.makedirs('judged', exist_ok=True)
    for model_name in ['gpt-4o', 'o3-mini']:
        for run_idx in range(0, 3):
            # Run judge for each model and run replicate
            run_name = f'cellbench_50_{model_name}_responses_{run_idx}'
            micro_avg, macro_avg = run_judge(run_name, gt_col)
            # Print the results
            print(f'Run {run_idx} - Model {model_name}: Micro Avg: {micro_avg}, Macro Avg: {macro_avg}')

if __name__ == "__main__":
    main()
