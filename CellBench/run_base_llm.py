import pandas as pd
import json
import os
from tqdm import tqdm
import openai

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def parse_response(response):
    if isinstance(response, str):
        response = response.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(response)
    elif isinstance(response, list):
        parsed = []
        for r in response:
            r = r.replace('```json', '').replace('```', '').strip()
            r = json.loads(r)
            parsed.append(r)
    return parsed

def run_cellbench(csv_path, model_name, run_idx):
    save_path = f'responses/cellbench_50_{model_name}_responses_{run_idx}.csv'
    if os.path.isfile(save_path):
        print(f"File {save_path} already exists. Skipping...")
        return
    df = pd.read_csv(csv_path)
    df['analyses_full'] = df['analyses_full'].apply(eval)

    df['raw_response'] = None
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        context = row['context']
        prompt = f"""
Given the following scientific background and research questions, propose {len(row['analyses_full'])} single-cell RNA-seq analyses that are consistent with the goals. Make sure your proposed analyses are specific and ONLY pertain to single-cell RNA-seq.

{context}

Give your answer in the form of a list of exactly {len(row['analyses_full'])} JSON objects, where each analysis is a JSON object with the following keys:
- title: a string describing the analysis
- description: a string describing the analysis in a detailed paragraph outlining how you will conduct the analysis.
"""

        response = client.responses.create(
            model=model_name,
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
        df.loc[idx, 'raw_response'] = response

    df['parsed_response'] = df['raw_response'].apply(parse_response)
    df.to_csv(save_path, index=False)

def main():
    csv_path = 'data/cellbench_50.csv'

    os.makedirs('responses', exist_ok=True)

    # Run for each model and run replicate
    for model_name in ['gpt-4o', 'o3-mini']:
        for run_idx in range(0, 3):
            run_cellbench(csv_path, model_name, run_idx)

if __name__ == "__main__":
    main()
