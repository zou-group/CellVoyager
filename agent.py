import openai
import os
import json
import scanpy as sc
import nbformat as nbf
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor
import tempfile
import numpy as np
import gc
import datetime
from logger import Logger
import base64
import h5py
from h5py import Dataset, Group
import re
import shutil
from PIL import Image
import io
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbformat.v4 import new_notebook, new_code_cell, new_output

AVAILABLE_PACKAGES = "scanpy, scvi, CellTypist, anndata, matplotlib, numpy, seaborn, pandas, scipy"
class AnalysisAgent:
    def __init__(self, h5ad_path, paper_summary_path, openai_api_key, model_name, analysis_name, 
                num_analyses=5, max_iterations=6, prompt_dir="prompts", output_home=".", log_home=".",
                use_self_critique=True, use_VLM=True, use_documentation=True):
        self.h5ad_path = h5ad_path
        self.paper_summary = open(paper_summary_path).read()
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.analysis_name = analysis_name
        self.max_iterations = max_iterations
        self.num_analyses = num_analyses
        self.prompt_dir = prompt_dir
        
        self.completed_analyses = []
        self.failed_analyses = []
        # Create unique output directory based on analysis name and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_home, "outputs", f"{analysis_name}_{timestamp}")
        
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize code memory to track the last few cells of code
        self.code_memory = []
        self.code_memory_size = 5  # Number of code cells to remember

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Coding guidelines: guide agent on how to write code and conduct analyses
        analyses_overview = open(os.path.join(self.prompt_dir, "DeepResearch_Analyses.txt")).read()
        if self.use_VLM:
            self.coding_guidelines = open(os.path.join(self.prompt_dir, "coding_guidelines.txt")).read()
        else:
            self.coding_guidelines = open(os.path.join(self.prompt_dir, "coding_guidelines_NO_VLM_ABLATION.txt")).read()
        self.coding_guidelines = self.coding_guidelines.format(name=self.analysis_name, adata_path=self.h5ad_path, available_packages=AVAILABLE_PACKAGES,
                                                               deepresearch_summary=analyses_overview)

        # System prompt for coding agents
        self.coding_system_prompt = open(os.path.join(self.prompt_dir, "coding_system_prompt.txt")).read()

        # Initialize logger: keeps track of all actions, prompts, responses, errors, etc.
        self.logger = Logger(self.analysis_name, log_dir=os.path.join(log_home, "logs"))
        self.logger.log_action(
            "Agent initialized", 
            f"h5ad_path: {h5ad_path}\n" +
            f"model: {model_name}\n" +
            f"max_iterations: {max_iterations}"
        )
        # Initialize notebook executor
        self.executor = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        # Initialize persistent kernel for efficient cell execution
        self.kernel_manager = None
        self.kernel_client = None

        # Primarily for ablation studies
        self.use_self_critique = use_self_critique
        self.use_VLM = use_VLM
        self.use_documentation = use_documentation

        # Load the .obs data from the anndata file
        if self.h5ad_path == "": # JUST FOR BENCHMARKING
            self.adata_summary = ""
        else:
            print("Loading anndata .obs for summarization...")
            self.adata_obs = self.load_h5ad_obs(self.h5ad_path)
            self.adata_summary = self.summarize_adata_metadata()
            print("ADATA SUMMARY: ", self.adata_summary)
            self.logger.log_action("Data loaded and summarized", self.adata_summary)
            print(f"✅ Loaded {self.h5ad_path}")
        


    def summarize_adata_metadata(self, length_cutoff=25):
        """
        Summarize the agent's anndata metadata

        Args:
            length_cutoff (int): How many max unique values to include for each metadata column
        """
        summarization_str = f"Below is a description of the columns in adata.obs: \n"
        columns = self.adata_obs.columns
        for col in columns:
            unique_vals = self.adata_obs[col].unique()
            if len(unique_vals) > length_cutoff:
                vals_str = str(unique_vals[:length_cutoff]) + f"and {len(unique_vals) - length_cutoff} other unique values..."
            else:
                vals_str = str(unique_vals)
            summarization_str += f"Column {col} contains the unique values {vals_str} \n"
        return summarization_str

    def load_h5ad_obs(self, h5ad_path):
        """Load just the .obs data from an h5ad file while preserving data types"""
        with h5py.File(h5ad_path, 'r') as f:
            obs_dict = {}
            
            # Process each column in obs
            for raw_k in [k for k in f['obs'].keys() if not k.startswith('_')]:
                # Decode the column name if it's bytes
                k = raw_k.decode('utf-8') if isinstance(raw_k, bytes) else raw_k
                
                item = f['obs'][raw_k]
                if isinstance(item, Dataset):
                    data = item[:]
                elif isinstance(item, Group) and \
                    'codes' in item.keys() and 'categories' in item.keys():
                    data = item['codes'][:]
                    categories = item['categories'][:]
                    categories = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in categories]
                    data = pd.Categorical.from_codes(
                        data.astype(int) if not np.issubdtype(data.dtype, np.integer) else data,
                        categories=categories
                    )
                else:
                    raise ValueError(f'uwu didnt account for this datatype in h5ad: {type(item)}')
                
                # Handle categorical data
                if 'categories' in item.attrs:
                    try:
                        # Get category values (handling references if needed)
                        cat_ref = item.attrs['categories']
                        if isinstance(cat_ref, h5py.h5r.Reference):
                            # Dereference to get categories
                            cat_vals = f[cat_ref][:]
                            categories = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in cat_vals]
                        else:
                            # Normal categories
                            cat_vals = cat_ref[:]
                            categories = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in cat_vals]
                        
                        # Create categorical data
                        data = pd.Categorical.from_codes(
                            data.astype(int) if not np.issubdtype(data.dtype, np.integer) else data,
                            categories=categories
                        )
                    except Exception as e:
                        print(f"Warning: Error with categorical {k}: {str(e)}")
                        data = np.array([x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in data])
                
                # Handle string data
                elif data.dtype.kind in ['S', 'O'] or h5py.check_string_dtype(f['obs'][raw_k].dtype) is not None:
                    try:
                        # Make sure all byte strings are decoded
                        data = np.array([x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in data])
                    except Exception as e:
                        print(f"Warning: Error decoding strings in {k}: {str(e)}")
                
                obs_dict[k] = data
            
            # Get index
            try:
                if '_index' in f['obs']:
                    idx = f['obs']['_index'][:]
                    # Decode index values if they're bytes
                    index = np.array([x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in idx])
                else:
                    index = None
            except Exception as e:
                print(f"Warning: Error processing index: {str(e)}")
                index = None
        
        # Create dataframe
        df = pd.DataFrame(obs_dict, index=index)
        print(f"Loaded obs data: {len(df)} rows × {len(df.columns)} columns")
        return df

    def generate_initial_analysis(self, attempted_analyses):
        prompt = open(os.path.join(self.prompt_dir, "first_draft.txt")).read()
        prompt = prompt.format(CODING_GUIDELINES=self.coding_guidelines, adata_summary=self.adata_summary, 
                               past_analyses=attempted_analyses, paper_txt=self.paper_summary)

        
        self.logger.log_prompt("user", prompt, "Initial Analysis")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        
        analysis = json.loads(result)
        return analysis
    
    def update_code_memory(self, notebook_cells):
        """Update the code memory with the latest code cells from the notebook"""
        # Extract code cells from the notebook
        code_cells = []
        for cell in notebook_cells:
            if cell.get('cell_type') == 'code':
                code_cells.append(cell['source'])
                
        # Keep only the most recent cells up to code_memory_size
        self.code_memory = code_cells[-self.code_memory_size:] if len(code_cells) > 0 else []
        
    def generate_next_step_analysis(self, analysis, attempted_analyses, notebook_cells, results_interpretation, num_steps_left):
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]
        
        # Update code memory with latest notebook cells
        self.update_code_memory(notebook_cells)
        
        # Use the code memory for generating the next step
        recent_code = "\n\n# Next Cell\n".join(reversed(self.code_memory))

        prompt = open(os.path.join(self.prompt_dir, "next_step.txt")).read()
        prompt = prompt.format(hypothesis=hypothesis, analysis_plan=analysis_plan, first_step_code=first_step_code,
                               CODING_GUIDELINES=self.coding_guidelines, results_interpretation=results_interpretation,
                               previous_code=recent_code, adata_summary=self.adata_summary, past_analyses=attempted_analyses,
                               paper_txt=self.paper_summary, num_steps_left=num_steps_left)
        
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content

        analysis = json.loads(result)
        return analysis

    def critique_step(self, analysis, past_analyses, notebook_cells):
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]

        if notebook_cells is None:
            recent_code = ""
        else:
            # Update code memory with latest notebook cells
            self.update_code_memory(notebook_cells)
            
            # Use the code memory for generating the next step
            recent_code = "\n\n# Next Cell\n".join(reversed(self.code_memory))

        prompt = open(os.path.join(self.prompt_dir, "critic.txt")).read()
        prompt = prompt.format(hypothesis=hypothesis, analysis_plan=analysis_plan, first_step_code=first_step_code,
                               CODING_GUIDELINES=self.coding_guidelines, adata_summary=self.adata_summary, past_analyses=past_analyses,
                               paper_txt=self.paper_summary, previous_code=recent_code)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a single-cell bioinformatics expert providing feedback on code and analysis plan."},
                {"role": "user", "content": prompt}
            ]
        )
        feedback = response.choices[0].message.content
        return feedback

    def incorporate_critique(self, analysis, feedback, notebook_cells):
        ## Return analysis object
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]

        if notebook_cells is None:
            recent_code = ""
        else:
            # Update code memory with latest notebook cells
            self.update_code_memory(notebook_cells)

        recent_code = "\n\n# Next Cell\n".join(reversed(self.code_memory))

        prompt = open(os.path.join(self.prompt_dir, "incorporate_critque.txt")).read()
        prompt = prompt.format(hypothesis=hypothesis, analysis_plan=analysis_plan, first_step_code=first_step_code,
                               CODING_GUIDELINES=self.coding_guidelines, adata_summary=self.adata_summary,
                               feedback=feedback, previous_code=recent_code)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        modified_analysis = json.loads(response.choices[0].message.content)

        self.logger.log_prompt("user", prompt, "Incorporate Critiques")

        return modified_analysis
    
    def fix_code(self, code, error, other_code=""):
        """Attempts to fix code that produced an error"""
        prompt = f"""Fix this code that produced an error:
        
        Code:
        ```python
        {code}
        ```
        
        Error:
        {error}

        Here is previous code/context (if any):
        {other_code}
        
        Provide only the fixed code with no explanation.
        You can only use the following packages: {AVAILABLE_PACKAGES}
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a coding assistant helping to fix code."},
                {"role": "user", "content": prompt}
            ]
        )
        fixed_code = response.choices[0].message.content
        
        return fixed_code

    def generate_code_description(self, code, context=""):
        """Generate a description for a code cell based on its content"""
        prompt = f"""Generate 1-2 sentences describing the goal of the code, what it is doing, and why.

        Code:
        ```python
        {code}
        ```
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a single-cell bioinformatics expert providing concise code descriptions."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()

    def interpret_results(self, notebook, past_analyses):
        # Get the last cell
        last_cell = notebook.cells[-1]
        no_interpretation = "No results found"

        if last_cell.get('cell_type') != 'code':
            print("Last cell is not a code cell")
            return no_interpretation
        
        #### Extract text output ####
        testing = False
        text_output = ""
        if 'outputs' in last_cell:
            for output in last_cell['outputs']:
                if output.get('output_type') == 'stream': # print statements
                    text_output += output.get('text', '')
                elif output.get('output_type') == 'execute_result': # variable outputs e.g. df.head()
                    text_output += str(output.get('data', {}).get('text/plain', ''))

        if testing:
            print("TEXT OUTPUT: ", text_output)
        
        #### Extract image outputs (if using VLM) ####
        if self.use_VLM:
            image_outputs = []
            if 'outputs' in last_cell:
                for i, output in enumerate(last_cell['outputs']):
                    if output.get('output_type') == 'display_data':
                        image_data = output.get('data', {}).get('image/png')
                        if image_data:
                            # Save image to file for testing
                            if testing:
                                try:
                                    img_bytes = base64.b64decode(image_data)
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    img_path = os.path.join(self.output_dir, f'debug_image_{timestamp}_{i}.png')
                                    with open(img_path, 'wb') as f:
                                        f.write(img_bytes)
                                    print(f"Saved debug image to {img_path}")
                                except Exception as e:
                                    print(f"Error saving debug image {i}: {str(e)}")
                                
                            image_outputs.append({
                                'data': image_data,
                                'format': 'image/png'
                            })

            if not text_output and not image_outputs: # no output found
                return no_interpretation
        else:
            if not text_output:
                return no_interpretation
        
        prompt = open(os.path.join(self.prompt_dir, "interp_results.txt")).read()
        prompt = prompt.format(text_output=text_output, paper_txt=self.paper_summary,
                               CODING_GUIDELINES=self.coding_guidelines, past_analyses=past_analyses)

        if self.use_VLM:
            user_content = []
            user_content.append({"type": "text", "text": prompt})
            try:
                for img in image_outputs:
                    try:
                        # Get the image data 
                        image_data = img['data']
                        
                        # Remove the base64 prefix if present
                        if isinstance(image_data, str) and "," in image_data:
                            image_data = image_data.split(",")[1]
                        
                        # Add the image to the content
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        })
                    except Exception as e:
                        print(f"Warning: Error processing image: {str(e)}")
                        continue  # Skip this image and continue with others
                        
                response = self.client.chat.completions.create(
                    model = "gpt-4o",
                    messages = [
                        {"role": "system", "content": "You are a single-cell transcriptomics expert providing feedback on Python code and analysis plan."},
                        {"role": "user", "content": user_content}
                    ]
                )
                feedback = response.choices[0].message.content
                self.logger.log_prompt("user", text_output, "Results Interpretation")
            finally:
                # Clean up image data
                image_outputs.clear()
                user_content.clear()
        else:
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = [
                    {"role": "system", "content": "You are a single-cell bioinformatics expert providing feedback on Python code and analysis plan."},
                    {"role": "user", "content": prompt}
                ]
            )
            feedback = response.choices[0].message.content
            self.logger.log_prompt("user", text_output, "Results Interpretation")
            
        return feedback
    
    def get_feedback(self, analysis, past_analyses, notebook_cells, iterations=1):
        current_analysis = analysis
        for i in range(iterations):
            feedback = self.critique_step(current_analysis, past_analyses, notebook_cells)
            current_analysis = self.incorporate_critique(current_analysis, feedback, notebook_cells)

        return current_analysis

    def create_ideas(self):
        past_analyses = ""
        analyses = []
        for analysis_idx in range(self.num_analyses):
            print(f"\n🚀 Starting Analysis {analysis_idx+1}")

            analysis = self.generate_initial_analysis(past_analyses)

            modified_analysis = self.get_feedback(analysis, past_analyses, None)
            summary = modified_analysis["summary"]

            past_analyses += f"{summary}\n"
            analyses.append(summary)

        return analyses

    def cleanup(self):
        """Clean up resources, including the persistent kernel"""
        self.stop_persistent_kernel()

    def start_persistent_kernel(self):
        """Start a persistent kernel for efficient cell execution"""
        try:
            # Create kernel manager
            self.kernel_manager = KernelManager(kernel_name='python3')
            self.kernel_manager.start_kernel()
            
            # Create kernel client
            self.kernel_client = self.kernel_manager.client()
            self.kernel_client.start_channels()
            self.kernel_client.wait_for_ready()
            
            print("✅ Persistent kernel started")
            return True
        except Exception as e:
            print(f"⚠️ Failed to start persistent kernel: {str(e)}")
            return False
    
    def stop_persistent_kernel(self):
        """Stop the persistent kernel"""
        if self.kernel_client:
            self.kernel_client.stop_channels()
        if self.kernel_manager:
            self.kernel_manager.shutdown_kernel()
        print("✅ Persistent kernel stopped")
    

    def run_last_cell(self, nb):
        """Executes the most recently added code cell and updates its outputs."""
        if not nb.cells:
            raise ValueError("No cells in notebook to run.")

        # Find the last code cell
        last_code_cell = None
        for cell in reversed(nb.cells):
            if cell.cell_type == 'code':
                last_code_cell = cell
                break
        
        if not last_code_cell:
            raise ValueError("No code cells found in notebook.")
            
        code = last_code_cell.source
        #print("Running code: ", code)
        msg_id = self.kernel_client.execute(code)
        outputs = []

        while True:
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=300)
            except Exception:
                break

            msg_type = msg['msg_type']
            content = msg['content']
            
            # Check for execution completion first (before parent header filter)
            if msg_type == 'status' and content.get('execution_state') == 'idle':
                break

            # Filter messages that don't belong to our execution
            if msg['parent_header'].get('msg_id') != msg_id:
                continue

            if msg_type == 'stream':
                outputs.append(new_output(output_type='stream', name=content['name'], text=content['text']))
            elif msg_type == 'execute_result':
                outputs.append(new_output(output_type='execute_result',
                                        data=content['data'],
                                        execution_count=content['execution_count']))
            elif msg_type == 'display_data':
                outputs.append(new_output(output_type='display_data',
                                        data=content['data'],
                                        metadata=content.get('metadata', {})))
            elif msg_type == 'error':
                outputs.append(new_output(output_type='error',
                                        ename=content['ename'],
                                        evalue=content['evalue'],
                                        traceback=content['traceback']))

        # Update outputs in the last code cell by finding its index
        #last_code_cell.outputs = outputs
        code_cell_index = nb.cells.index(last_code_cell)
        nb.cells[code_cell_index].outputs = outputs

        # Check for errors
        for output in outputs:
            if output.output_type == "error":
                error_msg = f"{output.ename}: {output.evalue}"
                return False, error_msg, nb

        return True, None, nb

    def run(self, max_fix_attempts=3):
        # TODO: incorporate previous code in self.fix_code for cases where the fix depends on previous code cells
        past_analyses = ""

        for analysis_idx in range(self.num_analyses):
            # Reset code memory for this analysis
            self.code_memory = []
            
            print(f"\n🚀 Starting Analysis {analysis_idx+1}")

            # Start the persistent kernel for this analysis
            if not self.start_persistent_kernel():
                print(f"⚠️ Failed to start kernel for analysis {analysis_idx+1}. Skipping...")
                continue

            # Create the intitial analysis plan
            analysis = self.generate_initial_analysis(past_analyses)
            hypothesis = analysis["hypothesis"]                
            analysis_plan = analysis["analysis_plan"]
            initial_code = analysis["first_step_code"]
            
            # Log only the output of the analysis
            self.logger.log_response(f"Hypothesis: {hypothesis}\n\nAnalysis Plan:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)]) + f"\n\nInitial Code:\n{initial_code}", "initial_analysis")
            
            # Get feedback for the initial analysis plan and modify it accordingly
            if self.use_self_critique:
                self.logger.log_response(f"APPLYING INITIAL SELF-CRITIQUE - Analysis {analysis_idx+1}", "initial_self_critique")
                modified_analysis = self.get_feedback(analysis, past_analyses, None)
            else:
                print("🚫 Skipping feedback on next step (no self-critique)")
                self.logger.log_response(f"SKIPPING INITIAL SELF-CRITIQUE - Analysis {analysis_idx+1}", "no_initial_self_critique")
                modified_analysis = analysis
            hypothesis = modified_analysis["hypothesis"]                
            analysis_plan = modified_analysis["analysis_plan"]
            current_code = modified_analysis["first_step_code"]

            print(f"🚀 Generated Initial Analysis Plan for Analysis {analysis_idx+1}")
            
            # Log revised analysis plan
            self.logger.log_response(f"Revised Hypothesis: {hypothesis}\n\nRevised Analysis Plan:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)]) + f"\n\nRevised Code:\n{current_code}", "revised_analysis")
            
            # Create a markdown cell with the analysis plan
            plan_markdown = "# Analysis Plan\n\n**Hypothesis**: " + hypothesis + "\n\n## Steps:\n"
            for step in analysis_plan:
                plan_markdown += f"- {step}\n"

            # Create initial notebook with the hypothesis and plan
            notebook = self.create_initial_notebook(hypothesis)

            # Run the setup code (pre-specified)
            _, _, notebook = self.run_last_cell(notebook)
            
            # Add the analysis plan as a markdown cell
            notebook.cells.append(nbf.v4.new_markdown_cell(plan_markdown))
            
            # Add a markdown cell for the first step description
            if analysis_plan:
                notebook.cells.append(nbf.v4.new_markdown_cell(f"## {modified_analysis['code_description']}"))
            
            # Add the first analysis code cell
            current_code = strip_code_markers(current_code)
            notebook.cells.append(new_code_cell(current_code))

            for iteration in range(self.max_iterations):
                # Execute the notebook
                #success, error_msg, notebook = self.execute_notebook(notebook)
                success, error_msg, notebook = self.run_last_cell(notebook)
                print(f"🚀 Beginning step {iteration + 1}...")
                
                # Log step execution attempt
                self.logger.log_response(f"STEP {iteration + 1} EXECUTION ATTEMPT - Analysis {analysis_idx+1}", "step_execution")

                if success:
                    results_interpretation = self.interpret_results(notebook, past_analyses)
                    # Log the interpretation
                    self.logger.log_response(results_interpretation, "results_interpretation")
                    # Add interpretation as a markdown cell
                    interpretation_cell = nbf.v4.new_markdown_cell(f"### Agent Interpretation\n\n{results_interpretation}")
                    notebook.cells.append(interpretation_cell)

                else:
                    print(f"⚠️ Code errored with: {error_msg}")
                    self.logger.log_error(error_msg, current_code)
                    fix_attempt, fix_successful = 0, False
                    results_interpretation = ""  # Initialize at start of error block
                    while fix_attempt < max_fix_attempts and not fix_successful:
                        fix_attempt += 1
                        print(f"  🔧 Fix attempt {fix_attempt}/{max_fix_attempts}")
                        
                        # Log fix attempt start
                        self.logger.log_response(f"FIX ATTEMPT {fix_attempt}/{max_fix_attempts} - Analysis {analysis_idx+1}, Step {iteration + 1}", "fix_attempt_start")

                        current_code = self.fix_code(current_code, error_msg)
                        current_code = strip_code_markers(current_code)
                        notebook.cells[-1] = nbf.v4.new_code_cell(current_code)

                        success, error_msg, notebook = self.run_last_cell(notebook)

                        if success:
                            fix_successful = True
                            print(f"  ✅ Fix successful on attempt {fix_attempt}")
                            
                            # Log successful fix
                            self.logger.log_response(f"FIX SUCCESSFUL on attempt {fix_attempt}/{max_fix_attempts} - Analysis {analysis_idx+1}, Step {iteration + 1}", "fix_attempt_success")
                            
                            # Generate updated code description for the fixed code
                            updated_description = self.generate_code_description(current_code)
                            
                            # Update the previous markdown cell with the corrected description
                            # Find the last markdown cell that contains a code description (starts with "##")
                            for i in range(len(notebook.cells) - 1, -1, -1):
                                if (notebook.cells[i].cell_type == 'markdown' and 
                                    notebook.cells[i].source.startswith('##') and 
                                    'Agent Interpretation' not in notebook.cells[i].source):
                                    notebook.cells[i].source = f"## {updated_description}"
                                    break
                            
                            results_interpretation = self.interpret_results(notebook, past_analyses)
                            # Log the interpretation
                            self.logger.log_response(results_interpretation, "results_interpretation")
                            # Add interpretation as a markdown cell
                            interpretation_cell = nbf.v4.new_markdown_cell(f"### Agent Interpretation\n\n{results_interpretation}")
                            notebook.cells.append(interpretation_cell)
                            break
                        else:
                            print(f"  ❌ Fix attempt {fix_attempt} failed")
                            
                            # Log failed fix attempt with error details
                            self.logger.log_error(f"FIX ATTEMPT FAILED {fix_attempt}/{max_fix_attempts} - Analysis {analysis_idx+1}, Step {iteration + 1}: {error_msg}", current_code)

                            if fix_attempt == max_fix_attempts:
                                print(f"  ⚠️ Failed to fix after {max_fix_attempts} attempts. Moving to next iteration.")
                                
                                # Log final failure
                                self.logger.log_error(f"ALL FIX ATTEMPTS EXHAUSTED - Analysis {analysis_idx+1}, Step {iteration + 1}. Failed after {max_fix_attempts} attempts.", current_code)
                                
                                results_interpretation = "Current analysis step failed to run. Try an alternative approach"
                                interpretation_cell = nbf.v4.new_markdown_cell(f"### Agent Interpretation\n\n{results_interpretation}")
                                notebook.cells.append(interpretation_cell)
                    if not results_interpretation:  # Only get interpretation if we haven't set the failure message
                        results_interpretation = self.interpret_results(notebook, past_analyses)
                        interpretation_cell = nbf.v4.new_markdown_cell(f"### Agent Interpretation\n\n{results_interpretation}")
                        notebook.cells.append(interpretation_cell)

                # Only generate next step if this is not the final iteration
                if iteration < self.max_iterations - 1:
                    analysis = {"hypothesis": hypothesis, "analysis_plan": analysis_plan, "first_step_code": current_code}
                    num_steps_left = self.max_iterations - iteration - 1
                    next_step_analysis = self.generate_next_step_analysis(analysis, past_analyses, notebook.cells, results_interpretation, num_steps_left)

                    # Log next step generation
                    self.logger.log_response(f"GENERATING NEXT STEP - Analysis {analysis_idx+1}, Step {iteration + 2}", "next_step_generation")

                    # Get feedback on the next step(s) (only if self-critique is enabled)
                    if self.use_self_critique:
                        print("Getting feedback on the next step(s)")
                        self.logger.log_response(f"APPLYING SELF-CRITIQUE - Analysis {analysis_idx+1}, Step {iteration + 2}", "self_critique")
                        modified_analysis = self.get_feedback(next_step_analysis, past_analyses, notebook.cells)
                    else:
                        print("🚫 Skipping feedback on next step (no self-critique)")
                        self.logger.log_response(f"SKIPPING SELF-CRITIQUE - Analysis {analysis_idx+1}, Step {iteration + 2}", "no_self_critique")
                        modified_analysis = next_step_analysis
                    
                    # Log the next step
                    self.logger.log_response(f"NEXT STEP PLAN - Analysis {analysis_idx+1}, Step {iteration + 2}: {modified_analysis['analysis_plan'][0]}\n\nCode:\n```python\n{modified_analysis['first_step_code']}\n```", "next_step")

                    # Add the next step to the notebook
                    code_description = modified_analysis["code_description"]
                    notebook.cells.append(nbf.v4.new_markdown_cell(f"## {code_description}"))
                    modified_code = strip_code_markers(modified_analysis["first_step_code"])
                    notebook.cells.append(new_code_cell(modified_code))
                    
                    # Update current_code for next iteration
                    current_code = modified_code
                    
                # Update the code memory with the current notebook state
                self.update_code_memory(notebook.cells)

            # Save the notebook
            notebook_path = os.path.join(self.output_dir, f"{self.analysis_name}_analysis_{analysis_idx+1}.ipynb")
            with open(notebook_path, 'w', encoding='utf-8') as f:
                # Clean notebook outputs before writing
                clean_notebook = self.cleanup_notebook_outputs(notebook)
                nbf.write(clean_notebook, f)
                print(f"💾 Saved notebook to: {notebook_path}")

            # Log analysis completion
            self.logger.log_response(f"ANALYSIS {analysis_idx+1} COMPLETED - Notebook saved to: {notebook_path}", "analysis_complete")

            self.stop_persistent_kernel()
            print(f"✅ Completed Analysis {analysis_idx+1}")

            # TODO: modify this to include the entire analysis plan
            past_analyses += f"{hypothesis}\n"


        # Clean up resources
        self.cleanup() # Could remove this I think

    def create_initial_notebook(self, hypothesis):
        notebook = nbf.v4.new_notebook()
        
        # Add markdown cell with hypothesis
        notebook.cells.append(nbf.v4.new_markdown_cell(f"# Analysis\n\n**Hypothesis**: {hypothesis}"))
        
        # Add setup code to import libraries and load data with enhanced visualization setup
        setup_code = f"""import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Set up visualization defaults for better plots
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.figsize = (8, 8)
sc.settings.dpi = 100
sc.settings.facecolor = 'white'
warnings.filterwarnings('ignore')

# Set Matplotlib and Seaborn styles for better visualization
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['savefig.dpi'] = 150
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.2)

# Load data
print("Loading data...")
adata = sc.read_h5ad("{self.h5ad_path}")
print(f"Data loaded: {{adata.shape[0]}} cells and {{adata.shape[1]}} genes")
"""
        notebook.cells.append(nbf.v4.new_code_cell(setup_code))
        
        self.logger.log_action("Created initial notebook", f"Setup code:\n```python\n{setup_code}\n```")
        return notebook

    def cleanup_notebook_outputs(self, notebook):
        """Clean notebook outputs to ensure they are proper nbformat objects"""
        for cell in notebook.cells:
            if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
                cleaned_outputs = []
                for output in cell.outputs:
                    if isinstance(output, dict):
                        # Convert dict to proper nbformat output
                        if output.get('output_type') == 'stream':
                            cleaned_outputs.append(nbf.v4.new_output('stream', 
                                                                    name=output.get('name', 'stdout'),
                                                                    text=output.get('text', '')))
                        elif output.get('output_type') == 'execute_result':
                            cleaned_outputs.append(nbf.v4.new_output('execute_result',
                                                                    data=output.get('data', {}),
                                                                    execution_count=output.get('execution_count', None)))
                        elif output.get('output_type') == 'display_data':
                            cleaned_outputs.append(nbf.v4.new_output('display_data',
                                                                    data=output.get('data', {})))
                        elif output.get('output_type') == 'error':
                            cleaned_outputs.append(nbf.v4.new_output('error',
                                                                    ename=output.get('ename', ''),
                                                                    evalue=output.get('evalue', ''),
                                                                    traceback=output.get('traceback', [])))
                    else:
                        # Already a proper object
                        cleaned_outputs.append(output)
                cell.outputs = cleaned_outputs
        
        return notebook


    def improve_notebook(self, notebook_path, feedback, output_path=None):
        """
        Improve a notebook using o3-mini based on expert feedback
        
        Args:
            notebook_path (str): Path to the notebook to improve
            feedback (str): Expert feedback for improvements
            output_path (str, optional): Output path for improved notebook
        
        Returns:
            str: Path to the improved notebook
        """
        # Set default output path
        if not output_path:
            base_name = os.path.splitext(os.path.basename(notebook_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_improved.ipynb")
        
        # Copy original notebook
        shutil.copy2(notebook_path, output_path)
        self.logger.log_action("Notebook copied for improvement", f"From: {notebook_path}\nTo: {output_path}")
        
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbf.read(f, as_version=4)
        
        # Extract content for context
        content = "Notebook content:\n\n"
        
        # Get all cells
        for i, cell in enumerate(notebook.cells):
            cell_num = i + 1
            if cell.cell_type == 'markdown':
                content += f"Markdown Cell {cell_num}:\n{cell.source}\n\n"
            elif cell.cell_type == 'code':
                content += f"Code Cell {cell_num}:\n{cell.source}\n\n"
                # Capture cell output
                if hasattr(cell, 'outputs'):
                    content += "Cell Output:\n"
                    for output in cell.outputs:
                        if output.get('output_type') == 'stream':  # print statements
                            content += output.get('text', '')
                        elif output.get('output_type') == 'execute_result':  # variable outputs
                            content += str(output.get('data', {}).get('text/plain', ''))
                    content += "\n"

        # Create prompt for o3-mini
        prompt = f"""

        You will be given a Jupyter notebook that execute a single-cell transcriptomics comptuational analysis.
        You will also be given expert feedback for how to improve the analysis.
        Your role is to generate python code for the new code cell that implements the suggested improvements.

        ONLY RETURN THE PYTHON CODE. NOTHING ELSE.

        Current Jupyter notebook content:
        {content}

        Expert feedback for improvement:
        {feedback}

        {self.coding_guidelines}

        You are given the following summary fo the anndata object:
        {self.adata_summary}
        """
        
        self.logger.log_prompt("user", prompt, "Notebook Improvement")
        
        # Try o3-mini first
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a coding assistant helping to improve single-cell analysis notebooks."},
                {"role": "user", "content": prompt}
            ]
        )
        new_code = response.choices[0].message.content.strip()
        
        # Clean code and add to notebook
        new_code = strip_code_markers(new_code)
        
        # Add feedback as markdown cell
        feedback_cell = nbf.v4.new_markdown_cell(f"## Expert Feedback\n\n{feedback}")
        notebook.cells.append(feedback_cell)
        
        # Add improvement description
        improvement_cell = nbf.v4.new_markdown_cell("## Improvement Implementation")
        notebook.cells.append(improvement_cell)
        
        # Add new code cell
        code_cell = nbf.v4.new_code_cell(new_code)
        notebook.cells.append(code_cell)
        
        # Save improved notebook
        with open(output_path, 'w', encoding='utf-8') as f:
            # Clean notebook outputs before writing
            clean_notebook = self.cleanup_notebook_outputs(notebook)
            nbf.write(clean_notebook, f)
        
        # Execute the new cell
        print("Executing improved code cell...")
        success, error_msg, executed_notebook = self.run_last_cell(notebook)
        
        if success:
            # Interpret results
            results_interpretation = self.interpret_results(executed_notebook, "")
            self.logger.log_response(results_interpretation, "improvement results")
            
            # Add interpretation
            interpretation_cell = nbf.v4.new_markdown_cell(f"### Results\n\n{results_interpretation}")
            executed_notebook.cells.append(interpretation_cell)
            
            # Save final notebook
            with open(output_path, 'w', encoding='utf-8') as f:
                # Clean notebook outputs before writing
                clean_notebook = self.cleanup_notebook_outputs(executed_notebook)
                nbf.write(clean_notebook, f)
            
            print(f"✅ Notebook improved successfully: {output_path}")
            
        else:
            fixed_code = new_code
            for attempt in range(3):
                print(f"⚠️ {'Improved code had errors' if attempt == 0 else f'Fix attempt {attempt} failed'}: {error_msg}")
                fixed_code = self.fix_code(fixed_code, error_msg, other_code = content)
                fixed_code = strip_code_markers(fixed_code)
                notebook.cells[-1] = nbf.v4.new_code_cell(fixed_code)
                
                success, error_msg, executed_notebook = self.run_last_cell(notebook)
                if success:
                    break
            
            if success:
                # Generate updated code description for the fixed code
                updated_description = self.generate_code_description(fixed_code)
                
                # Update the improvement description cell
                for i in range(len(executed_notebook.cells) - 1, -1, -1):
                    if (executed_notebook.cells[i].cell_type == 'markdown' and 
                        executed_notebook.cells[i].source.startswith('## Improvement Implementation')):
                        executed_notebook.cells[i].source = f"## Improvement Implementation\n\n{updated_description}"
                        break
                
                results_interpretation = self.interpret_results(executed_notebook, "")
                interpretation_cell = nbf.v4.new_markdown_cell(f"### Results \n\n{results_interpretation}")
                executed_notebook.cells.append(interpretation_cell)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    # Clean notebook outputs before writing
                    clean_notebook = self.cleanup_notebook_outputs(executed_notebook)
                    nbf.write(clean_notebook, f)
                print(f"✅ Notebook improved after fix: {output_path}")
            else:
                print(f"❌ Could not fix improved code after 3 attempts: {error_msg}")
                self.logger.log_error(f"Improvement failed after 3 attempts: {error_msg}", fixed_code)
        
        return output_path

def strip_code_markers(text):
    # Remove ```python, ``` and ```
    return re.sub(r'```python|```', '', text)
