import openai
import os
import json
import nbformat as nbf
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np
import datetime
from logger import Logger
import base64
import h5py
from h5py import Dataset, Group
import re
import shutil
from jupyter_client import KernelManager
from nbformat.v4 import new_code_cell, new_output
from deepresearch import DeepResearcher
from utils import get_documentation

AVAILABLE_PACKAGES = "scanpy, scvi, anndata, matplotlib, numpy, seaborn, pandas, scipy"
class AnalysisAgent:
    def __init__(self, h5ad_path, paper_summary_path, openai_api_key, model_name, analysis_name, 
                num_analyses=5, max_iterations=6, prompt_dir="prompts", output_home=".", log_home=".",
                use_self_critique=True, use_VLM=True, use_documentation=True, log_prompts = False,
                max_fix_attempts=3, use_deepresearch_background=True):
        self.h5ad_path = h5ad_path
        self.paper_summary = open(paper_summary_path).read()
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.analysis_name = analysis_name
        self.max_iterations = max_iterations
        self.num_analyses = num_analyses
        self.prompt_dir = prompt_dir
        self.log_prompts = log_prompts
        self.max_fix_attempts = max_fix_attempts
        self.use_deepresearch_background = use_deepresearch_background
        
        # Create unique output directory based on analysis name and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_home, "outputs", f"{analysis_name}_{timestamp}")
        
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize code memory to track the last few cells of code
        self.code_memory = []
        self.code_memory_size = 5  # Number of code cells to remember

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Primarily for ablation studies
        self.use_self_critique = use_self_critique
        self.use_VLM = use_VLM
        self.use_documentation = use_documentation

        # Coding guidelines: guide agent on how to write code and conduct analyses
        self._analyses_overview = open(os.path.join(self.prompt_dir, "DeepResearch_Analyses.txt")).read()
        if self.use_VLM:
            self.coding_guidelines_template = open(os.path.join(self.prompt_dir, "coding_guidelines.txt")).read()
        else:
            self.coding_guidelines_template = open(os.path.join(self.prompt_dir, "ablations", "coding_guidelines_NO_VLM_ABLATION.txt")).read()

        # System prompt for coding agents
        self.coding_system_prompt = open(os.path.join(self.prompt_dir, "coding_system_prompt.txt")).read().format(max_iterations=self.max_iterations)

        # Finalize coding guidelines with the (possibly augmented) overview
        self.coding_guidelines = self.coding_guidelines_template.format(
            name=self.analysis_name,
            adata_path=self.h5ad_path,
            available_packages=AVAILABLE_PACKAGES,
            analyses_overview=self._analyses_overview,
        )

        # Initialize logger: keeps track of all actions, prompts, responses, errors, etc.
        self.logger = Logger(self.analysis_name, log_dir=os.path.join(log_home, "logs"))
        # Initialize notebook executor
        self.executor = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        # Initialize persistent kernel for efficient cell execution
        self.kernel_manager = None
        self.kernel_client = None

        # Load the .obs data from the anndata file
        if self.h5ad_path == "": # JUST FOR BENCHMARKING
            self.adata_summary = ""
        else:
            print("Loading anndata .obs for summarization...")
            self.adata_obs = self.load_h5ad_obs(self.h5ad_path)
            self.adata_summary = self.summarize_adata_metadata()
            print("ADATA SUMMARY: ", self.adata_summary)
            print(f"✅ Loaded {self.h5ad_path}")

        if self.use_deepresearch_background:
            # DeepResearch for idea generation
            print("Running DeepResearch...")
            try:
                deepresearch = DeepResearcher(self.openai_api_key)

                # Always initialize background string so attribute exists even if DeepResearch fails
                self.deepresearch_background = ""

                # Provide both the paper summary and dataset metadata so deep research can tailor background
                dr_summary = deepresearch.research_from_paper_summary(self.paper_summary, self.adata_summary, AVAILABLE_PACKAGES)
                self.deepresearch_background = dr_summary.strip()
                print("✅ DeepResearch completed")
                print("DEEPRESEARCH BACKGROUND: ", self.deepresearch_background[:100])
            except Exception as e:
                print(f"Warning: DeepResearch failed or was skipped: {e}")
        


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

    def generate_jupyter_summary(self, notebook_cells):
        """Generate a comprehensive summary of notebook code cells, markdown, and errors"""
        if notebook_cells is None:
            return ""
            
        jupyter_summary = ""
        for cell in notebook_cells:
            if cell['cell_type'] == 'code' or cell['cell_type'] == 'markdown' or cell['cell_type'] == 'error':
                jupyter_summary += f"{cell['source']}\n"
        
        return jupyter_summary

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
                               past_analyses=attempted_analyses, paper_txt=self.paper_summary,
                               deepresearch_background=self.deepresearch_background if self.use_deepresearch_background else "")

        if self.log_prompts:
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
        
        # Debug logging for API response issues
        if result is None:
            print(f"⚠️ API returned None response in generate_initial_analysis")
            print(f"   Model: {self.model_name}")
            print(f"   Response object: {response}")
            # Check if this is a refusal
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                print(f"   Refusal reason: {response.choices[0].message.refusal}")
                raise ValueError(f"OpenAI API refused to generate response: {response.choices[0].message.refusal}")
            else:
                raise ValueError("OpenAI API returned None response for initial analysis")
        
        try:
            analysis = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error in generate_initial_analysis: {e}")
            print(f"   Raw result: {repr(result)}")
            raise
        
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

        jupyter_summary = self.generate_jupyter_summary(notebook_cells)
        
        # Use the code memory for generating the next step
        recent_code = "\n\n# Next Cell\n".join(reversed(self.code_memory))

        if seeded:
            prompt = open(os.path.join(self.prompt_dir, "next_step_seeded.txt")).read()
            prompt = prompt.format(hypothesis=hypothesis, analysis_plan = analysis_plan, num_steps_left=num_steps_left,
                                 CODING_GUIDELINES=self.coding_guidelines, jupyter_notebook=jupyter_summary,
                                 jupyter_notebook=jupyter_summary, adata_summary=self.adata_summary, past_analyses=attempted_analyses,
                                 paper_txt=self.paper_summary)
        else:
            prompt = open(os.path.join(self.prompt_dir, "next_step.txt")).read()
            prompt = prompt.format(hypothesis=hypothesis, analysis_plan=analysis_plan,
                                CODING_GUIDELINES=self.coding_guidelines, jupyter_notebook=jupyter_summary,
                                adata_summary=self.adata_summary, past_analyses=attempted_analyses,
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

        # Debug logging for API response issues
        if result is None:
            print(f"⚠️ API returned None response in generate_next_step")
            print(f"   Model: {self.model_name}")
            print(f"   Response object: {response}")
            # Check if this is a refusal
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                print(f"   Refusal reason: {response.choices[0].message.refusal}")
                raise ValueError(f"OpenAI API refused to generate response: {response.choices[0].message.refusal}")
            else:
                raise ValueError("OpenAI API returned None response for next step")
        
        try:
            analysis = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error in generate_next_step: {e}")
            print(f"   Raw result: {repr(result)}")
            raise
        
        return analysis

    def critique_step(self, analysis, past_analyses, notebook_cells):
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]

        jupyter_summary = self.generate_jupyter_summary(notebook_cells)

        if notebook_cells is None:
            recent_code = ""
        else:
            # Update code memory with latest notebook cells
            self.update_code_memory(notebook_cells)
            
            # Use the code memory for generating the next step
            recent_code = "\n\n# Next Cell\n".join(reversed(self.code_memory))

        if self.use_documentation:
            prompt = open(os.path.join(self.prompt_dir, "critic.txt")).read()
            # Get relevant documentation on the single-cell packages being used in the first step code
            try:
                documentation = get_documentation(first_step_code)
            except Exception as e:
                print(f"⚠️ Documentation extraction failed: {e}")
                documentation = ""
            prompt = prompt.format(hypothesis=hypothesis, analysis_plan=analysis_plan, first_step_code=first_step_code,
                                CODING_GUIDELINES=self.coding_guidelines, adata_summary=self.adata_summary, past_analyses=past_analyses,
                                paper_txt=self.paper_summary, jupyter_notebook=jupyter_summary, documentation=documentation)
        else:
            prompt = open(os.path.join(self.prompt_dir, "ablations", "critic_NO_DOCUMENTATION.txt")).read()
            prompt = prompt.format(hypothesis=hypothesis, analysis_plan=analysis_plan, first_step_code=first_step_code,
                                CODING_GUIDELINES=self.coding_guidelines, adata_summary=self.adata_summary, past_analyses=past_analyses,
                                paper_txt=self.paper_summary, jupyter_notebook=jupyter_summary)

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

        #recent_code = "\n\n# Next Cell\n".join(reversed(self.code_memory))
        jupyter_summary = self.generate_jupyter_summary(notebook_cells)

        prompt = open(os.path.join(self.prompt_dir, "incorporate_critque.txt")).read()
        prompt = prompt.format(hypothesis=hypothesis, analysis_plan=analysis_plan, first_step_code=first_step_code,
                               CODING_GUIDELINES=self.coding_guidelines, adata_summary=self.adata_summary,
                               feedback=feedback, jupyter_notebook=jupyter_summary)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        
        # Debug logging for API response issues
        if result is None:
            print(f"⚠️ API returned None response in incorporate_critique")
            print(f"   Model: {self.model_name}")
            print(f"   Response object: {response}")
            # Check if this is a refusal
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                print(f"   Refusal reason: {response.choices[0].message.refusal}")
                raise ValueError(f"OpenAI API refused to generate response: {response.choices[0].message.refusal}")
            else:
                raise ValueError("OpenAI API returned None response for critique incorporation")
        
        try:
            modified_analysis = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error in incorporate_critique: {e}")
            print(f"   Raw result: {repr(result)}")
            raise

        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "Incorporate Critiques")

        return modified_analysis
    
    def fix_code(self, code, error, other_code="", documentation=""):
        """Attempts to fix code that produced an error"""
        
        # Manage context length for fix_code to prevent token limit errors
        max_error_chars = 2000          # ~500 tokens
        max_other_code_chars = 3000     # ~750 tokens  
        max_past_context_chars = 4000   # ~1000 tokens
        max_documentation_chars = 3000  # ~750 tokens
        
        # Truncate error message (keep end as it's usually most relevant)
        truncated_error = error[-max_error_chars:] if len(error) > max_error_chars else error
        if len(error) > max_error_chars:
            truncated_error = "...(error truncated)...\n" + truncated_error
        
        # Truncate other_code context
        truncated_other_code = other_code[-max_other_code_chars:] if len(other_code) > max_other_code_chars else other_code
        if len(other_code) > max_other_code_chars:
            truncated_other_code = "...(context truncated)...\n" + truncated_other_code
        
        # Get the last 5 code cells (reduced from 8) for context
        past_code_context = ""
        if self.code_memory:
            past_cells = self.code_memory[-5:]  # Reduced from 8 to 5 cells
            past_code_context = "\n\n".join([f"# Previous code cell {i+1}:\n{cell}" for i, cell in enumerate(past_cells)])
            # Truncate if still too long
            if len(past_code_context) > max_past_context_chars:
                past_code_context = past_code_context[-max_past_context_chars:]
                past_code_context = "...(context truncated)...\n" + past_code_context
        
        # Truncate documentation
        truncated_documentation = documentation[-max_documentation_chars:] if len(documentation) > max_documentation_chars else documentation
        if len(documentation) > max_documentation_chars:
            truncated_documentation = "...(documentation truncated)...\n" + truncated_documentation
        
        # Build the base prompt
        prompt = f"""Fix this code that produced an error:
        
        Code:
        ```python
        {code}
        ```
        
        Error:
        {truncated_error}

        Provide only the fixed code with no explanation.
        You can only use the following packages: {AVAILABLE_PACKAGES}

        Here is previous code/context (if any):
        {truncated_other_code}
        
        Here are the past code cells for additional context (last 5 cells):
        {past_code_context}"""
        
        # Only add documentation section if use_documentation is enabled
        if self.use_documentation and truncated_documentation:
            prompt += f"""

        Finally, here is documentation about some of the functions being called, ensure that the code is using the proper parameters/functions:
        {truncated_documentation}"""
        
        # Check prompt length before sending
        estimated_tokens = len(prompt) // 4  # Rough estimation
        if estimated_tokens > 50000:  # Conservative limit for fix_code
            print(f"⚠️ Warning: Large fix_code prompt detected ({estimated_tokens} estimated tokens)")
        
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

    def interpret_results(self, notebook, past_analyses, hypothesis, analysis_plan, code):
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
                               CODING_GUIDELINES=self.coding_guidelines, past_analyses=past_analyses,
                               hypothesis=hypothesis, analysis_plan=analysis_plan, code=code)

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
                if self.log_prompts:
                    self.logger.log_prompt("user", text_output, "Results Interpretation")
            finally:
                # Clean up image data to prevent memory leaks
                image_outputs.clear()
                user_content.clear()
                import gc
                gc.collect()
        else:
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = [
                    {"role": "system", "content": "You are a single-cell bioinformatics expert providing feedback on Python code and analysis plan."},
                    {"role": "user", "content": prompt}
                ]
            )
            feedback = response.choices[0].message.content
            if self.log_prompts:
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
        try:
            self.stop_persistent_kernel()
        except Exception as e:
            print(f"⚠️ Warning: Error during cleanup: {e}")

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
        """Stop the persistent kernel with proper error handling"""
        try:
            if self.kernel_client:
                try:
                    self.kernel_client.stop_channels()
                except Exception as e:
                    print(f"⚠️ Warning: Error stopping kernel channels: {e}")
            
            if self.kernel_manager:
                try:
                    self.kernel_manager.shutdown_kernel(now=True)
                except Exception as e:
                    print(f"⚠️ Warning: Error shutting down kernel: {e}")
            
            print("✅ Persistent kernel stopped")
        except Exception as e:
            print(f"⚠️ Warning: Error during kernel cleanup: {e}")
        finally:
            # Reset kernel references
            self.kernel_client = None
            self.kernel_manager = None
    

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

        # Set up strict timeout mechanism
        import time
        start_time = time.time()
        max_execution_time = 600  # 10 minutes maximum (600 seconds)
        message_timeout = 30  # 30 seconds timeout for individual messages
        timed_out = False
        
        while True:
            # Check if we've exceeded the overall maximum execution time
            elapsed_time = time.time() - start_time
            if elapsed_time > max_execution_time:
                print(f"⏰ Maximum execution time exceeded ({max_execution_time/60:.1f} minutes)")
                timed_out = True
                break
            
            # Calculate remaining time for this message
            remaining_time = max_execution_time - elapsed_time
            current_timeout = min(message_timeout, remaining_time)
            
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=current_timeout)
            except Exception as e:
                # Check if we still have time left
                if time.time() - start_time >= max_execution_time:
                    print(f"⏰ Timeout reached during message wait ({elapsed_time/60:.1f} minutes)")
                    timed_out = True
                    break
                continue

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

        # Add timeout message to outputs if execution timed out
        if timed_out:
            timeout_message = f"""
⏰ EXECUTION TIMEOUT OCCURRED

This code cell took longer than expected to complete (>{max_execution_time/60:.0f} minutes).
The execution may still be running in the background.

Consider these alternatives for the next analysis step:
1. Use faster algorithms or smaller parameter values
2. Subsample the data for computational efficiency  
3. Use simpler analysis methods
4. For model training: reduce max_epochs, enable early_stopping
5. Try a completely different analytical approach
"""
            
            # Add timeout message as stream output
            timeout_output = new_output(
                output_type='stream',
                name='stderr', 
                text=timeout_message
            )
            outputs.append(timeout_output)

        # Update outputs in the last code cell by finding its index
        #last_code_cell.outputs = outputs
        code_cell_index = nb.cells.index(last_code_cell)
        nb.cells[code_cell_index].outputs = outputs

        # Check for errors
        for output in outputs:
            if output.output_type == "error":
                error_msg = f"{output.ename}: {output.evalue}"
                return False, error_msg, nb

        # Final check: ensure kernel is actually idle before declaring success
        try:
            # Wait a bit and check if kernel is truly idle
            final_msg = self.kernel_client.get_iopub_msg(timeout=5)
            if final_msg['msg_type'] == 'status' and final_msg['content'].get('execution_state') == 'busy':
                print("⚠️ Warning: Kernel still appears busy after supposed completion")
        except:
            # Timeout is expected if kernel is truly idle
            pass

        # Return success even if timed out - the timeout message in outputs will guide the agent
        return True, None, nb

    def generate_idea(self, past_analyses, analysis_idx=None, seeded_hypothesis=None):
        """
        Phase 1: Idea Generation
        
        Args:
            past_analyses: String of past analysis summaries
            analysis_idx: Analysis index for logging (optional)
            seeded_hypothesis: Simple hypothesis string to guide AI generation (optional)
        
        Returns:
            dict: Analysis containing hypothesis, analysis_plan, first_step_code, etc.
        """
        if seeded_hypothesis is not None:
            print(f"🌱 Using seeded hypothesis: {seeded_hypothesis}")
            return self.generate_analysis_from_hypothesis(seeded_hypothesis, past_analyses, analysis_idx)
            
        print("🧠 Generating new analysis idea...")
        
        # Create the initial analysis plan
        analysis = self.generate_initial_analysis(past_analyses)
        
        if analysis_idx is not None:
            step_name = f"{analysis_idx+1}_1"
            hypothesis = analysis["hypothesis"]                
            analysis_plan = analysis["analysis_plan"]
            initial_code = analysis["first_step_code"]
            
            # Log only the output of the analysis
            self.logger.log_response(f"Hypothesis: {hypothesis}\n\nAnalysis Plan:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)]) + f"\n\nInitial Code:\n{initial_code}", f"initial_analysis_{step_name}")
        
        # Get feedback for the initial analysis plan and modify it accordingly
        if self.use_self_critique:
            modified_analysis = self.get_feedback(analysis, past_analyses, None)
            
            if analysis_idx is not None:
                self.logger.log_response(f"APPLIED INITIAL SELF-CRITIQUE - Analysis {analysis_idx+1}", f"self_critique_{step_name}")
                
                hypothesis = modified_analysis["hypothesis"]                
                analysis_plan = modified_analysis["analysis_plan"]
                current_code = modified_analysis["first_step_code"]

                # Log revised analysis plan
                self.logger.log_response(f"Revised Hypothesis: {hypothesis}\n\nRevised Analysis Plan:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)]) + f"\n\nRevised Code:\n{current_code}", f"revised_analysis_{step_name}")
            
            return modified_analysis
        else:
            if analysis_idx is not None:
                print("🚫 Skipping feedback on next step (no self-critique)")
                self.logger.log_response(f"SKIPPING INITIAL SELF-CRITIQUE - Analysis {analysis_idx+1}", f"no_self_critique_{step_name}")
            
            return analysis

    def generate_analysis_from_hypothesis(self, hypothesis, past_analyses, analysis_idx=None):
        """
        Generate an analysis plan from a simple hypothesis string using AI
        
        Args:
            hypothesis: Simple hypothesis string
            past_analyses: String of past analysis summaries
            analysis_idx: Analysis index for logging (optional)
            
        Returns:
            dict: Analysis containing hypothesis, analysis_plan, first_step_code, etc.
        """
        # Create a modified prompt that incorporates the seeded hypothesis
        prompt = open(os.path.join(self.prompt_dir, "ablations", "analysis_from_hypothesis.txt")).read()
        prompt = prompt.format(hypothesis=hypothesis, 
                               coding_guidelines=self.coding_guidelines,
                               adata_summary=self.adata_summary, 
                               paper_summary=self.paper_summary)



        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "Seeded Hypothesis Analysis")

        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        
        # Debug logging for API response issues
        if result is None:
            print(f"⚠️ API returned None response in generate_analysis_from_hypothesis")
            print(f"   Model: {self.model_name}")
            print(f"   Response object: {response}")
            # Check if this is a refusal
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                print(f"   Refusal reason: {response.choices[0].message.refusal}")
                raise ValueError(f"OpenAI API refused to generate response: {response.choices[0].message.refusal}")
            else:
                raise ValueError("OpenAI API returned None response for hypothesis analysis")
        
        try:
            analysis = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error in generate_analysis_from_hypothesis: {e}")
            print(f"   Raw result: {repr(result)}")
            raise
        
        # Ensure the hypothesis matches what was provided
        analysis["hypothesis"] = hypothesis
        
        # Log the seeded hypothesis analysis
        if analysis_idx is not None:
            step_name = f"{analysis_idx+1}_1"
            analysis_plan = analysis["analysis_plan"]
            initial_code = analysis["first_step_code"]
            
            # Log the seeded hypothesis analysis
            self.logger.log_response(f"Seeded Hypothesis: {hypothesis}\n\nGenerated Analysis Plan:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)]) + f"\n\nInitial Code:\n{initial_code}", f"seeded_hypothesis_{step_name}")
                
            return analysis

    def execute_idea(self, analysis, past_analyses, analysis_idx, seeded = False):
        """
        Phase 2: Idea Execution
        
        Args:
            analysis: Analysis dict from generate_idea phase
            past_analyses: String of past analysis summaries  
            analysis_idx: Analysis index for logging
            seeded: Boolean indicating if the analysis is seeded

        Returns:
            tuple: (hypotheses_analysis list, updated past_analyses string)
        """
        def namer(analysis_idx, step_idx):
            return f"{analysis_idx+1}_{step_idx}"
            
        hypotheses_analysis = []
        
        # Reset code memory for this analysis
        self.code_memory = []
        
        print(f"\n🚀 Executing Analysis {analysis_idx+1}")

        # Start the persistent kernel for this analysis
        if not self.start_persistent_kernel():
            print(f"⚠️ Failed to start kernel for analysis {analysis_idx+1}. Skipping...")
            return hypotheses_analysis, past_analyses

        hypothesis = analysis["hypothesis"]                
        analysis_plan = analysis["analysis_plan"]
        current_code = analysis["first_step_code"]
        
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
            notebook.cells.append(nbf.v4.new_markdown_cell(f"## {analysis['code_description']}"))
        
        # Add the first analysis code cell
        current_code = strip_code_markers(current_code)
        notebook.cells.append(new_code_cell(current_code))

        for iteration in range(self.max_iterations):
            step_name = namer(analysis_idx, iteration + 1)
            # Execute the notebook
            #success, error_msg, notebook = self.execute_notebook(notebook)
            success, error_msg, notebook = self.run_last_cell(notebook)
            print(f"🚀 Beginning step {iteration + 1}...")
    

            if success:
                self.logger.log_response(f"STEP {iteration + 1} RAN SUCCESSFULLY - Analysis {analysis_idx+1}", f"step_execution_success_{step_name}")
                results_interpretation = self.interpret_results(notebook, past_analyses, hypothesis, analysis_plan, current_code)
                # Log the interpretation
                self.logger.log_response(results_interpretation, f"results_interpretation_{step_name}")
                # Add interpretation as a markdown cell
                interpretation_cell = nbf.v4.new_markdown_cell(f"### Agent Interpretation\n\n{results_interpretation}")
                notebook.cells.append(interpretation_cell)

            else:
                print(f"⚠️ Code errored with: {error_msg}")
                self.logger.log_response(f"STEP {iteration + 1} FAILED - Analysis {analysis_idx+1}\n\nCode:\n```python\n{current_code}\n\n Error:\n{error_msg}```", f"step_execution_failed_{step_name}")
                fix_attempt, fix_successful = 0, False
                results_interpretation = ""  # Initialize at start of error block
                while fix_attempt < self.max_fix_attempts and not fix_successful:
                    fix_attempt += 1
                    print(f"  🔧 Fix attempt {fix_attempt}/{self.max_fix_attempts}")
                    
                    # Log fix attempt start
                    #self.logger.log_response(f"FIX ATTEMPT {fix_attempt}/{max_fix_attempts} - Analysis {analysis_idx+1}, Step {iteration + 1}", "fix_attempt_start")

                    # Get relevant documentation on the single-cell packages being used
                    documentation = ""
                    if self.use_documentation:
                        try:
                            documentation = get_documentation(current_code)
                        except Exception as e:
                            print(f"⚠️ Documentation extraction failed: {e}")
                            documentation = ""
                    
                    current_code = self.fix_code(current_code, error_msg, documentation=documentation)
                    current_code = strip_code_markers(current_code)
                    notebook.cells[-1] = nbf.v4.new_code_cell(current_code)

                    success, error_msg, notebook = self.run_last_cell(notebook)

                    if success:
                        fix_successful = True
                        print(f"  ✅ Fix successful on attempt {fix_attempt}")
                        
                        # Log successful fix
                        self.logger.log_response(f"FIX SUCCESSFUL on attempt {fix_attempt}/{self.max_fix_attempts} - Analysis {analysis_idx+1}, Step {iteration + 2}", f"fix_attempt_success_{step_name}_{fix_attempt}")
                        
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
                        
                        results_interpretation = self.interpret_results(notebook, past_analyses, hypothesis, analysis_plan, current_code)
                        # Log the interpretation
                        self.logger.log_response(results_interpretation, f"results_interpretation_{step_name}")
                        # Add interpretation as a markdown cell
                        interpretation_cell = nbf.v4.new_markdown_cell(f"### Agent Interpretation\n\n{results_interpretation}")
                        notebook.cells.append(interpretation_cell)
                        break
                    else:
                        print(f"  ❌ Fix attempt {fix_attempt} failed")
                        
                        # Log failed fix attempt with error details
                        self.logger.log_response(f"FIX ATTEMPT FAILED {fix_attempt}/{self.max_fix_attempts} - Analysis {analysis_idx+1}, Step {iteration + 1}: {error_msg}\n\nCode:\n```python\n{current_code}\n```", f"fix_attempt_failed_{step_name}_{fix_attempt}")

                        if fix_attempt == self.max_fix_attempts:
                            print(f"  ⚠️ Failed to fix after {self.max_fix_attempts} attempts. Moving to next iteration.")
                            self.logger.log_response(f"ALL FIX ATTEMPTS EXHAUSTED - Analysis {analysis_idx+1}, Step {iteration + 1}. Failed after {self.max_fix_attempts} attempts.", f"fix_attempt_exhausted_{step_name}")
                            
                            # Log final failure
                            #self.logger.log_error(f"ALL FIX ATTEMPTS EXHAUSTED - Analysis {analysis_idx+1}, Step {iteration + 1}. Failed after {max_fix_attempts} attempts.", current_code)
                            
                            results_interpretation = "Current analysis step failed to run. Try an alternative approach"
                            interpretation_cell = nbf.v4.new_markdown_cell(f"### Agent Interpretation\n\n{results_interpretation}")
                            notebook.cells.append(interpretation_cell)
                if not results_interpretation:  # Only get interpretation if we haven't set the failure message
                    results_interpretation = self.interpret_results(notebook, past_analyses, hypothesis, analysis_plan, current_code)
                    interpretation_cell = nbf.v4.new_markdown_cell(f"### Agent Interpretation\n\n{results_interpretation}")
                    notebook.cells.append(interpretation_cell)

            hypotheses_analysis.append(hypothesis)

            # Only generate next step if this is not the final iteration
            if iteration < self.max_iterations - 1:
                analysis = {"hypothesis": hypothesis, "analysis_plan": analysis_plan, "first_step_code": current_code}
                num_steps_left = self.max_iterations - iteration - 1
                
                try:
                    next_step_analysis = self.generate_next_step_analysis(analysis, past_analyses, notebook.cells, results_interpretation, num_steps_left, seeded = seeded)
                except ValueError as e:
                    if "OpenAI API refused" in str(e) or "OpenAI API returned None" in str(e):
                        print(f"🚫 API refusal/error for next step. Skipping remaining iterations for this analysis.")
                        self.logger.log_response(f"API REFUSAL/ERROR - Ending Analysis {analysis_idx+1} early at step {iteration + 1}: {str(e)}", f"api_refusal_{step_name}")
                        # Add a note to the notebook about the early termination
                        termination_note = f"### Analysis Terminated Early\n\nAPI refusal/error occurred when generating next step: {str(e)}\n\nCompleted {iteration + 1} of {self.max_iterations} planned iterations."
                        notebook.cells.append(nbf.v4.new_markdown_cell(termination_note))
                        break  # Exit the iteration loop early
                    else:
                        # Re-raise other ValueErrors
                        raise

                self.logger.log_response(f"NEXT STEP PLAN - Analysis {analysis_idx+1}, Step {iteration + 2}: {next_step_analysis['analysis_plan'][0]}\n\nCode:\n```python\n{next_step_analysis['first_step_code']}\n```", f"initial_analysis_{step_name}")

                
                if self.use_self_critique:
                    try:
                        modified_analysis = self.get_feedback(next_step_analysis, past_analyses, notebook.cells)
                        self.logger.log_response(f"APPLIED SELF-CRITIQUE - Analysis {analysis_idx+1}, Step {iteration + 2}", f"self_critique_{step_name}")
                    except ValueError as e:
                        if "OpenAI API refused" in str(e) or "OpenAI API returned None" in str(e):
                            print(f"🚫 API refusal/error during self-critique. Using analysis without critique.")
                            self.logger.log_response(f"API REFUSAL DURING CRITIQUE - Using original analysis: {str(e)}", f"critique_refusal_{step_name}")
                            modified_analysis = next_step_analysis
                        else:
                            # Re-raise other ValueErrors
                            raise

                    hypothesis = modified_analysis["hypothesis"]                
                    analysis_plan = modified_analysis["analysis_plan"]
                    current_code = modified_analysis["first_step_code"]
                    # Log revised analysis plan
                    self.logger.log_response(f"Revised Hypothesis: {hypothesis}\n\nRevised Analysis Plan:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)]) + f"\n\nRevised Code:\n{current_code}", f"revised_analysis_{step_name}")
                else:
                    print("🚫 Skipping feedback on next step (no self-critique)")
                    self.logger.log_response(f"SKIPPING INITIAL SELF-CRITIQUE - Analysis {analysis_idx+1}", f"no_self_critique_{step_name}")
                    modified_analysis = next_step_analysis

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
        
        # Clean up notebook (memory leakage)
        del notebook
        import gc
        gc.collect()
        
        print(f"✅ Completed Analysis {analysis_idx+1}")

        return past_analyses + "\n".join(hypotheses_analysis)

    def run(self, seeded_hypotheses=None):
        """
        Main run method that orchestrates both idea generation and execution phases.
        
        Args:
            seeded_hypotheses: Optional list of hypothesis strings for AI to develop into full analyses.
        """
        past_analyses = ""

        for analysis_idx in range(self.num_analyses):
            # Phase 1: Idea Generation
            seeded_hypothesis, seeded = None, False
            
            if seeded_hypotheses and analysis_idx < len(seeded_hypotheses):
                seeded_hypothesis = seeded_hypotheses[analysis_idx]
                seeded = True
            
            try:
                analysis = self.generate_idea(past_analyses, analysis_idx, seeded_hypothesis)
                print(f"🚀 Generated Initial Analysis Plan for Analysis {analysis_idx+1}")
                
                # Phase 2: Idea Execution  
                past_analyses = self.execute_idea(analysis, past_analyses, analysis_idx, seeded = seeded)
                
            except ValueError as e:
                if "OpenAI API refused" in str(e) or "OpenAI API returned None" in str(e):
                    print(f"🚫 API refusal/error for Analysis {analysis_idx+1}. Skipping to next analysis.")
                    print(f"   Error: {str(e)}")
                    # Add this analysis as a skipped entry to past_analyses
                    past_analyses += f"Analysis {analysis_idx+1}: Skipped due to API refusal/error.\n\n"
                    continue  # Skip to next analysis
                else:
                    # Re-raise other ValueErrors
                    raise
        # Clean up resources
        self.cleanup()
        import gc
        gc.collect()

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


def strip_code_markers(text):
    # Remove ```python, ``` and ```
    return re.sub(r'```python|```', '', text)
