"""
Idea execution module.
Extracted from agent.py - Phase 2: Idea Execution.
"""
import os
import re
import json
import base64
import datetime
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_output
from jupyter_client import KernelManager
from cellvoyager.utils import get_documentation

AVAILABLE_PACKAGES = "scanpy, scvi, anndata, matplotlib, numpy, seaborn, pandas, scipy"


def strip_code_markers(text):
    """Remove ```python, ``` and ``` from code blocks."""
    return re.sub(r"```python|```", "", text)


class IdeaExecutor:
    """
    Executes analysis ideas (hypotheses) as Jupyter notebooks.
    Called after the hypothesis generation phase.
    """

    def __init__(
        self,
        hypothesis_generator,
        client,
        model_name,
        prompt_dir,
        coding_guidelines,
        coding_system_prompt,
        adata_summary,
        paper_summary,
        logger,
        h5ad_path,
        output_dir,
        analysis_name,
        max_iterations=6,
        max_fix_attempts=3,
        use_self_critique=True,
        use_VLM=True,
        use_documentation=True,
    ):
        self.hypothesis_generator = hypothesis_generator
        self.client = client
        self.model_name = model_name
        self.prompt_dir = prompt_dir
        self.coding_guidelines = coding_guidelines
        self.coding_system_prompt = coding_system_prompt
        self.adata_summary = adata_summary
        self.paper_summary = paper_summary
        self.logger = logger
        self.h5ad_path = h5ad_path
        self.output_dir = output_dir
        self.analysis_name = analysis_name
        self.max_iterations = max_iterations
        self.max_fix_attempts = max_fix_attempts
        self.use_self_critique = use_self_critique
        self.use_VLM = use_VLM
        self.use_documentation = use_documentation

        # Code memory for context
        self.code_memory = []
        self.code_memory_size = 5
        self.kernel_manager = None
        self.kernel_client = None

    def update_code_memory(self, notebook_cells):
        """Update the code memory with the latest code cells from the notebook"""
        code_cells = []
        for cell in notebook_cells:
            if cell.get("cell_type") == "code":
                code_cells.append(cell["source"] if isinstance(cell, dict) else cell.source)

        self.code_memory = code_cells[-self.code_memory_size :] if len(code_cells) > 0 else []

    def generate_jupyter_summary(self, notebook_cells):
        """Generate a comprehensive summary of notebook cells including source code and outputs (including errors)"""
        if notebook_cells is None:
            return ""

        jupyter_summary = ""
        for cell in notebook_cells:
            cell_type = cell.get("cell_type") if isinstance(cell, dict) else getattr(cell, "cell_type", None)
            source = cell.get("source", "") if isinstance(cell, dict) else getattr(cell, "source", "")
            if cell_type in ("code", "markdown", "error"):
                jupyter_summary += f"{source}\n"

        return jupyter_summary

    def generate_next_step_analysis(self, analysis, attempted_analyses, notebook_cells, num_steps_left, seeded):
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]

        # Update code memory with latest notebook cells
        self.update_code_memory(notebook_cells)

        # Generate comprehensive jupyter summary including outputs and errors
        jupyter_summary = self.generate_jupyter_summary(notebook_cells)

        if seeded:
            prompt = open(os.path.join(self.prompt_dir, "next_step_seeded.txt")).read()
            prompt = prompt.format(
                hypothesis=hypothesis,
                analysis_plan=analysis_plan,
                num_steps_left=num_steps_left,
                CODING_GUIDELINES=self.coding_guidelines,
                jupyter_notebook=jupyter_summary,
                adata_summary=self.adata_summary,
                past_analyses=attempted_analyses,
                paper_txt=self.paper_summary,
            )
        else:
            prompt = open(os.path.join(self.prompt_dir, "next_step.txt")).read()
            prompt = prompt.format(
                hypothesis=hypothesis,
                analysis_plan=analysis_plan,
                CODING_GUIDELINES=self.coding_guidelines,
                jupyter_notebook=jupyter_summary,
                adata_summary=self.adata_summary,
                past_analyses=attempted_analyses,
                paper_txt=self.paper_summary,
                num_steps_left=num_steps_left,
            )

        # For MiniMax and other models that don't support response_format,
        # add explicit JSON instructions to the prompt
        is_minimax = any(x in self.model_name.lower() for x in ["minimax", "minmax"])
        if is_minimax:
            prompt += "\n\nIMPORTANT: Your response MUST be a valid JSON object with the following structure:\n"
            prompt += '{\n  "analysis_plan": [\n    "step 1 description",\n    "step 2 description",\n    ...\n  ],\n  "first_step_code": "python code string here"\n}\n'
            prompt += "Do NOT include any text before or after the JSON object. Only return the JSON."

        # Retry logic for generating valid analysis plan
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                # Check if model supports response_format (MiniMax doesn't support it)
                supports_response_format = not is_minimax
                
                create_kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.coding_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1 if is_minimax else None,  # Lower temperature for MiniMax for more deterministic output
                }
                
                # Only add response_format for models that support it
                if supports_response_format:
                    create_kwargs["response_format"] = {"type": "json_object"}
                
                response = self.client.chat.completions.create(**create_kwargs)
                result = response.choices[0].message.content

                if result is None:
                    print(f"⚠️ API returned None response in generate_next_step (attempt {attempt + 1})")
                    if attempt == max_retries:
                        raise ValueError("OpenAI API returned None response for next step after all retries")
                    continue

                # Clean the result for MiniMax and other models that might add extra text
                if is_minimax:
                    result = result.strip()
                    # Remove <think> tags and their content (thinking process from reasoning models)
                    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL | re.IGNORECASE)
                    result = result.strip()
                    # Try to extract JSON from the response if it contains extra text
                    json_match = re.search(r'\{.*\}', result, re.DOTALL)
                    if json_match:
                        result = json_match.group(0)
                    # Also try to extract from markdown code blocks
                    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result, re.DOTALL)
                    if md_match:
                        result = md_match.group(1)

                try:
                    analysis = json.loads(result)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error in generate_next_step (attempt {attempt + 1}): {e}")
                    print(f"   Raw response: {result[:500]}..." if len(result) > 500 else f"   Raw response: {result}")
                    if attempt == max_retries:
                        raise
                    continue

                if "analysis_plan" not in analysis:
                    if attempt == max_retries:
                        raise ValueError("Generated analysis missing 'analysis_plan' key after all retries")
                    continue

                if "first_step_code" not in analysis:
                    if attempt == max_retries:
                        raise ValueError("Generated analysis missing 'first_step_code' key after all retries")
                    continue

                if not isinstance(analysis["analysis_plan"], list):
                    if attempt == max_retries:
                        raise ValueError("Generated analysis 'analysis_plan' is not a list after all retries")
                    continue

                if len(analysis["analysis_plan"]) == 0:
                    if attempt == max_retries:
                        raise ValueError("Generated analysis has empty 'analysis_plan' after all retries")
                    continue

                if len(analysis["analysis_plan"]) > num_steps_left:
                    if attempt == max_retries:
                        print(f"   Truncating analysis plan to {num_steps_left} steps")
                        analysis["analysis_plan"] = analysis["analysis_plan"][:num_steps_left]

                print(f"✅ Valid analysis plan generated (attempt {attempt + 1}): {len(analysis['analysis_plan'])} steps")
                break

            except Exception as e:
                if attempt == max_retries:
                    print(f"❌ All retry attempts failed for generate_next_step_analysis")
                    raise
                print(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying...")
                continue

        if seeded:
            analysis["hypothesis"] = hypothesis

        return analysis

    def fix_code(self, code, error, other_code="", documentation=""):
        """Attempts to fix code that produced an error"""
        max_error_chars = 2000
        max_other_code_chars = 3000
        max_past_context_chars = 4000
        max_documentation_chars = 3000

        truncated_error = error[-max_error_chars:] if len(error) > max_error_chars else error
        if len(error) > max_error_chars:
            truncated_error = "...(error truncated)...\n" + truncated_error

        truncated_other_code = other_code[-max_other_code_chars:] if len(other_code) > max_other_code_chars else other_code
        if len(other_code) > max_other_code_chars:
            truncated_other_code = "...(context truncated)...\n" + truncated_other_code

        past_code_context = ""
        if self.code_memory:
            past_cells = self.code_memory[-5:]
            past_code_context = "\n\n".join(
                [f"# Previous code cell {i+1}:\n{cell}" for i, cell in enumerate(past_cells)]
            )
            if len(past_code_context) > max_past_context_chars:
                past_code_context = past_code_context[-max_past_context_chars:]
                past_code_context = "...(context truncated)...\n" + past_code_context

        truncated_documentation = (
            documentation[-max_documentation_chars:] if len(documentation) > max_documentation_chars else documentation
        )
        if len(documentation) > max_documentation_chars:
            truncated_documentation = "...(documentation truncated)...\n" + truncated_documentation

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

        if self.use_documentation and truncated_documentation:
            prompt += f"""

        Finally, here is documentation about some of the functions being called, ensure that the code is using the proper parameters/functions:
        {truncated_documentation}"""

        estimated_tokens = len(prompt) // 4
        if estimated_tokens > 50000:
            print(f"⚠️ Warning: Large fix_code prompt detected ({estimated_tokens} estimated tokens)")

        # Check if model supports response_format (MiniMax doesn't support it)
        is_minimax = any(x in self.model_name.lower() for x in ["minimax", "minmax"])
        supports_response_format = not is_minimax
        
        create_kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a coding assistant helping to fix code."},
                {"role": "user", "content": prompt},
            ],
        }
        
        # Only add response_format for models that support it
        if supports_response_format:
            create_kwargs["response_format"] = {"type": "text"}
        
        response = self.client.chat.completions.create(**create_kwargs)
        fixed_code = response.choices[0].message.content
        
        # Clean the fixed code: remove <think> tags and extract code blocks
        if is_minimax:
            fixed_code = re.sub(r'<think>.*?</think>', '', fixed_code, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract code from markdown blocks if present
        code_block_match = re.search(r'```(?:python)?\n(.*?)\n```', fixed_code, re.DOTALL)
        if code_block_match:
            fixed_code = code_block_match.group(1).strip()
        else:
            fixed_code = strip_code_markers(fixed_code).strip()

        return fixed_code

    def generate_code_description(self, code, context=""):
        """Generate a description for a code cell based on its content"""
        prompt = f"""Generate 1-2 sentences describing the goal of the code, what it is doing, and why.

        Code:
        ```python
        {code}
        ```
        """

        # Check if model supports response_format (MiniMax doesn't support it)
        supports_response_format = not any(
            x in self.model_name.lower() for x in ["minimax", "minmax"]
        )
        
        create_kwargs = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a single-cell bioinformatics expert providing concise code descriptions.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        
        # Only add response_format for models that support it
        if supports_response_format:
            create_kwargs["response_format"] = {"type": "text"}
        
        response = self.client.chat.completions.create(**create_kwargs)

        return response.choices[0].message.content.strip()

    def interpret_results(self, notebook, past_analyses, hypothesis, analysis_plan, code):
        last_cell = notebook.cells[-1]
        no_interpretation = "No results found"

        if last_cell.get("cell_type") != "code":
            print("Last cell is not a code cell")
            return no_interpretation

        text_output = ""
        if "outputs" in last_cell:
            for output in last_cell["outputs"]:
                if output.get("output_type") == "stream":
                    text_output += output.get("text", "")
                elif output.get("output_type") == "execute_result":
                    text_output += str(output.get("data", {}).get("text/plain", ""))

        if self.use_VLM:
            image_outputs = []
            if "outputs" in last_cell:
                for i, output in enumerate(last_cell["outputs"]):
                    if output.get("output_type") == "display_data":
                        image_data = output.get("data", {}).get("image/png")
                        if image_data:
                            image_outputs.append({"data": image_data, "format": "image/png"})

            if not text_output and not image_outputs:
                return no_interpretation
        else:
            if not text_output:
                return no_interpretation

        prompt = open(os.path.join(self.prompt_dir, "interp_results.txt")).read()
        prompt = prompt.format(
            text_output=text_output,
            paper_txt=self.paper_summary,
            CODING_GUIDELINES=self.coding_guidelines,
            past_analyses=past_analyses,
            hypothesis=hypothesis,
            analysis_plan=analysis_plan,
            code=code,
        )

        if self.use_VLM:
            user_content = [{"type": "text", "text": prompt}]
            try:
                for img in image_outputs:
                    try:
                        image_data = img["data"]
                        if isinstance(image_data, str) and "," in image_data:
                            image_data = image_data.split(",")[1]
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"},
                            }
                        )
                    except Exception as e:
                        print(f"Warning: Error processing image: {str(e)}")
                        continue

                # Check if model supports response_format (MiniMax doesn't support it)
                supports_response_format = not any(
                    x in self.model_name.lower() for x in ["minimax", "minmax"]
                )
                
                create_kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a single-cell transcriptomics expert providing feedback on Python code and analysis plan.",
                        },
                        {"role": "user", "content": user_content},
                    ],
                }
                
                # Only add response_format for models that support it
                if supports_response_format:
                    create_kwargs["response_format"] = {"type": "text"}
                
                response = self.client.chat.completions.create(**create_kwargs)
                feedback = response.choices[0].message.content
            finally:
                image_outputs.clear()
                user_content.clear()
                import gc
                gc.collect()
        else:
            # Check if model supports response_format (MiniMax doesn't support it)
            supports_response_format = not any(
                x in self.model_name.lower() for x in ["minimax", "minmax"]
            )
            
            create_kwargs = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a single-cell bioinformatics expert providing feedback on Python code and analysis plan.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            
            # Only add response_format for models that support it
            if supports_response_format:
                create_kwargs["response_format"] = {"type": "text"}
            
            response = self.client.chat.completions.create(**create_kwargs)
            feedback = response.choices[0].message.content

        return feedback

    def start_persistent_kernel(self):
        """Start a persistent kernel for efficient cell execution"""
        try:
            self.kernel_manager = KernelManager(kernel_name="python3")
            self.kernel_manager.start_kernel()
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
            self.kernel_client = None
            self.kernel_manager = None

    def run_last_cell(self, nb):
        """Executes the most recently added code cell and updates its outputs."""
        if not nb.cells:
            raise ValueError("No cells in notebook to run.")

        last_code_cell = None
        for cell in reversed(nb.cells):
            if cell.cell_type == "code":
                last_code_cell = cell
                break

        if not last_code_cell:
            raise ValueError("No code cells found in notebook.")

        code = last_code_cell.source
        msg_id = self.kernel_client.execute(code)
        outputs = []

        while True:
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=300)
            except Exception:
                break

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "status" and content.get("execution_state") == "idle":
                break

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            if msg_type == "stream":
                outputs.append(
                    new_output(output_type="stream", name=content["name"], text=content["text"])
                )
            elif msg_type == "execute_result":
                outputs.append(
                    new_output(
                        output_type="execute_result",
                        data=content["data"],
                        execution_count=content["execution_count"],
                    )
                )
            elif msg_type == "display_data":
                outputs.append(
                    new_output(
                        output_type="display_data",
                        data=content["data"],
                        metadata=content.get("metadata", {}),
                    )
                )
            elif msg_type == "error":
                outputs.append(
                    new_output(
                        output_type="error",
                        ename=content["ename"],
                        evalue=content["evalue"],
                        traceback=content["traceback"],
                    )
                )

        code_cell_index = nb.cells.index(last_code_cell)
        nb.cells[code_cell_index].outputs = outputs

        for output in outputs:
            if output.output_type == "error":
                error_msg = f"{output.ename}: {output.evalue}"
                return False, error_msg, nb

        return True, None, nb

    def create_initial_notebook(self, hypothesis):
        notebook = nbf.v4.new_notebook()
        notebook.cells.append(nbf.v4.new_markdown_cell(f"# Analysis\n\n**Hypothesis**: {hypothesis}"))

        setup_code = f"""import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Set up visualization defaults for better plots
sc.settings.verbosity = 3
sc.settings.figsize = (8, 8)
sc.settings.dpi = 100
sc.settings.facecolor = 'white'
warnings.filterwarnings('ignore')

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
            if cell.cell_type == "code" and hasattr(cell, "outputs"):
                cleaned_outputs = []
                for output in cell.outputs:
                    if isinstance(output, dict):
                        if output.get("output_type") == "stream":
                            cleaned_outputs.append(
                                nbf.v4.new_output(
                                    "stream",
                                    name=output.get("name", "stdout"),
                                    text=output.get("text", ""),
                                )
                            )
                        elif output.get("output_type") == "execute_result":
                            cleaned_outputs.append(
                                nbf.v4.new_output(
                                    "execute_result",
                                    data=output.get("data", {}),
                                    execution_count=output.get("execution_count", None),
                                )
                            )
                        elif output.get("output_type") == "display_data":
                            cleaned_outputs.append(
                                nbf.v4.new_output("display_data", data=output.get("data", {}))
                            )
                        elif output.get("output_type") == "error":
                            cleaned_outputs.append(
                                nbf.v4.new_output(
                                    "error",
                                    ename=output.get("ename", ""),
                                    evalue=output.get("evalue", ""),
                                    traceback=output.get("traceback", []),
                                )
                            )
                    else:
                        cleaned_outputs.append(output)
                cell.outputs = cleaned_outputs

        return notebook

    def execute_idea(self, analysis, past_analyses, analysis_idx, seeded=False):
        """
        Phase 2: Idea Execution

        Args:
            analysis: Analysis dict from generate_idea phase
            past_analyses: String of past analysis summaries
            analysis_idx: Analysis index for logging
            seeded: Boolean indicating if the analysis is seeded

        Returns:
            updated past_analyses string
        """

        def namer(analysis_idx, step_idx):
            return f"{analysis_idx+1}_{step_idx}"

        hypotheses_analysis = []
        self.code_memory = []

        print(f"\n🚀 Executing Analysis {analysis_idx+1}")

        if not self.start_persistent_kernel():
            print(f"⚠️ Failed to start kernel for analysis {analysis_idx+1}. Skipping...")
            return past_analyses

        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        current_code = analysis["first_step_code"]

        plan_markdown = "# Analysis Plan\n\n**Hypothesis**: " + hypothesis + "\n\n## Steps:\n"
        for step in analysis_plan:
            plan_markdown += f"- {step}\n"

        notebook = self.create_initial_notebook(hypothesis)
        _, _, notebook = self.run_last_cell(notebook)

        notebook.cells.append(nbf.v4.new_markdown_cell(plan_markdown))

        if analysis_plan:
            notebook.cells.append(nbf.v4.new_markdown_cell(f"## {analysis['code_description']}"))

        current_code = strip_code_markers(current_code)
        notebook.cells.append(new_code_cell(current_code))

        for iteration in range(self.max_iterations):
            step_name = namer(analysis_idx, iteration + 1)
            success, error_msg, notebook = self.run_last_cell(notebook)
            print(f"🚀 Beginning step {iteration + 1}...")

            if success:
                self.logger.log_response(
                    f"STEP {iteration + 1} RAN SUCCESSFULLY - Analysis {analysis_idx+1}",
                    f"step_execution_success_{step_name}",
                )
                results_interpretation = self.interpret_results(
                    notebook, past_analyses, hypothesis, analysis_plan, current_code
                )
                self.logger.log_response(results_interpretation, f"results_interpretation_{step_name}")
                interpretation_cell = nbf.v4.new_markdown_cell(
                    f"### Agent Interpretation\n\n{results_interpretation}"
                )
                notebook.cells.append(interpretation_cell)

            else:
                print(f"⚠️ Code errored with: {error_msg}")
                self.logger.log_response(
                    f"STEP {iteration + 1} FAILED - Analysis {analysis_idx+1}\n\nCode:\n```python\n{current_code}\n\n Error:\n{error_msg}```",
                    f"step_execution_failed_{step_name}",
                )
                fix_attempt, fix_successful = 0, False
                results_interpretation = ""
                while fix_attempt < self.max_fix_attempts and not fix_successful:
                    fix_attempt += 1
                    print(f"  🔧 Fix attempt {fix_attempt}/{self.max_fix_attempts}")

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
                        self.logger.log_response(
                            f"FIX SUCCESSFUL on attempt {fix_attempt}/{self.max_fix_attempts} - Analysis {analysis_idx+1}, Step {iteration + 2}",
                            f"fix_attempt_success_{step_name}_{fix_attempt}",
                        )
                        updated_description = self.generate_code_description(current_code)
                        for i in range(len(notebook.cells) - 1, -1, -1):
                            cell = notebook.cells[i]
                            if (
                                cell.cell_type == "markdown"
                                and str(cell.source).startswith("##")
                                and "Agent Interpretation" not in str(cell.source)
                            ):
                                cell.source = f"## {updated_description}"
                                break
                        results_interpretation = self.interpret_results(
                            notebook, past_analyses, hypothesis, analysis_plan, current_code
                        )
                        self.logger.log_response(
                            results_interpretation, f"results_interpretation_{step_name}"
                        )
                        interpretation_cell = nbf.v4.new_markdown_cell(
                            f"### Agent Interpretation\n\n{results_interpretation}"
                        )
                        notebook.cells.append(interpretation_cell)
                        break
                    else:
                        print(f"  ❌ Fix attempt {fix_attempt} failed")
                        self.logger.log_response(
                            f"FIX ATTEMPT FAILED {fix_attempt}/{self.max_fix_attempts} - Analysis {analysis_idx+1}, Step {iteration + 1}: {error_msg}\n\nCode:\n```python\n{current_code}\n```",
                            f"fix_attempt_failed_{step_name}_{fix_attempt}",
                        )
                        if fix_attempt == self.max_fix_attempts:
                            print(
                                f"  ⚠️ Failed to fix after {self.max_fix_attempts} attempts. Moving to next iteration."
                            )
                            self.logger.log_response(
                                f"ALL FIX ATTEMPTS EXHAUSTED - Analysis {analysis_idx+1}, Step {iteration + 1}. Failed after {self.max_fix_attempts} attempts.",
                                f"fix_attempt_exhausted_{step_name}",
                            )
                            results_interpretation = (
                                "Current analysis step failed to run. Try an alternative approach"
                            )
                            interpretation_cell = nbf.v4.new_markdown_cell(
                                f"### Agent Interpretation\n\n{results_interpretation}"
                            )
                            notebook.cells.append(interpretation_cell)
                if not results_interpretation:
                    results_interpretation = self.interpret_results(
                        notebook, past_analyses, hypothesis, analysis_plan, current_code
                    )
                    interpretation_cell = nbf.v4.new_markdown_cell(
                        f"### Agent Interpretation\n\n{results_interpretation}"
                    )
                    notebook.cells.append(interpretation_cell)

            hypotheses_analysis.append(hypothesis)

            if iteration < self.max_iterations - 1:
                num_steps_left = self.max_iterations - iteration - 1

                analysis = {
                    "hypothesis": hypothesis,
                    "analysis_plan": analysis_plan,
                    "first_step_code": current_code,
                }
                next_step_analysis = self.generate_next_step_analysis(
                    analysis, past_analyses, notebook.cells, num_steps_left, seeded
                )

                first_step_description = (
                    next_step_analysis["analysis_plan"][0]
                    if next_step_analysis["analysis_plan"]
                    else "No additional analysis steps generated"
                )
                self.logger.log_response(
                    f"NEXT STEP PLAN - Analysis {analysis_idx+1}, Step {iteration + 2}: {first_step_description}\n\nCode:\n```python\n{next_step_analysis['first_step_code']}\n```",
                    f"initial_analysis_{step_name}",
                )

                if self.use_self_critique:
                    modified_analysis = self.hypothesis_generator.get_feedback(
                        next_step_analysis, past_analyses, notebook.cells, num_steps_left
                    )
                    self.logger.log_response(
                        f"APPLIED SELF-CRITIQUE - Analysis {analysis_idx+1}, Step {iteration + 2}",
                        f"self_critique_{step_name}",
                    )
                    hypothesis = modified_analysis["hypothesis"]
                    analysis_plan = modified_analysis["analysis_plan"]
                    current_code = modified_analysis["first_step_code"]
                    self.logger.log_response(
                        f"Revised Hypothesis: {hypothesis}\n\nRevised Analysis Plan:\n"
                        + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)])
                        + f"\n\nRevised Code:\n{current_code}",
                        f"revised_analysis_{step_name}",
                    )
                else:
                    print("🚫 Skipping feedback on next step (no self-critique)")
                    self.logger.log_response(
                        f"SKIPPING INITIAL SELF-CRITIQUE - Analysis {analysis_idx+1}",
                        f"no_self_critique_{step_name}",
                    )
                    modified_analysis = next_step_analysis

                print(
                    f"ANALYSIS PLAN AFTER NEXT STEP GENERATION AND AFTER CRITIQUE (length: {len(modified_analysis['analysis_plan'])}):",
                    modified_analysis["analysis_plan"],
                )

                steps_text = "\n".join(
                    [f"Step {i+1}: {item}" for i, item in enumerate(modified_analysis["analysis_plan"])]
                )
                next_step_cell = nbf.v4.new_markdown_cell(f"## Next Steps\n{steps_text}")
                notebook.cells.append(next_step_cell)
                code_description = modified_analysis["code_description"]
                notebook.cells.append(nbf.v4.new_markdown_cell(f"## {code_description}"))
                modified_code = strip_code_markers(modified_analysis["first_step_code"])
                notebook.cells.append(new_code_cell(modified_code))
                current_code = modified_code

            self.update_code_memory(notebook.cells)

        notebook_path = os.path.join(
            self.output_dir, f"{self.analysis_name}_analysis_{analysis_idx+1}.ipynb"
        )
        with open(notebook_path, "w", encoding="utf-8") as f:
            clean_notebook = self.cleanup_notebook_outputs(notebook)
            nbf.write(clean_notebook, f)
            print(f"💾 Saved notebook to: {notebook_path}")

        self.logger.log_response(
            f"ANALYSIS {analysis_idx+1} COMPLETED - Notebook saved to: {notebook_path}",
            "analysis_complete",
        )

        self.stop_persistent_kernel()

        del notebook
        import gc
        gc.collect()

        print(f"✅ Completed Analysis {analysis_idx+1}")

        if hypotheses_analysis:
            analysis_summary = f"Analysis {analysis_idx+1}: {hypothesis}\n"
            return past_analyses + analysis_summary
        else:
            return past_analyses
