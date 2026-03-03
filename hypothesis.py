"""
Hypothesis generation module.
Extracted from agent.py - Phase 1: Idea Generation.
"""
import os
import json
from utils import get_documentation


class HypothesisGenerator:
    """
    Generates and refines analysis hypotheses/ideas.
    Called during the idea generation phase before execution.
    """

    def __init__(
        self,
        client,
        model_name,
        prompt_dir,
        coding_guidelines,
        coding_system_prompt,
        adata_summary,
        paper_summary,
        logger,
        use_self_critique=True,
        use_documentation=True,
        max_iterations=6,
        deepresearch_background="",
        log_prompts=False,
    ):
        self.client = client
        self.model_name = model_name
        self.prompt_dir = prompt_dir
        self.coding_guidelines = coding_guidelines
        self.coding_system_prompt = coding_system_prompt
        self.adata_summary = adata_summary
        self.paper_summary = paper_summary
        self.logger = logger
        self.use_self_critique = use_self_critique
        self.use_documentation = use_documentation
        self.max_iterations = max_iterations
        self.deepresearch_background = deepresearch_background
        self.log_prompts = log_prompts

    def generate_jupyter_summary(self, notebook_cells):
        """Generate a comprehensive summary of notebook cells including source code and outputs (including errors)"""
        if notebook_cells is None:
            return ""

        jupyter_summary = ""
        for cell in notebook_cells:
            if cell["cell_type"] == "code" or cell["cell_type"] == "markdown" or cell["cell_type"] == "error":
                jupyter_summary += f"{cell['source']}\n"

        return jupyter_summary

    def generate_initial_analysis(self, attempted_analyses):
        prompt = open(os.path.join(self.prompt_dir, "first_draft.txt")).read()
        prompt = prompt.format(
            CODING_GUIDELINES=self.coding_guidelines,
            adata_summary=self.adata_summary,
            past_analyses=attempted_analyses,
            paper_txt=self.paper_summary,
            deepresearch_background=self.deepresearch_background,
            max_iterations=self.max_iterations,
        )

        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "Initial Analysis")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content

        # Debug logging for API response issues
        if result is None:
            print(f"⚠️ API returned None response in generate_initial_analysis")
            print(f"   Model: {self.model_name}")
            print(f"   Response object: {response}")
            raise ValueError("OpenAI API returned None response for initial analysis")

        try:
            analysis = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error in generate_initial_analysis: {e}")
            print(f"   Raw result: {repr(result)}")
            raise

        return analysis

    def critique_step(self, analysis, past_analyses, notebook_cells, num_steps_left):
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]

        # Generate comprehensive jupyter summary including outputs and errors
        jupyter_summary = self.generate_jupyter_summary(notebook_cells)

        if self.use_documentation:
            prompt = open(os.path.join(self.prompt_dir, "critic.txt")).read()
            # Get relevant documentation on the single-cell packages being used in the first step code
            try:
                documentation = get_documentation(first_step_code)
            except Exception as e:
                print(f"⚠️ Documentation extraction failed: {e}")
                documentation = ""
            prompt = prompt.format(
                hypothesis=hypothesis,
                analysis_plan=analysis_plan,
                first_step_code=first_step_code,
                CODING_GUIDELINES=self.coding_guidelines,
                adata_summary=self.adata_summary,
                past_analyses=past_analyses,
                paper_txt=self.paper_summary,
                jupyter_notebook=jupyter_summary,
                documentation=documentation,
                num_steps_left=num_steps_left,
            )
        else:
            prompt = open(os.path.join(self.prompt_dir, "ablations", "critic_NO_DOCUMENTATION.txt")).read()
            prompt = prompt.format(
                hypothesis=hypothesis,
                analysis_plan=analysis_plan,
                first_step_code=first_step_code,
                CODING_GUIDELINES=self.coding_guidelines,
                adata_summary=self.adata_summary,
                past_analyses=past_analyses,
                paper_txt=self.paper_summary,
                jupyter_notebook=jupyter_summary,
                num_steps_left=num_steps_left,
            )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a single-cell bioinformatics expert providing feedback on code and analysis plan.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        feedback = response.choices[0].message.content
        return feedback

    def incorporate_critique(self, analysis, feedback, notebook_cells, num_steps_left):
        ## Return analysis object
        hypothesis = analysis["hypothesis"]
        analysis_plan = analysis["analysis_plan"]
        first_step_code = analysis["first_step_code"]

        # Generate comprehensive jupyter summary including outputs and errors
        jupyter_summary = self.generate_jupyter_summary(notebook_cells)

        prompt = open(os.path.join(self.prompt_dir, "incorporate_critque.txt")).read()
        prompt = prompt.format(
            hypothesis=hypothesis,
            analysis_plan=analysis_plan,
            first_step_code=first_step_code,
            CODING_GUIDELINES=self.coding_guidelines,
            adata_summary=self.adata_summary,
            feedback=feedback,
            jupyter_notebook=jupyter_summary,
            num_steps_left=num_steps_left,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content

        # Debug logging for API response issues
        if result is None:
            print(f"⚠️ API returned None response in incorporate_critique")
            print(f"   Model: {self.model_name}")
            print(f"   Response object: {response}")
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

    def get_feedback(self, analysis, past_analyses, notebook_cells, num_steps_left, iterations=1):
        current_analysis = analysis
        for i in range(iterations):
            feedback = self.critique_step(current_analysis, past_analyses, notebook_cells, num_steps_left)
            current_analysis = self.incorporate_critique(
                current_analysis, feedback, notebook_cells, num_steps_left
            )

        return current_analysis

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
            self.logger.log_response(
                f"Hypothesis: {hypothesis}\n\nAnalysis Plan:\n"
                + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)])
                + f"\n\nInitial Code:\n{initial_code}",
                f"initial_analysis_{step_name}",
            )

        # Get feedback for the initial analysis plan and modify it accordingly
        if self.use_self_critique:
            modified_analysis = self.get_feedback(analysis, past_analyses, None, self.max_iterations)

            if analysis_idx is not None:
                self.logger.log_response(
                    f"APPLIED INITIAL SELF-CRITIQUE - Analysis {analysis_idx+1}",
                    f"self_critique_{step_name}",
                )

                hypothesis = modified_analysis["hypothesis"]
                analysis_plan = modified_analysis["analysis_plan"]
                current_code = modified_analysis["first_step_code"]

                # Log revised analysis plan
                self.logger.log_response(
                    f"Revised Hypothesis: {hypothesis}\n\nRevised Analysis Plan:\n"
                    + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)])
                    + f"\n\nRevised Code:\n{current_code}",
                    f"revised_analysis_{step_name}",
                )

            return modified_analysis
        else:
            if analysis_idx is not None:
                print("🚫 Skipping feedback on next step (no self-critique)")
                self.logger.log_response(
                    f"SKIPPING INITIAL SELF-CRITIQUE - Analysis {analysis_idx+1}",
                    f"no_self_critique_{step_name}",
                )

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
        prompt = prompt.format(
            hypothesis=hypothesis,
            coding_guidelines=self.coding_guidelines,
            adata_summary=self.adata_summary,
            paper_summary=self.paper_summary,
        )

        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "Seeded Hypothesis Analysis")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content

        # Debug logging for API response issues
        if result is None:
            print(f"⚠️ API returned None response in generate_analysis_from_hypothesis")
            print(f"   Model: {self.model_name}")
            print(f"   Response object: {response}")
            raise ValueError("OpenAI API returned None response for hypothesis analysis")

        try:
            analysis = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error in generate_analysis_from_hypothesis: {e}")
            print(f"   Raw result: {repr(result)}")
            raise

        analysis = self.get_feedback(analysis, past_analyses, None, self.max_iterations)

        # Ensure the hypothesis matches what was provided
        analysis["hypothesis"] = hypothesis

        # Log the seeded hypothesis analysis
        if analysis_idx is not None:
            step_name = f"{analysis_idx+1}_1"
            analysis_plan = analysis["analysis_plan"]
            initial_code = analysis["first_step_code"]

            # Log the seeded hypothesis analysis
            self.logger.log_response(
                f"Seeded Hypothesis: {hypothesis}\n\nGenerated Analysis Plan:\n"
                + "\n".join([f"{i+1}. {step}" for i, step in enumerate(analysis_plan)])
                + f"\n\nInitial Code:\n{initial_code}",
                f"seeded_hypothesis_{step_name}",
            )

        return analysis
