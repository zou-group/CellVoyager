"""
Hypothesis generation module.
Extracted from agent.py - Phase 1: Idea Generation.
"""
import os
import instructor
import litellm
from pydantic import BaseModel
from utils import get_documentation

litellm.drop_params = True  # ignore unsupported params per-model silently

# Instructor client wrapping LiteLLM — handles retries, validation, and structured output
# for all OpenAI and Anthropic models uniformly.
_instructor_client = instructor.from_litellm(litellm.completion)


class AnalysisPlan(BaseModel):
    hypothesis: str
    analysis_plan: list[str]
    first_step_code: str
    code_description: str = ""
    summary: str = ""


class HypothesisGenerator:
    """
    Generates and refines analysis hypotheses/ideas.
    Called during the idea generation phase before execution.
    """

    def __init__(
        self,
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
        client=None,  # kept for backward compat, unused
    ):
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

    def _complete_structured(self, messages: list) -> dict:
        """Call LiteLLM via instructor and return a validated AnalysisPlan dict."""
        result = _instructor_client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            response_model=AnalysisPlan,
        )
        return result.model_dump()

    def _complete(self, messages: list) -> str:
        """Call LiteLLM for plain-text responses (e.g. critique feedback)."""
        response = litellm.completion(model=self.model_name, messages=list(messages))
        return response.choices[0].message.content

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

        return self._complete_structured([
            {"role": "system", "content": self.coding_system_prompt},
            {"role": "user", "content": prompt},
        ])

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

        return self._complete([
            {
                "role": "system",
                "content": "You are a single-cell bioinformatics expert providing feedback on code and analysis plan.",
            },
            {"role": "user", "content": prompt},
        ])

    def incorporate_critique(self, analysis, feedback, notebook_cells, num_steps_left):
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

        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "Incorporate Critiques")

        return self._complete_structured([
            {"role": "system", "content": self.coding_system_prompt},
            {"role": "user", "content": prompt},
        ])

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

        analysis = self._complete_structured([
            {"role": "system", "content": self.coding_system_prompt},
            {"role": "user", "content": prompt},
        ])

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
