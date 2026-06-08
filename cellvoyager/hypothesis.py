"""
Hypothesis generation module.
Extracted from agent.py - Phase 1: Idea Generation.
"""

import json
import os
import traceback
import uuid

import instructor
import litellm
from pydantic import BaseModel

from cellvoyager.utils import get_documentation

litellm.drop_params = True  # ignore unsupported params per-model silently

# Instructor client wrapping LiteLLM — handles retries, validation, and structured output
# for all OpenAI and Anthropic models uniformly.
_instructor_client = instructor.from_litellm(litellm.completion)

_ollama_instructor_client = instructor.from_litellm(
    litellm.completion,
    mode=instructor.Mode.JSON,
)


_MODEL_ALIASES = {
    "gpt-5.3": "openai/gpt-5.3-chat-latest",
    "gpt-5.2": "openai/gpt-5.2-chat-latest",
    "kimi-k2": "moonshot/kimi-k2",
    "kimi-k2.5": "moonshot/kimi-k2.5",
    "kimi-latest": "moonshot/kimi-latest",
}


def _normalize_model_name(model: str) -> str:
    """Add provider prefix for litellm if not already present."""
    if model in _MODEL_ALIASES:
        return _MODEL_ALIASES[model]
    if "/" in model:
        return model  # already has provider prefix
    if model.startswith("claude-") or model.startswith("anthropic"):
        return model  # litellm auto-detects Anthropic models
    # Moonshot/Kimi models — LiteLLM uses moonshot/ prefix, env: MOONSHOT_API_KEY
    if model.startswith(("kimi-", "moonshot-v1")):
        return f"moonshot/{model}"
    # Known auto-detected OpenAI models
    _auto_detected = {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1",
        "o3-mini",
        "o3",
        "o4-mini",
    }
    if model in _auto_detected:
        return model
    # For newer OpenAI models add the prefix
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return f"openai/{model}"
    return model


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
        model_name,  # kept for backward compat, unused
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
        api_base_url=None,
        client=None,  # kept for backward compat, unused
    ):
        # Ensure litellm can route the model — add provider prefix if needed
        self.model_name = _normalize_model_name(model_name)

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
        self.api_base_url = api_base_url

    def _format_messages_for_log(self, messages: list) -> str:
        """Readable formatting for LLM chat messages."""
        return json.dumps(messages, indent=2, ensure_ascii=False, default=str)

    def _complete_structured(
        self, messages: list, phase: str = "structured_completion"
    ) -> dict:
        """Call LiteLLM via instructor and return a validated AnalysisPlan dict."""

        call_id = uuid.uuid4().hex[:10]

        if self.log_prompts:
            self.logger.log_prompt(
                "llm_messages",
                self._format_messages_for_log(messages),
                f"LLM REQUEST [{phase}] call_id={call_id}",
            )

        try:
            is_ollama = self.model_name.startswith(("ollama/", "ollama_chat/"))

            client = _ollama_instructor_client if is_ollama else _instructor_client

            kwargs = {
                "model": self.model_name,
                "messages": list(messages),
                "response_model": AnalysisPlan,
                "max_retries": 2,
            }

            if self.api_base_url:
                kwargs["api_base"] = self.api_base_url

            if is_ollama:
                kwargs["format"] = AnalysisPlan.model_json_schema()
                kwargs["timeout"] = 300.0

            result = client.chat.completions.create(**kwargs)

            output = result.model_dump()

            self.logger.log_response(
                json.dumps(output, indent=2, ensure_ascii=False, default=str),
                f"LLM STRUCTURED RESPONSE [{phase}] call_id={call_id}",
            )

            return output

        except Exception as e:
            self.logger.log_error(
                f"LLM STRUCTURED ERROR [{phase}] call_id={call_id}\n"
                f"{type(e).__name__}: {e}\n\n"
                f"{traceback.format_exc()}"
            )
            raise

    def _complete(self, messages: list, phase: str = "text_completion") -> str:
        """Call LiteLLM for plain-text responses, e.g. critique feedback."""
        call_id = uuid.uuid4().hex[:10]

        if self.log_prompts:
            self.logger.log_prompt(
                "llm_messages",
                self._format_messages_for_log(messages),
                f"LLM REQUEST [{phase}] call_id={call_id}",
            )

        try:
            if self.api_base_url:
                response = litellm.completion(
                    model=self.model_name,
                    messages=list(messages),
                    api_base=self.api_base_url,
                )

            else:
                response = litellm.completion(
                    model=self.model_name,
                    messages=list(messages),
                )

            content = response.choices[0].message.content

            usage = getattr(response, "usage", None)
            usage_text = ""
            if usage is not None:
                usage_text = f"\n\nUSAGE:\n{json.dumps(usage, indent=2, default=str)}"

            self.logger.log_response(
                f"{content}{usage_text}",
                f"LLM TEXT RESPONSE [{phase}] call_id={call_id}",
            )

            return content

        except Exception as e:
            self.logger.log_error(
                f"LLM TEXT ERROR [{phase}] call_id={call_id}\n"
                f"{type(e).__name__}: {e}\n\n"
                f"{traceback.format_exc()}"
            )
            raise

    def generate_jupyter_summary(self, notebook_cells):
        """Generate a comprehensive summary of notebook cells including source code and outputs (including errors)"""
        if notebook_cells is None:
            return ""

        jupyter_summary = ""
        for cell in notebook_cells:
            if (
                cell["cell_type"] == "code"
                or cell["cell_type"] == "markdown"
                or cell["cell_type"] == "error"
            ):
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

        return self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            phase="initial_analysis",
        )

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
            prompt = open(
                os.path.join(
                    self.prompt_dir, "ablations", "critic_NO_DOCUMENTATION.txt"
                )
            ).read()
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

        return self._complete(
            [
                {
                    "role": "system",
                    "content": "You are a single-cell bioinformatics expert providing feedback on code and analysis plan.",
                },
                {"role": "user", "content": prompt},
            ],
            phase="critique_step",
        )

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

        return self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

    def get_feedback(
        self, analysis, past_analyses, notebook_cells, num_steps_left, iterations=1
    ):
        current_analysis = analysis
        for i in range(iterations):
            self.logger.log_response(
                f"Starting self-critique itteration {i + 1}/{iterations}",
                f"self_critique_iteration_{i + 1}",
            )

            feedback = self.critique_step(
                current_analysis, past_analyses, notebook_cells, num_steps_left
            )

            self.logger.log_response(
                feedback,
                f"self_critique_feedback_iteration_{i + 1}",
            )

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
            return self.generate_analysis_from_hypothesis(
                seeded_hypothesis, past_analyses, analysis_idx
            )

        print("🧠 Generating new analysis idea...")

        # Create the initial analysis plan
        analysis = self.generate_initial_analysis(past_analyses)

        if analysis_idx is not None:
            step_name = f"{analysis_idx + 1}_1"
            hypothesis = analysis["hypothesis"]
            analysis_plan = analysis["analysis_plan"]
            initial_code = analysis["first_step_code"]

            # Log only the output of the analysis
            self.logger.log_response(
                f"Hypothesis: {hypothesis}\n\nAnalysis Plan:\n"
                + "\n".join(
                    [f"{i + 1}. {step}" for i, step in enumerate(analysis_plan)]
                )
                + f"\n\nInitial Code:\n{initial_code}",
                f"initial_analysis_{step_name}",
            )

        # Get feedback for the initial analysis plan and modify it accordingly
        if self.use_self_critique:
            modified_analysis = self.get_feedback(
                analysis, past_analyses, None, self.max_iterations
            )

            if analysis_idx is not None:
                self.logger.log_response(
                    f"APPLIED INITIAL SELF-CRITIQUE - Analysis {analysis_idx + 1}",
                    f"self_critique_{step_name}",
                )

                hypothesis = modified_analysis["hypothesis"]
                analysis_plan = modified_analysis["analysis_plan"]
                current_code = modified_analysis["first_step_code"]

                # Log revised analysis plan
                self.logger.log_response(
                    f"Revised Hypothesis: {hypothesis}\n\nRevised Analysis Plan:\n"
                    + "\n".join(
                        [f"{i + 1}. {step}" for i, step in enumerate(analysis_plan)]
                    )
                    + f"\n\nRevised Code:\n{current_code}",
                    f"revised_analysis_{step_name}",
                )

            return modified_analysis
        else:
            if analysis_idx is not None:
                print("🚫 Skipping feedback on next step (no self-critique)")
                self.logger.log_response(
                    f"SKIPPING INITIAL SELF-CRITIQUE - Analysis {analysis_idx + 1}",
                    f"no_self_critique_{step_name}",
                )

            return analysis

    def generate_analysis_from_hypothesis(
        self, hypothesis, past_analyses, analysis_idx=None
    ):
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
        prompt = open(
            os.path.join(self.prompt_dir, "ablations", "analysis_from_hypothesis.txt")
        ).read()
        prompt = prompt.format(
            hypothesis=hypothesis,
            coding_guidelines=self.coding_guidelines,
            adata_summary=self.adata_summary,
            paper_summary=self.paper_summary,
        )

        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "Seeded Hypothesis Analysis")

        analysis = self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

        analysis = self.get_feedback(analysis, past_analyses, None, self.max_iterations)

        # Ensure the hypothesis matches what was provided
        analysis["hypothesis"] = hypothesis

        # Log the seeded hypothesis analysis
        if analysis_idx is not None:
            step_name = f"{analysis_idx + 1}_1"
            analysis_plan = analysis["analysis_plan"]
            initial_code = analysis["first_step_code"]

            # Log the seeded hypothesis analysis
            self.logger.log_response(
                f"Seeded Hypothesis: {hypothesis}\n\nGenerated Analysis Plan:\n"
                + "\n".join(
                    [f"{i + 1}. {step}" for i, step in enumerate(analysis_plan)]
                )
                + f"\n\nInitial Code:\n{initial_code}",
                f"seeded_hypothesis_{step_name}",
            )

        return analysis
