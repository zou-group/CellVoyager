"""
Agent v2: Modular architecture with separate hypothesis generation and execution.
Uses hypothesis.py (HypothesisGenerator) and either execution_old.py (IdeaExecutor)
or execution.py (ClaudeJupyterExecutor). agent.py remains unchanged.
"""
import os
import datetime
import pandas as pd
import numpy as np
import openai
import h5py
from h5py import Dataset, Group

from hypothesis import HypothesisGenerator
from execution_old import IdeaExecutor
from logger import Logger
from deepresearch import DeepResearcher

AVAILABLE_PACKAGES = "scanpy, scvi, anndata, matplotlib, numpy, seaborn, pandas, scipy"


class AnalysisAgentV2:
    def __init__(
        self,
        h5ad_path,
        paper_summary_path,
        openai_api_key,
        model_name,
        analysis_name,
        num_analyses=5,
        max_iterations=6,
        prompt_dir="prompts",
        output_home=".",
        log_home=".",
        use_self_critique=True,
        use_VLM=True,
        use_documentation=True,
        log_prompts=False,
        max_fix_attempts=3,
        use_deepresearch_background=True,
        execution_mode="legacy",
        anthropic_api_key=None,
        **execution_kwargs,
    ):
        """
        Args:
            execution_mode: "legacy" (default) uses IdeaExecutor from execution_old.py;
                "claude" uses ClaudeJupyterExecutor from execution.py (live Jupyter + Claude Agent SDK).
            anthropic_api_key: Required when execution_mode="claude". Can also set ANTHROPIC_API_KEY env.
            **execution_kwargs: Passed to ClaudeJupyterExecutor when execution_mode="claude",
                e.g. jupyter_port=8888, auto_start_jupyter=True, stop_jupyter_on_complete=False.
        """
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

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_home, "outputs", f"{analysis_name}_{timestamp}")

        self.client = openai.OpenAI(api_key=openai_api_key)

        self.use_self_critique = use_self_critique
        self.use_VLM = use_VLM
        self.use_documentation = use_documentation

        os.makedirs(self.output_dir, exist_ok=True)

        # Coding guidelines (same as agent.py)
        self._analyses_overview = open(os.path.join(self.prompt_dir, "DeepResearch_Analyses.txt")).read()
        if self.use_VLM:
            coding_guidelines_template = open(
                os.path.join(self.prompt_dir, "coding_guidelines.txt")
            ).read()
        else:
            coding_guidelines_template = open(
                os.path.join(self.prompt_dir, "ablations", "coding_guidelines_NO_VLM_ABLATION.txt")
            ).read()

        self.coding_system_prompt = open(
            os.path.join(self.prompt_dir, "coding_system_prompt.txt")
        ).read().format(max_iterations=self.max_iterations)

        self.coding_guidelines = coding_guidelines_template.format(
            name=self.analysis_name,
            adata_path=self.h5ad_path,
            available_packages=AVAILABLE_PACKAGES,
            analyses_overview=self._analyses_overview,
        )

        self.logger = Logger(self.analysis_name, log_dir=os.path.join(log_home, "logs"))

        # Load adata and build summary
        if self.h5ad_path == "":
            self.adata_summary = ""
        else:
            print("Loading anndata .obs for summarization...")
            self.adata_obs = self._load_h5ad_obs(self.h5ad_path)
            self.adata_summary = self._summarize_adata_metadata()
            print("ADATA SUMMARY: ", self.adata_summary)
            print(f"✅ Loaded {self.h5ad_path}")

        # DeepResearch for idea generation
        self.deepresearch_background = ""
        if self.use_deepresearch_background:
            print("Running DeepResearch...")
            try:
                deepresearch = DeepResearcher(self.openai_api_key)
                dr_summary = deepresearch.research_from_paper_summary(
                    self.paper_summary, self.adata_summary, AVAILABLE_PACKAGES
                )
                self.deepresearch_background = dr_summary.strip()
                print("✅ DeepResearch completed")
                print("DEEPRESEARCH BACKGROUND: ", self.deepresearch_background[:100])
            except Exception as e:
                print(f"Warning: DeepResearch failed or was skipped: {e}")

        # (1) Hypothesis generation module
        self.hypothesis_generator = HypothesisGenerator(
            client=self.client,
            model_name=self.model_name,
            prompt_dir=self.prompt_dir,
            coding_guidelines=self.coding_guidelines,
            coding_system_prompt=self.coding_system_prompt,
            adata_summary=self.adata_summary,
            paper_summary=self.paper_summary,
            logger=self.logger,
            use_self_critique=self.use_self_critique,
            use_documentation=self.use_documentation,
            max_iterations=self.max_iterations,
            deepresearch_background=self.deepresearch_background,
            log_prompts=self.log_prompts,
        )

        # (2) Idea execution module
        shared_executor_kwargs = dict(
            hypothesis_generator=self.hypothesis_generator,
            client=self.client,
            model_name=self.model_name,
            prompt_dir=self.prompt_dir,
            coding_guidelines=self.coding_guidelines,
            coding_system_prompt=self.coding_system_prompt,
            adata_summary=self.adata_summary,
            paper_summary=self.paper_summary,
            logger=self.logger,
            h5ad_path=self.h5ad_path,
            output_dir=self.output_dir,
            analysis_name=self.analysis_name,
            max_iterations=self.max_iterations,
            max_fix_attempts=self.max_fix_attempts,
            use_self_critique=self.use_self_critique,
            use_VLM=self.use_VLM,
            use_documentation=self.use_documentation,
        )

        if execution_mode == "claude":
            from execution import ClaudeJupyterExecutor
            self.executor = ClaudeJupyterExecutor(
                **shared_executor_kwargs,
                anthropic_api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"),
                **execution_kwargs,
            )
        else:
            self.executor = IdeaExecutor(**shared_executor_kwargs)

    def _summarize_adata_metadata(self, length_cutoff=25):
        summarization_str = "Below is a description of the columns in adata.obs: \n"
        columns = self.adata_obs.columns
        for col in columns:
            unique_vals = self.adata_obs[col].unique()
            if len(unique_vals) > length_cutoff:
                vals_str = str(unique_vals[:length_cutoff]) + f"and {len(unique_vals) - length_cutoff} other unique values..."
            else:
                vals_str = str(unique_vals)
            summarization_str += f"Column {col} contains the unique values {vals_str} \n"
        return summarization_str

    def _load_h5ad_obs(self, h5ad_path):
        """Load just the .obs data from an h5ad file while preserving data types"""
        with h5py.File(h5ad_path, "r") as f:
            obs_dict = {}

            for raw_k in [k for k in f["obs"].keys() if not k.startswith("_")]:
                k = raw_k.decode("utf-8") if isinstance(raw_k, bytes) else raw_k
                item = f["obs"][raw_k]
                if isinstance(item, Dataset):
                    data = item[:]
                elif isinstance(item, Group) and "codes" in item.keys() and "categories" in item.keys():
                    data = item["codes"][:]
                    categories = item["categories"][:]
                    categories = [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in categories
                    ]
                    data = pd.Categorical.from_codes(
                        data.astype(int) if not np.issubdtype(data.dtype, np.integer) else data,
                        categories=categories,
                    )
                else:
                    raise ValueError(f"Didnt account for this datatype in h5ad: {type(item)}")

                if "categories" in item.attrs:
                    try:
                        cat_ref = item.attrs["categories"]
                        if isinstance(cat_ref, h5py.h5r.Reference):
                            cat_vals = f[cat_ref][:]
                            categories = [
                                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                                for x in cat_vals
                            ]
                        else:
                            cat_vals = cat_ref[:]
                            categories = [
                                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                                for x in cat_vals
                            ]
                        data = pd.Categorical.from_codes(
                            data.astype(int) if not np.issubdtype(data.dtype, np.integer) else data,
                            categories=categories,
                        )
                    except Exception as e:
                        print(f"Warning: Error with categorical {k}: {str(e)}")
                        data = np.array(
                            [
                                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                                for x in data
                            ]
                        )
                elif (
                    data.dtype.kind in ["S", "O"]
                    or h5py.check_string_dtype(f["obs"][raw_k].dtype) is not None
                ):
                    try:
                        data = np.array(
                            [
                                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                                for x in data
                            ]
                        )
                    except Exception as e:
                        print(f"Warning: Error decoding strings in {k}: {str(e)}")

                obs_dict[k] = data

            try:
                if "_index" in f["obs"]:
                    idx = f["obs"]["_index"][:]
                    index = np.array(
                        [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in idx]
                    )
                else:
                    index = None
            except Exception as e:
                print(f"Warning: Error processing index: {str(e)}")
                index = None

        df = pd.DataFrame(obs_dict, index=index)
        print(f"Loaded obs data: {len(df)} rows × {len(df.columns)} columns")
        return df

    def run(self, seeded_hypotheses=None):
        """
        Main run method that orchestrates both idea generation and execution phases.

        Args:
            seeded_hypotheses: Optional list of hypothesis strings for AI to develop into full analyses.
        """
        past_analyses = ""

        for analysis_idx in range(self.num_analyses):
            seeded_hypothesis, seeded = None, False

            if seeded_hypotheses and analysis_idx < len(seeded_hypotheses):
                seeded_hypothesis = seeded_hypotheses[analysis_idx]
                seeded = True

            try:
                # Phase 1: Idea Generation (hypothesis.py)
                analysis = self.hypothesis_generator.generate_idea(
                    past_analyses, analysis_idx, seeded_hypothesis
                )
                print(f"🚀 Generated Initial Analysis Plan for Analysis {analysis_idx+1}")

                # Phase 2: Idea Execution
                past_analyses = self.executor.execute_idea(
                    analysis, past_analyses, analysis_idx, seeded=seeded
                )

            except ValueError as e:
                if "OpenAI API refused" in str(e) or "OpenAI API returned None" in str(e):
                    print(f"🚫 API refusal/error for Analysis {analysis_idx+1}. Skipping to next analysis.")
                    print(f"   Error: {str(e)}")
                    past_analyses += f"Analysis {analysis_idx+1}: Skipped due to API refusal/error.\n\n"
                    continue
                else:
                    raise

        # Clean up resources (IdeaExecutor owns the kernel; ClaudeJupyterExecutor manages Jupyter)
        if hasattr(self.executor, "stop_persistent_kernel"):
            self.executor.stop_persistent_kernel()
        import gc
        gc.collect()
