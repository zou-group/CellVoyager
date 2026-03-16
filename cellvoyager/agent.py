"""
CellVoyager agent: Modular architecture with separate hypothesis generation and execution.
Uses HypothesisGenerator (cellvoyager.hypothesis) and either IdeaExecutor
(cellvoyager.execution.legacy) or ClaudeJupyterExecutor (cellvoyager.execution.claude).
"""
import os
import datetime
import pandas as pd
import numpy as np
import openai
import h5py
from h5py import Dataset, Group

import anndata

from cellvoyager.hypothesis import HypothesisGenerator
from cellvoyager.execution.legacy import IdeaExecutor
from cellvoyager.logger import Logger
from cellvoyager.deepresearch import DeepResearcher

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
        prompt_dir=None,
        output_home=".",
        output_dir=None,
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
            execution_mode: "legacy" (default) uses IdeaExecutor;
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
        self.prompt_dir = prompt_dir or os.path.join(os.path.dirname(__file__), "prompts")
        self.log_prompts = log_prompts
        self.max_fix_attempts = max_fix_attempts
        self.use_deepresearch_background = use_deepresearch_background

        if output_dir is not None:
            self.output_dir = os.path.abspath(output_dir)
        else:
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

        # Load adata metadata and build summary for planning.
        # In Claude mode, avoid a full anndata load here because the notebook setup cell
        # loads adata into memory for execution.
        if self.h5ad_path == "":
            self.adata_summary = ""
        else:
            if execution_mode == "claude":
                print("Loading h5ad metadata for summarization (no full AnnData load)...")
                self.adata_summary = self._summarize_adata_obs_only(self.h5ad_path, length_cutoff=25)
            else:
                print("Loading anndata for summarization...")
                self.adata_summary = self._summarize_adata_full(self.h5ad_path)
            print(f"✅ Loaded summary from {self.h5ad_path}")

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
            from cellvoyager.execution.claude import ClaudeJupyterExecutor
            self.executor = ClaudeJupyterExecutor(
                **shared_executor_kwargs,
                anthropic_api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"),
                **execution_kwargs,
            )
        else:
            self.executor = IdeaExecutor(**shared_executor_kwargs)

    def _summarize_adata_full(self, h5ad_path, length_cutoff=25):
        """Summarize all AnnData attributes: .obs, .var, .obsm, .layers, .uns, .obsp, .varp."""
        try:
            adata = anndata.read_h5ad(h5ad_path, backed="r")
        except Exception as e:
            try:
                fallback = self._summarize_adata_obs_only(h5ad_path, length_cutoff)
                return f"Could not load full adata ({e}). Falling back to .obs only.\n\n" + fallback
            except Exception as e2:
                return f"Could not load adata: {e}. Fallback also failed: {e2}"

        parts = []

        # adata shape
        parts.append(f"adata shape: {adata.n_obs} cells × {adata.n_vars} genes\n")

        # .obs
        parts.append("--- adata.obs ---")
        if adata.obs is not None and len(adata.obs) > 0:
            parts.append(self._summarize_df(adata.obs, length_cutoff))
        else:
            parts.append("  (empty)")

        # .var
        if adata.var is not None and len(adata.var.columns) > 0:
            parts.append("\n--- adata.var ---")
            parts.append(self._summarize_df(adata.var, length_cutoff))

        # .obsm
        if adata.obsm is not None and len(adata.obsm) > 0:
            parts.append("\n--- adata.obsm ---")
            for k, v in adata.obsm.items():
                sh = getattr(v, "shape", "?")
                parts.append(f"  {k}: shape {sh}")

        # .varm
        if adata.varm is not None and len(adata.varm) > 0:
            parts.append("\n--- adata.varm ---")
            for k, v in adata.varm.items():
                sh = getattr(v, "shape", "?")
                parts.append(f"  {k}: shape {sh}")

        # .layers
        if adata.layers is not None and len(adata.layers) > 0:
            parts.append("\n--- adata.layers ---")
            for k, v in adata.layers.items():
                sh = getattr(v, "shape", "?")
                parts.append(f"  {k}: shape {sh}")

        # .obsp
        if adata.obsp is not None and len(adata.obsp) > 0:
            parts.append("\n--- adata.obsp ---")
            for k, v in adata.obsp.items():
                sh = getattr(v, "shape", "?")
                parts.append(f"  {k}: shape {sh}")

        # .varp
        if adata.varp is not None and len(adata.varp) > 0:
            parts.append("\n--- adata.varp ---")
            for k, v in adata.varp.items():
                sh = getattr(v, "shape", "?")
                parts.append(f"  {k}: shape {sh}")

        # .uns
        if adata.uns is not None and len(adata.uns) > 0:
            parts.append("\n--- adata.uns ---")
            for k, v in adata.uns.items():
                t = type(v).__name__
                if isinstance(v, (list, np.ndarray)):
                    extras = f" len={len(v)}"
                elif isinstance(v, dict):
                    extras = f" keys={list(v.keys())[:5]}..."
                else:
                    extras = ""
                parts.append(f"  {k}: {t}{extras}")

        if hasattr(adata, "file") and adata.file is not None:
            try:
                adata.file.close()
            except Exception:
                pass

        return "\n".join(parts)

    def _summarize_df(self, df, length_cutoff):
        if df is None or len(df) == 0:
            return "  (empty)"
        lines = []
        for col in df.columns:
            try:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) > length_cutoff:
                    vals_str = str(list(unique_vals[:length_cutoff])) + f" ... and {len(unique_vals) - length_cutoff} more"
                else:
                    vals_str = str(list(unique_vals))
                lines.append(f"  {col}: {vals_str}")
            except Exception:
                lines.append(f"  {col}: (could not summarize)")
        return "\n".join(lines)

    def _summarize_adata_obs_only(self, h5ad_path, length_cutoff):
        """Fallback: summarize only .obs when full load fails."""
        self.adata_obs = self._load_h5ad_obs(h5ad_path)
        return "Below is a description of the columns in adata.obs:\n" + self._summarize_df(self.adata_obs, length_cutoff)

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
                print(f"✅ Completed Analysis {analysis_idx+1}")

                # In interactive mode, pause between analyses so the user can review
                # the completed notebook and optionally provide feedback before continuing.
                if analysis_idx + 1 < self.num_analyses and hasattr(self.executor, "inter_analysis_pause"):
                    nb_path = os.path.join(
                        self.output_dir, f"{self.analysis_name}_analysis_{analysis_idx + 1}.ipynb"
                    )
                    _user_stopped = False
                    while True:
                        feedback = self.executor.inter_analysis_pause(nb_path, analysis_idx)
                        if feedback in ("__STOP__", "__FINISH__"):
                            print(f"⏹ User stopped after Analysis {analysis_idx + 1}.")
                            _user_stopped = True
                            break
                        if feedback.startswith("__CONTINUE_CURRENT__"):
                            user_note = feedback[len("__CONTINUE_CURRENT__"):].lstrip(":").strip()
                            print(f"📝 Extending Analysis {analysis_idx + 1} further...")
                            self.executor.resume_from_notebook(
                                nb_path, analysis_idx,
                                user_feedback=user_note or None,
                                extend=True,
                            )
                            continue
                        if feedback:
                            past_analyses += f"User feedback before Analysis {analysis_idx + 2}: {feedback}\n\n"
                        break
                    if _user_stopped:
                        break

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

    def run_resume(self, notebook_path: str, analysis_idx: int = 0):
        """
        Resume a completed analysis: re-run the notebook to restore kernel state,
        then enter interactive mode for the user to run, edit, and give feedback.
        Only supported with ClaudeJupyterExecutor.
        """
        if hasattr(self.executor, "resume_from_notebook"):
            self.executor.resume_from_notebook(notebook_path, analysis_idx)
        else:
            raise ValueError("Resume requires execution_mode=claude (ClaudeJupyterExecutor)")
