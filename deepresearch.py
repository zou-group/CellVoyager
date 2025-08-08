from __future__ import annotations
import os
from typing import Optional
import openai


class DeepResearcher:
    """Thin wrapper over OpenAI Deep Research models.

    Keeps the original public interface so callers in `agent.py` do not need to change.
    """

    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        # Allow overriding via env; default to lightweight for faster turnaround
        self.model = os.environ.get(
            "DEEP_RESEARCH_MODEL",
            "o4-mini-deep-research",
        )

    def _extract_output_text(self, response) -> str:
        # Best effort extraction across SDK snapshots
        try:
            text = getattr(response, "output_text", None)
            if isinstance(text, str) and text.strip():
                return text
        except Exception:
            pass

        try:
            parts = []
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            t = getattr(c, "text", None)
                            if isinstance(t, str):
                                parts.append(t)
                        elif getattr(c, "type", None) == "text":
                            t = getattr(c, "text", None)
                            if isinstance(t, str):
                                parts.append(t)
                            elif isinstance(t, dict) and "value" in t:
                                parts.append(str(t.get("value", "")))
            return "\n".join(p for p in parts if p).strip()
        except Exception:
            return ""

    def _run_deep_research(self, prompt: str, max_output_tokens: Optional[int] = None) -> str:
        try:
            kwargs = {
                "model": self.model,
                "input": prompt,
                "tools": [{"type": "web_search_preview"}],  # Required for deep research models
            }
            # Respect optional max tokens; some users report truncation defaults
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens

            #print("PROMPT: ", prompt)
            #print("KWARGS: ", kwargs)
            response = self.client.responses.create(**kwargs)
            print("RESPONSE: ", response)
            text = self._extract_output_text(response)
            print("TEXT: ", text)
            return text or ""
        except Exception as e:
            # Be silent to keep upstream behavior unchanged; caller already guards
            print("ERROR: ", e)
            return ""

    def research_from_paper_summary(
        self, paper_summary: str, adata_summary: Optional[str]) -> str:
        """Invoke Deep Research using provided dataset/paper context.
        """
        context_sections = []
        if paper_summary:
            context_sections.append(f"Paper summary:\n{paper_summary}")
        if adata_summary:
            context_sections.append(f"AnnData overview:\n{adata_summary}")

        context = "\n\n".join(context_sections)
        user_prompt = f"""
You are a computational single-cell transcriptomics expert.

Use the paper context below and perform additional web research as needed. Produce four concise sections with citations. Emphasize biological background relevant for scRNA-seq and avoid repeating analyses already performed in the paper.

1) Background for scRNA-seq: Concise biological background on the disease that is relevant for scRNA-seq analysis. Do not restate analyses or findings already performed in the paper. Cite sources.
2) Already tried in the paper: Brief bullet list summarizing analyses/methods the paper already performed (to establish what to avoid duplicating). Keep this short.
3) Dataset-aware considerations: Based on dataset metadata in the context (e.g., samples, tissues, cell types, perturbations), list biological considerations and potential confounders that matter for downstream scRNA-seq analysis. Cite sources where applicable.
4) Untested, feasible analyses: 4–8 bullet points for analyses not yet tested in the paper, feasible with scanpy/scvi/CellTypist only, and explicitly distinct from (2). Cite where relevant.

Context:\n{context}
""".strip()

        return self._run_deep_research(user_prompt, max_output_tokens=64_000)
