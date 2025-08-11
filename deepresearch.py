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

            response = self.client.responses.create(**kwargs)
            text = self._extract_output_text(response)
            return text or ""
        except Exception as e:
            # Be silent to keep upstream behavior unchanged; caller already guards
            return ""

    def research_from_paper_summary(
        self, paper_summary: str, adata_summary: Optional[str], available_packages: str) -> str:
        """Invoke Deep Research using provided dataset/paper context.
        """
        user_prompt = open(os.path.join(os.path.dirname(__file__), "prompts", "deepresearch.txt")).read()
        user_prompt = user_prompt.format(paper_summary=paper_summary, adata_summary=adata_summary, available_packages=available_packages)

        return self._run_deep_research(user_prompt, max_output_tokens=64_000)
