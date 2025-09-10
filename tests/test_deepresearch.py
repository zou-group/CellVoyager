import os
import sys
from types import SimpleNamespace
import types
from pathlib import Path
import openai
import builtins
import pytest


# Make module under test importable
THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent  # salber/CellVoyager
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import deepresearch  # noqa: E402


def test_real_openai_deepresearch_runs_and_mentions_covid():
    # Use the real client (no monkeypatch) to verify API wiring works.
    # This asserts non-empty output and that COVID-related terms appear.
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"DEBUG: API key from env: {api_key[:10] if api_key else 'None'}...")
    dr = deepresearch.DeepResearcher(openai_api_key=api_key)
    paper_summary = (
        "Study of COVID-19 patient PBMC scRNA-seq; prior analyses included clustering and marker-based annotation."
    )
    adata_summary = (
        "PBMC, multiple donors, severe/mild cohorts, timepoints; cell types include T cells, B cells, monocytes."
    )
    result = dr.research_from_paper_summary(paper_summary, adata_summary)
    print("RESULT: ", result)
    assert isinstance(result, str) and len(result.strip()) > 0
    low = result.lower()
    assert ("covid" in low) or ("sars" in low) or ("coronavirus" in low)


