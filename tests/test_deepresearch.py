import os
import sys
from types import SimpleNamespace
from pathlib import Path

import builtins
import pytest


# Make module under test importable
THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent  # salber/CellVoyager
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import types

# Ensure importing module under test doesn't fail if openai isn't installed
if 'openai' not in sys.modules:
    sys.modules['openai'] = types.SimpleNamespace(OpenAI=lambda *a, **k: None)

import deepresearch  # noqa: E402


class _MockResponses:
    def __init__(self, response_obj):
        self._response_obj = response_obj
        self.captured_kwargs = None

    def create(self, **kwargs):
        # capture call for assertions
        self.captured_kwargs = kwargs
        return self._response_obj


class _OpenAIRecorder:
    def __init__(self, response_obj):
        self.responses = _MockResponses(response_obj)
        self.init_api_key = None

    def __call__(self, *args, **kwargs):
        # emulate openai.OpenAI(api_key=...)
        self.init_api_key = kwargs.get("api_key")
        return self


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_MODEL", raising=False)


def test_research_from_paper_summary_uses_output_text(monkeypatch):
    # Prepare a response object with output_text present
    response_obj = SimpleNamespace(output_text="DEEP REPORT TEXT")
    recorder = _OpenAIRecorder(response_obj)
    # Patch the OpenAI client constructor
    monkeypatch.setattr(deepresearch.openai, "OpenAI", recorder)

    dr = deepresearch.DeepResearcher(openai_api_key="dummy_key")
    result = dr.research_from_paper_summary("Paper S", "Adata S")

    assert result == "DEEP REPORT TEXT"
    # Ensure prompt contains provided context
    called = dr.client.responses.captured_kwargs
    assert isinstance(called, dict)
    assert "input" in called
    assert "Paper S" in called["input"]
    assert "Adata S" in called["input"]
    # Default model is set
    assert dr.model == os.environ.get("DEEP_RESEARCH_MODEL", "o4-mini-deep-research")


def test_research_from_paper_summary_parses_nested_output(monkeypatch):
    # Prepare nested output shape without output_text
    response_obj = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="text", text="Part A"), SimpleNamespace(type="text", text="Part B")],
            )
        ]
    )
    recorder = _OpenAIRecorder(response_obj)
    monkeypatch.setattr(deepresearch.openai, "OpenAI", recorder)

    dr = deepresearch.DeepResearcher(openai_api_key="dummy_key")
    result = dr.research_from_paper_summary("Summary", None)

    assert "Part A" in result and "Part B" in result


def test_env_overrides_model_and_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")
    monkeypatch.setenv("DEEP_RESEARCH_MODEL", "o3-deep-research")

    response_obj = SimpleNamespace(output_text="OK")
    recorder = _OpenAIRecorder(response_obj)
    monkeypatch.setattr(deepresearch.openai, "OpenAI", recorder)

    dr = deepresearch.DeepResearcher(openai_api_key="ctor_key")
    # Ensure env-based model override applied
    assert dr.model == "o3-deep-research"

    # Trigger a call so constructor is exercised
    _ = dr.research_from_paper_summary("P", "A")

    # The OpenAI constructor should receive the env key, not the provided ctor_key
    assert recorder.init_api_key == "env_key"


