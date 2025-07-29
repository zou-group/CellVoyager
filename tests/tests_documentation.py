import pytest
import importlib
import inspect
import math
import json

from utils import *

# pip install pytest
# pytest tests.py

###### ---------- Function Extraction From Code ---------- #######

def test_extract_call_names_simple():
    source = "import math\nx = math.sqrt(4)\n"
    names = extract_call_names(source)
    assert names == ["math.sqrt"]

def test_extract_call_names_multiple():
    source = '''
import os
from sys import platform
val = os.path.join("a", "b")
res = platform.startswith("win")
'''
    names = extract_call_names(source)
    assert sorted(names) == sorted(["os.path.join", "platform.startswith"])

def test_extract_call_names_ignores_non_calls():
    source = "x = 1 + 2\n"
    names = extract_call_names(source)
    assert names == []

def test_load_namespace_exec_and_imports():
    source = "import math\ny = math.pi\nz = 42"
    ns = load_namespace(source)
    # math should be imported into the namespace
    assert "math" in ns
    # math.pi should match the real math.pi
    assert ns["math"].pi == math.pi
    # simple variables should be set
    assert ns["z"] == 42

def test_resolve_obj_from_namespace():
    source = "import math\nfrom functools import reduce"
    ns = load_namespace(source)
    # resolve a name imported into the namespace
    fn_reduce = resolve_obj("reduce", ns)
    assert fn_reduce == ns["reduce"]
    # resolve a module method in the namespace
    fn_sin = resolve_obj("math.sin", ns)
    assert fn_sin == ns["math"].sin

def test_resolve_obj_from_import():
    ns = {}
    # resolve a standard library function not yet in namespace
    fn_dumps = resolve_obj("json.dumps", ns)
    assert fn_dumps == json.dumps


def test_get_documentation_single_call():
    code = "import math\nx = math.sqrt(16)\n"
    docs = get_documentation(code)
    # Should mention the function name
    assert "math.sqrt:" in docs
    # Should include the actual docstring for math.sqrt
    assert inspect.getdoc(math.sqrt) in docs

def test_get_documentation_multiple_calls():
    code = """
import math
y = math.cos(0)
z = math.sin(0)
"""
    docs = get_documentation(code)
    # Both functions should be documented
    assert "math.cos:" in docs
    assert "math.sin:" in docs
    # Their docstrings should be present
    assert inspect.getdoc(math.cos) in docs
    assert inspect.getdoc(math.sin) in docs

def test_get_documentation_unresolved_call():
    code = "foo_bar_baz()\n"
    docs = get_documentation(code)
    # The unresolved function name should appear
    assert "foo_bar_baz:" in docs
    # It should report a resolution error
    assert "<could not resolve:" in docs