import ast
import inspect
import importlib
import sys

def extract_call_names(source: str):
    """
    Parse `source` and return a sorted, deduplicated list of all
    function‐call names as dotted strings.
    """
    tree = ast.parse(source)
    calls = set()

    def get_full_name(node):
        # Recursively reconstruct dotted name from ast.Attribute/Name
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parent = get_full_name(node.value)
            return f"{parent}.{node.attr}"
        else:
            return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = get_full_name(node.func)
            if fname:
                calls.add(fname)
    return sorted(calls)


def resolve_obj(fqname: str, namespace: dict):
    """
    Given a dotted name and the exec’d namespace,
    traverse attributes to return the actual Python object.
    """
    parts = fqname.split(".")
    # Try first from the namespace, else import the top module
    if parts[0] in namespace:
        obj = namespace[parts[0]]
    else:
        obj = importlib.import_module(parts[0])
    for attr in parts[1:]:
        obj = getattr(obj, attr)
    return obj

def load_namespace(source: str, filename="<string>"):
    """
    Try to exec the full source. If that errors (e.g. due to an undefined call),
    fall back to only executing the import statements.
    """
    namespace = {}
    try:
        exec(compile(source, filename, 'exec'), namespace)
    except Exception:
        # Collect only import and from‑import nodes
        tree    = ast.parse(source)
        imports = [node for node in tree.body 
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        import_mod = ast.Module(body=imports, type_ignores=[])
        exec(compile(import_mod, filename, 'exec'), namespace)
    return namespace

def get_documentation(code: str) -> str:
    call_names = extract_call_names(code)
    ns         = load_namespace(code)

    docs = []
    for name in sorted(set(call_names)):
        # Only include functions that start with 'sc.' (scanpy) or 'scvi.' (scvi-tools)
        ###### MODIFY IN NEEDED FOR OTHER PACKAGES ######
        if not (name.startswith('sc.') or name.startswith('scvi.') or name.startswith('scanpy.')):
            continue
            
        try:
            fn  = resolve_obj(name, ns)
            doc = inspect.getdoc(fn) or "<no docstring>"
        except Exception as e:
            doc = f"<could not resolve: {e}>"
        docs.append(f"{name}:\n{doc}")
    return "\n\n".join(docs)


# --- Usage Example ---

code = """
import scanpy as sc
import numpy as np
from matplotlib import pyplot as plt

X = np.vstack([np.random.random((100, 2)), np.random.random((100, 2))])
plt.scatter(X[:,0], X[:,1])
"""

print(len(get_documentation(code)))