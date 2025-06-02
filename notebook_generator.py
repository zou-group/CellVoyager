import nbformat
import os

def generate_notebook(completed_analyses, output_dir):
    # Create a new notebook
    nb = nbformat.v4.new_notebook()
    
    # Add a title cell
    nb.cells.append(nbformat.v4.new_markdown_cell("# Analysis Results"))
    
    # Loop through all completed analyses and add them as markdown + code cells
    for analysis in completed_analyses:
        description = analysis.get("description", "No description provided")
        code = analysis.get("code", "")
        conclusion = analysis.get("conclusion", "No conclusion provided")
        
        nb.cells.append(nbformat.v4.new_markdown_cell(f"## Hypothesis\n\n{description}"))
        nb.cells.append(nbformat.v4.new_code_cell(code))
        nb.cells.append(nbformat.v4.new_markdown_cell(f"### Conclusion\n\n{conclusion}"))
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the notebook
    notebook_path = os.path.join(output_dir, "analysis_results.ipynb")
    with open(notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    print(f"✅ Notebook generated: {notebook_path}")
