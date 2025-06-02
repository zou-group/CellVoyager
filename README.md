# CellVoyager
## Running CellVoyager
To create the necessary environment, run
```
conda env create -f CellVoyager_env.yaml
conda activate CellVoyager
```
In `run.py` set the following parameters
1. Set `h5ad_path` to the path of the anndata `.h5ad` file
2. Set `paper_summary_path` to an LLM or human generated summary of the paper (expects a `.txt` file)
3. Set `analysis_name` to what you want your analysis files to be saved under
