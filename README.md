<div align="center">
<img src="gui/assets/logo.jpeg" alt="CellVoyager Logo" width="700">
</div>

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/zou-group/CellVoyager.git
cd CellVoyager
conda env create -f environment.yml
conda activate CellVoyager
```

CellVoyager requires API keys depending on which execution mode you use. Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

## Usage

### Terminal

```bash
python run_cellvoyager.py --h5ad-path PATH_TO_H5AD_DATASET \
                          --paper-path PATH_TO_PAPER_SUMMARY \
                          --analysis-name RUN_NAME
```

| Argument | Description |
|---|---|
| `--h5ad-path` | Path to the anndata `.h5ad` file |
| `--paper-path` | Path to a `.txt` file containing a summary of the paper / biological context |
| `--analysis-name` | Name for the analysis output directory |
| `--execution-mode` | `claude` (default) or `legacy` |
| `--model-name` | LLM for hypothesis generation (default: `claude-sonnet-4-6`) |
| `--num-analyses` | Number of analyses to run (default: 1) |
| `--max-iterations` | Max iterations per analysis (default: 8) |
| `--interactive` | Pause after each step so you can edit the notebook in Jupyter |

Run `python run_cellvoyager.py --help` for the full list of options.

### GUI

```bash
streamlit run gui/app.py
```

This opens a browser-based interface where you can upload datasets, configure settings, and monitor analyses in real time.

## Example

We use the COVID-19 case study from [Wilk et al. 2020](https://www.nature.com/articles/s41591-020-0944-y).

Download the `.h5ad` object:

```bash
curl -o example/covid19.h5ad "https://hosted-matrices-prod.s3-us-west-2.amazonaws.com/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection-25/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection.h5ad"
```

An example manuscript summary is already included in `example/covid19_summary.txt`.

Then run with defaults (uses the COVID-19 dataset):

```bash
python run_cellvoyager.py
```

Output notebooks appear in the `outputs/` directory and update in real time.

## CellBench

To run base LLMs (gpt-4o, o3-mini) 3x on CellBench:

```
cd CellBench
python run_base_llm.py
python run_llm_judge.py
```

To run agent 3x on CellBench:

```
cd CellBench
python run_agent.py {gpt-4o|o3-mini}
```

Metrics should be printed to stdout and saved in the `responses` and `judged` dirs.
