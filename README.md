<div align="center">
<img src="gui/assets/logo.jpeg" alt="CellVoyager Logo" width="700">
</div>

# Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/zou-group/CellVoyager.git
cd CellVoyager
conda env create -f environment.yml
conda activate CellVoyager
```

CellVoyager requires API keys depending on which models you use. Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

# Usage

## GUI (Recommended)

```bash
streamlit run gui/app.py
```

This opens a browser-based interface where you can upload datasets, configure settings, and monitor analyses in real time. 

It also simulatenously builds a Jupyter notebook in your `outputs/` folder.

### GUI Home Screen

<img width="1881" height="906" alt="image" src="https://github.com/user-attachments/assets/bb78208d-fe36-4b5b-a8f1-4f9eff1fb34e" />


| Input | Description |
|---|---|
| Dataset | Drag and drop a .h5ad AnnData file from your computer |
| Dataset Summary | A summary for the dataset you are inputting (e.g. which diseases, tissues, etc. are in the dataset) |
| Past Analyses Tried | Any analyses that you've already conducted and want the agent to build on top of |
| Directions to Focus On | Guides the agent on which general topics you want to explor (e.g. IL-17 pathway genes) |
| Additional Biological Background | Any biological background you think would benefit the agent |
| Analysis Name | The name under which your analysis will be saved in the `outputs/` folder |
| Analyses | How many analyses you want the agent to conduct |
| Max steps per analysis | Maximum number of steps for each analysis (can always extend during the analysis if needed) |
| Interactive mode | Pauses at every N steps (specified below the checkbox) for user feedback; recommended to have on |
| Notify | Plays a notification sound when the agent is ready for user feedback |
| DeepResearch | Whether or not to call OpenAI's DeepResearch agent to get additional biological background prior to idea generation |
| Execution model | Select from the options which LLM to use for all code generation (if using custom; use LiteLLM naming convention) |
| Hypothesis generation model | Select from the options which LLM to use for hypothesis generation |

### GUI Interactive Screen

<img width="2982" height="1422" alt="image" src="https://github.com/user-attachments/assets/ece22db1-dcb9-4779-a5ac-51b0301afb51" />


| Input | Description |
|---|---|
| Feedback for the agent | Any feedback you want to give the agent for its next N steps |
| Continue Analysis | Continues the analysis with any provided feedback |
| Edit Analysis | Lets you edit, insert, and run code cells. Ideal for manually extending/exploring an agent's analysis |
| Finish Analysis | Instructs the agent to wrap up the analysis and summarize its findings |
| Chat with Agent | Live chatbox (one for each analysis) with the agent |


## Terminal

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

The agent will work in a live Jupyter notebook and the user can interact with the agent via the terminal (if `--interactive` is enabled).

# Example

We use the COVID-19 case study from [Wilk et al. 2020](https://www.nature.com/articles/s41591-020-0944-y).

Download the `.h5ad` object:

```bash
curl -o example/covid19.h5ad "https://hosted-matrices-prod.s3-us-west-2.amazonaws.com/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection-25/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection.h5ad"
```

Then either run the GUI and drag the dataset into it or run it with `python run_cellvoyager.py`

# CellBench

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
