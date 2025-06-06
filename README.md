<div align="center">
<img src="images/logo.jpeg" alt="CellVoyager Logo" width="700">
</div>

## Running CellVoyager
First clone the current repository as:
```
git clone https://github.com/zou-group/CellVoyager.git
cd CellVoyager
```


To create the necessary environment, run
```
conda env create -f CellVoyager_env.yaml
conda activate CellVoyager
```
In `run.py` set the following parameters
1. Set `h5ad_path` to the absolute path of the anndata `.h5ad` file
2. Set `paper_summary_path` to the absolute path of a `.txt` file containing the LLM or human generated summary of the paper
3. Set `analysis_name` to what you want your analysis files to be saved under


From there, all you need to do to run the agent is to execute: `python run.py`


The current implementation of the model only support OpenAI models. As a result, it assumes you have `.env` file that contains
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

## Example
We are going to use the COVID-19 case study from the CellVoyager, which builds on [this paper](https://www.nature.com/articles/s41591-020-0944-y).


To download the `.h5ad` object run
```
curl -o example/covid19.h5ad "https://hosted-matrices-prod.s3-us-
west-2.amazonaws.com/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection-25/Single_cell_atlas_of_peripheral_immune_response_to_S
ARS_CoV_2_infection.h5ad"
```
An example summary of the associated manuscript is already included in `example/covid19_summary.txt`.


Then simply run `python run.py` which by default uses the COVID-19 dataset and manuscript summary. You will see the Jupyter notebooks in an `outputs` directory, which will update the notebook in real-time. Currently, the notebooks are run sequentially, but we are currently experimenting with ways to parallelize this.

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
