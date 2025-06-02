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

The current implementation of the model only support OpenAI models. As a result, it assumes you have `.env` file that contains
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

## Example
We are going to use the COVID-19 case study from the CellVoyager, which builds on this paper.
To download the `.h5ad` object run
```
curl -o example/covid19.h5ad "https://hosted-matrices-prod.s3-us-
west-2.amazonaws.com/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection-25/Single_cell_atlas_of_peripheral_immune_response_to_S
ARS_CoV_2_infection.h5ad"
```
An example summary of the associated is already included in `example/covid19_summary.txt`, but feel free to replace this with your own summary of the COVID-19 manuscript, where you can add analyses you wish CellVoyager to avoid.
Then simply run `python run.py` which by default uses the COVID-19 dataset and manuscript summary.
