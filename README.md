<img src="img/logo.jpg" width="300">

This is the repository for the paper: [Predicting sepsis using deep learning across international sites: a retrospective development and validation study](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(23)00301-2/fulltext) 

### Reference: 

```latex
@article{moor2023predicting,
  title={Predicting sepsis using deep learning across international sites: a retrospective development and validation study},
  author={Moor, Michael and Bennett, Nicolas and Ple{\v{c}}ko, Drago and Horn, Max and Rieck, Bastian and Meinshausen, Nicolai and B{\"u}hlmann, Peter and Borgwardt, Karsten},
  journal={eClinicalMedicine},
  volume={62},
  pages={102124},
  year={2023},
  publisher={Elsevier}
}
```

### Code:  

Over the next days, we plan to step-by-step clean-up the following components:  

- R code for data loading / harmonization  
- Python code for pre-prorcessing (feature extraction), normalization etc. (assumes a Dask pipeline that can be run on a large CPU server or cluster)  
- Python code for model development (both deep learning models in PyTorch, and classic models using sklearn), finetuning, calibration   

### Acknowledgements:  

This project was a massive effort stretching over 4 years and over 1.5K commits. 

Code contributors:  

[Michael](https://github.com/mi92), [Nicolas](https://github.com/nbenn), [Max](https://github.com/ExpectationMax), [Bastian](https://github.com/Pseudomanifold), and [Drago](https://github.com/dplecko) 


## Data setup

In order to set up the datasets, the R package `ricu` (available via CRAN) is required alongside access credentials for [PhysioNet](https://physionet.org) and a download token for [AmsterdamUMCdb](https://amsterdammedicaldatascience.nl/#amsterdamumcdb). This information can then be made available to `ricu` by setting the environment variables `RICU_PHYSIONET_USER`, `RICU_PHYSIONET_PASS` and `RICU_AUMC_TOKEN`.

```r
install.packages("ricu")
Sys.setenv(
    RICU_PHYSIONET_USER = "my-username",
    RICU_PHYSIONET_PASS = "my-password",
    RICU_AUMC_TOKEN = "my-token"
)
```

Then, by sourcing the files in `r/utils`, which will require further R packages to be installed (see `r/utils/zzz-demps.R`), the function `export_data()` becomes available. This roughly loads data corresponding to the specification in `config/features.json`, on an hourly grid, performs some patient filtering and concludes with some missingness imputation/feature augmentation steps. The script under `r/scripts/create_dataset.R` can be used to carry out these steps.

```r
install.packages(
    c("here", "arrow", "bigmemory", "jsonlite", "data.table", "readr",
      "optparse", "assertthat", "cli", "memuse", "dplyr",
      "biglasso", "ranger", "qs", "lightgbm", "cowplot", "roll")
)

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

for (x in c("mimic", "eicu", "hirid", "aumc")) {

  if (!is_data_avail(x)) {
    msg("setting up `{x}`\n")
    setup_src_data(x)
  }

  msg("exporting data for `{x}`\n")
  export_data(x)
}
```

In order to preprocess the [PhysioNet 2019 challenge dataset](https://physionet.org/content/challenge-2019/1.0.0/), the downloaded data can be unpacked, followed by running `export_data()` as

```r
physio_dir <- data_path("physionet2019")
download.file(
    paste("https://archive.physionet.org/users/shared/challenge-2019",
          "training_setB.zip", sep = "/"),
    file.path(physio_dir, "training_setB.zip")
)
unzip(file.path(physio_dir, "training_setB.zip"), exdir = physio_dir)
export_data("physionet2019", data_dir = physio_dir)
```

If `export_data()` is called with a default argument of `data_path("export")` for `dest_dir`, this will create one parquet file per data source under `data-export`. This procedure can also be run using the PhysioNet demo datasets for debugging and to make sure it runs through:

```r
install.packages(
  c("mimic.demo", "eicu.demo"),
  repos = "https://eth-mds.github.io/physionet-demo"
)

for (x in c("mimic_demo", "eicu_demo")) {
  export_data(x)
}
```

## Python pipeline (for the machine learning / modelling side):  

For transparency, we include the full list of requirements we used throughout this study in  
```requirements_full.txt```
However, some individual packages may not be supported anymore, hence to get started you may want to start with  
```requirements_minimal.txt```  

For example, by activating your virtual environment, and running:  
```pip install -r requirements_minimal.txt```  

For setting up this project, we ran:    
```>pipenv install```  
```>pipenv shell``` 
Hence, feel free to also check out the Pipfile / Pipfile.lock 




### Datasets   

(internal usage: run ``` source scripts/download_from_euler.sh ``` )  
All data will be downloaded to:   
```datasets/downloads/```  

### Source code

`src`:
- `torch`: pytorch-based pipeline and models (currently an attention model)  
    TODO: add docu for training a  model  
- `sklearn`: sklearn-based pipeline for boosted trees baselines

## Preprocessing  
 
### Running the preprocessing
```source scripts/run_preprocessing.sh```  

## Training  

### Model overview   
- src/torch: pytorch-based pipeline and models (currently GRU and attention model)  
- src/sklearn: sklearn-based pipeline for lightGBM and LogReg models 

### Running the LightGBM hyperparameter search      
 ```>source scripts/run_lgbm.sh <results_folder_name>```   

### After having run the LightGBM hyperparameter search, run repetitions with:        
 ```>source scripts/run_lgbm_rep.sh <results_folder_name>```   

### Running the baseline models hyperparameter search + repetitions (in one)   
 ```>source scripts/run_baselines.sh <results_folder_name>```   

### Deep models / torch pipeline
These jobs we currently run on bs-slurm-02.

First, compile a sweep on wandb.ai, using the sweep-id, (only the id -- not the entire id-path) run:  
 ```>source scripts/wandb/submit_job.sh sweep-id```  
In this submit_job script you can configure the variable `n_runs`, i.e. how many evaluations should be run (e.g. 25 during coarse or fine tuning search,
or 5 for repetition runs)

#### Training a single dataset and model
Fitting an attention-model on Physionet: #TODO update this

## Evaluation pipeline  

### Shallow models + Baselines  

```>source scripts/eval_sklearn.sh <results_folder_name>``` where the results folder refers to the output folder of the hyperparameter search
Make sure that the eval_sklearn script reads all those methods you wish to evaluate. This script already assumes that repetitions are available.  

### Deep models  

First determine the best run of your sweep, giving you a run-id.
First apply this model to all datasets:  
```>source scripts/wandb/submit_evals.sh run-id```   
Once this is completed, the prediction files can be processed in the patient eval:  
```>source scripts/eval_torch.sh run-id```  

For evaluating a repetition sweep, run (on slurm)   
```>pipenv run python scripts/wandb/get_repetition_runs.py sweep-id1 sweep-id2 ..``` and once completed, run (again cpu server):    
```>python scripts/wandb/get_repetition_evals.py sweep-id1 sweep-id2 ..```.  

## Results and plots

For gathering all repetition results, run:  
```>python -m scripts.plots.gather_data --input_path results/evaluation_validation/evaluation_output_subsampled --output_path results/evaluation_validation/plots/ ```  

For creating ROC plots, run:  
```>python scripts/plots/plot_roc.py --input_path results/evaluation/plots/result_data.csv```  

For creating precision/earliness plots, run:
```>python -m scripts.plots.plot_scatterplots results/evaluation/plots/result_data.csv --r 0.80 --point-alpha 0.35 --line-alpha 1.0 --output results/evaluation/plots/```  
For the scatter data, in order to return 50 measures (5 repetition splits, 10 subsamplings), set ```--aggregation micro```

## Pooled predictions  

First, we need to create a mapping from experiments (data_train,data_eval, model etc) to the prediction files:  
```>python scripts/map_model_to_result_files.py <path_to_predictons> --output_path <output_json_path> ``` Use --overwrite, to overwrite an existing mapping json. 
 
Next we actually pool the predictions:  
```>source scripts/pool_predictions.sh```    

Then, we evaluate them:  
```>source scripts/eval_pooled.sh```  
To create plots with the pooled predictions, run:  
```>python -m scripts.plots.gather_data --input_path results/evaluation_test/prediction_pooled_subsampled/max/evaluation_output --output_path results/evaluation_test/prediction_pooled_subsampled/max/plots/```  
```>python scripts/plots/plot_roc.py --input_path results/evaluation_test/prediction_pooled_subsampled/max/plots/result_data_subsampled.csv```  
For computing precision/earliness, run:  
```python -m scripts.plots.plot_scatterplots results/evaluation_test/prediction_pooled_subsampled/max/plots/result_data_subsampled.csv --r 0.80 --point-alpha 0.35 --line-alpha 1.0 --output results/evaluation_test/prediction_pooled_subsampled/max/plots/``` 
And heatmap incl. pooled preds:  
```>python -m scripts.make_heatmap results/evaluation_test/plots/roc_summary_subsampled.csv --pooled_path results/evaluation_test/prediction_pooled_subsampled/max/plots/roc_summary_subsampled.csv``` 

