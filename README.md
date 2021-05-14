# multicenter-sepsis

In this repo, we gather pipelining code for a multicenter sepsis prediction effort.

## Python pipeline:  
To set up python libraries run:  
```>pipenv install```  
```>pipenv shell```  

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
```>python -m scripts.plots.gather_data --input_path results/evaluation/evaluation_output_subsampled/ ```  

For creating ROC plots, run:  
```>python scripts/plots/plot_roc.py --input_path results/evaluation/plots/result_data.csv```  

For creating precision/earliness plots, run:
```>python -m scripts.plots.plot_scatterplots results/evaluation/plots/result_data.csv --r 0.80 --point-alpha 0.65 --line-alpha 1.0 --output results/evaluation/plots/```  

## R-code pipeline
This was used for creating the harmonized raw datasets.

To set up a dataset, run the Rscript in `r/bin` for example as

```r
.r/bin/create_dataset.R -s "mimic_demo"
```

This requires the packages `optparse` and `ricu` with can be installed as

```r
install.packages(c("optparse", "devtools"))
devtools::install_github("septic-tank/ricu")
```

