# multicenter-sepsis

In this repo, we gather pipelining code for a multicenter sepsis prediction effort.

## 0. Python environment:  
To set up python libraries run:  
```>pipenv install --skip-lock```  
```>pipenv shell```  

`datasets`: (code for downloading and extracting this dataset)
- `physionet2019`
--> inside the dataset folder, to download and extract the dataset, simply run:  
 ```make```  

for the other datasets: mimic3, eicu, hirid: first download the zip files from polybox (internal for now),
then move the zips to the following path (example with e-ICU):  
```datasets/eicu/data/downloads/eicu.zip```  

`src`:
- `torch`: pytorch-based pipeline and models (currently an attention model)  
    TODO: add docu for training a  model  
- `sklearn`: sklearn-based pipeline for boosted trees baselines
    
To set up python libraries run:  
```pipenv install --skip-lock```  
```pipenv shell```  
  
For the sklearn pipeline:  
```python src/sklearn/data/make_dataframe.py``` to run online and parallelized preprocessing, feature extraction  
```python src/sklearn/main.py``` to run a hyperparameter search of a sklearn-based online classifier  
  
For additional arguments, use ```--help```.  
 
For preprocessing a single dataset, e.g. "hirid", simply run   
```>python src/sklearn/data/make_dataframe.py --dataset hirid```  

### Sanity checks:
For checking, if all datasets were preprocessed properly (e.g. dimensions of all instances are consistent within a dataset (one-hot encoding of categorical variables could cause problems over different splits)), run:  
```>python -m scripts.sanity_checks.check_datasets```
 
## 3. Model Pipelines
### Overview   
- src/torch: pytorch-based pipeline and models (currently GRU and attention model)  
- src/sklearn: sklearn-based pipeline for lightGBM and LogReg models 

### Running a sklearn model hyperparameter search      
   
For running a hyperparameter search over all datasets using Logistic Regression, run:  
```>source scripts/run_lr.sh <results_folder_name>```  
For running a hyperparameter search of all datasets using LightGBM, run:  
```>source scripts/run_lgbm.sh <results_folder_name>```

#### Training a single dataset and classifier:     
```>python src/sklearn/main.py``` runs a hyperparameter search of a sklearn-based online classifier.
For additional arguments, use ```--help```. 

#### Example: training a LGBM on e-ICU:   
```>python -m src.sklearn.main --result_path=internal_validation/ --dataset eicu --method=lgbm --n_iter_search=100 --cv_n_jobs=1 ```

### Deep models / torch pipeline
These jobs we currently run on bs-slurm-02.

#### Starting hyperparameter search for all datasets and all deep models:
```>source scripts/submit_jobs.sh results/hypersearch3``` where `results/hypersearch3` is an example of a result path.  
 
#### Training a single dataset and model
--> this should also work on bs-gpu13!   
Fitting an attention-model on Physionet:    
```>python -m src.torch.train_model --dataset PreprocessedPhysionet2019Dataset --model AttentionModel --max-epochs=100 --label-propagation=6 --gpus 0 ```   


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

