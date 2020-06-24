# multicenter-sepsis

In this repo, we gather pipelining code for a multicenter sepsis prediction effort.


datasets: (code for downloading and extracting this dataset)
- physionet2019   
--> inside the dataset folder, to download and extract the dataset, simply run:  
 ```make```  

for the other datasets: mimic3, eicu, hirid: first download the zip files from polybox (internal for now),
then move the zips to the following path (example with e-ICU):  
```datasets/eicu/data/downloads/eicu.zip```  


src:
- torch: pytorch-based pipeline and models (currently an attention model)  
    TODO: add docu for training a  model  
- sklearn: sklearn-based pipeline for boosted trees baselines
    
To set up python libraries run:  
```pipenv install --skip-lock```  
```pipenv shell```  
  
For the sklearn pipeline:  
```python src/sklearn/data/make_dataframe.py``` to run online and parallelized preprocessing, feature extraction  
```python src/sklearn/main.py``` to run a hyperparameter search of a sklearn-based online classifier  
  
For additional arguments, use ```--help```.  
 


## Training a LGBM on e-ICU:  
```>python -m src.sklearn.main --result_path=internal_validation/ --dataset eicu --method=lgbm --n_iter_search=100 --cv_n_jobs=1 ```


## Fitting an attention-model on Physionet:  
```python -m src.torch.train_model --dataset PreprocessedPhysionet2019Dataset --max-epochs=100 --label-propagation=6 --gpus 0 ```  


## R-code:
To set up a dataset, run the Rscript in `r/bin` for example as

```r
.r/bin/create_dataset.R -s "mimic_demo"
```

This requires the packages `optparse` and `ricu` with can be installed as

```r
install.packages(c("optparse", "devtools"))
devtools::install_github("septic-tank/ricu")
```

