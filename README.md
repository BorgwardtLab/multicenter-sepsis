# multicenter-sepsis

In this repo, we gather pipelining code for a multicenter sepsis prediction effort.


datasets: (code for downloading and extracting this dataset)
- physionet2019 
- mimic3
--> inside the dataset folder, to download and extract the dataset, simply run:
 ```make```

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

