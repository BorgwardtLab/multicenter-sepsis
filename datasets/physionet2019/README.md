# Physionet 2019

In this directory, we gather code and files covering the Physionet 2019 Dataset.

To set up the environment, run 
```>pipenv install --skip-lock```

```>pipenv shell```

To download, extract this dataset, simply run 
```>make```

This will 
    - download the Physionet2019 (public) datasets trainingA and trainingB into ```data/downloads```
    - extract the zip files of both sets patients in to one directory ```data/extracted```
    - (additionally) gather all patient files in one binary pickle file ```data/combined.pck```    
    - create stratified shuffled splits (5 by default) and write the split infos to  ```data/split_info.pkl```    
