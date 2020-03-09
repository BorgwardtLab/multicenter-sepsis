# multicenter-sepsis

In this repo, we gather pipelining code for a multicenter sepsis prediction effort.

To set up a dataset, run the Rscript in `/bin` for example as

```
./bin/create_dataset.R -s "mimic_demo"
```

datasets:
    - physionet2019 (code for downloading and extracting this dataset)

src: 
    - torch: pytorch-based pipeline and models (currently an attention model)
    - sklearn: sklearn-based pipeline for boosted trees baselines
   




## TODOs:
    - make sure that all pipelines utilize the same predefined splits! 
        - for phyisionet2019, use ```datasets/physionet2019/data/split_info.pkl``` as reference 
        
