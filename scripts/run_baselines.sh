result_path=$1

datasets=(aumc physionet2019 eicu mimic3 hirid)
baselines=(sofa sirs qsofa mews news)

for dataset in ${datasets[@]}; 
    do
    for baseline in ${baselines[@]};
    do  
    python -m src.sklearn.main --dataset $dataset --method $baseline --result_path $result_path  --cv_n_jobs=1
    done
done

