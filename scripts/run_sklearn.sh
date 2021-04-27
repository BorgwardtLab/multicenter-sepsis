#!/bin/bash
result_path=$1
#res1=results/hypersearch12_regression_5iter/
#res2=results/hypersearch11_classification/
feature_set=middle
cost=5
##LGBM
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=20 --n_iter_search=1 --dataset=mimic_demo --task regression
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=20 --dataset=mimic --task regression

# python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=5 --dataset=mimic --task classification 
# python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=5 --dataset=mimic --task regression 

#python -m src.sklearn.main --result_path $res1 --cv_n_jobs=1 --n_iter_search=5 --dataset=aumc --task regression
#python -m src.sklearn.main --result_path $res2 --cv_n_jobs=1 --n_iter_search=5 --dataset=aumc --task classification  

#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=1 --dataset=mimic_demo --task regression --feature_set $feature_set
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=5 --dataset=mimic --task regression --feature_set $feature_set 
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=5 --dataset=aumc --task regression --feature_set $feature_set 

#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=1 --dataset=mimic_demo --task regression --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=5 --dataset=mimic --task regression --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=5 --dataset=aumc --task regression --feature_set $feature_set --cost=$cost

#hp10:
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=5 --dataset=mimic --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=5 --dataset=aumc --task classification --feature_set $feature_set --cost=$cost

#hp13:
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=aumc --task regression --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=mimic --task regression --feature_set $feature_set --cost=$cost

#hp10:
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=aumc --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=mimic --task classification --feature_set $feature_set --cost=$cost

#hp13:
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=aumc --task regression --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=mimic --task regression --feature_set $feature_set --cost=$cost

#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=10 --n_iter_search=20 --dataset=aumc --task classification --feature_set $feature_set --target_name average_precision 

#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=10 --n_iter_search=20 --dataset=aumc --task classification --feature_set $feature_set --method rf 

#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=10 --n_iter_search=20 --dataset=aumc --task classification --feature_set $feature_set --method rf --label-propagation 0 

#hp13:
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=10 --n_iter_search=50 --dataset=aumc --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=4 --n_iter_search=50 --dataset=mimic --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=4 --n_iter_search=50 --dataset=hirid --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=eicu --task classification --feature_set $feature_set --cost=$cost


#hp10:
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=10 --n_iter_search=50 --dataset=aumc --task regression --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=4 --n_iter_search=50 --dataset=mimic --task regression --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=4 --n_iter_search=50 --dataset=hirid --task regression --feature_set $feature_set --cost=$cost

#hp13:
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=3 --n_iter_search=50 --dataset=eicu --task regression --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=3 --n_iter_search=50 --dataset=eicu --task classification --feature_set $feature_set --cost=$cost

## CURRENT:

#hp13:
#LGBM:
#python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=aumc --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=mimic --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=hirid --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=eicu --task classification --feature_set $feature_set --cost=$cost

## LGBM REPETITIONS:
#for rep in {0..4}; do
#    python -m src.sklearn.fit_rep --method lgbm --result_path $result_path --dataset=aumc --task classification --feature_set $feature_set --cost=$cost --rep $rep
#    python -m src.sklearn.fit_rep --method lgbm --result_path $result_path --dataset=mimic --task classification --feature_set $feature_set --cost=$cost --rep $rep
#    python -m src.sklearn.fit_rep --method lgbm --result_path $result_path --dataset=hirid --task classification --feature_set $feature_set --cost=$cost --rep $rep
#    python -m src.sklearn.fit_rep --method lgbm --result_path $result_path --dataset=eicu --task classification --feature_set $feature_set --cost=$cost --rep $rep
#done 

## BASELINES REPETITIONS:

methods=(news) #sofa qsofa sirs mews news)
for method in ${methods[@]}; do
    for rep in {0..4}; do
        python -m src.sklearn.main --method $method --result_path $result_path --cv_n_jobs=5 --n_iter_search=50 --dataset=aumc --task classification --feature_set $feature_set --cost=$cost --rep $rep 
        python -m src.sklearn.main --method $method --result_path $result_path --cv_n_jobs=3 --n_iter_search=50 --dataset=mimic --task classification --feature_set $feature_set --cost=$cost --rep $rep
        python -m src.sklearn.main --method $method --result_path $result_path --cv_n_jobs=3 --n_iter_search=50 --dataset=hirid --task classification --feature_set $feature_set --cost=$cost --rep $rep
        python -m src.sklearn.main --method $method --result_path $result_path --cv_n_jobs=3 --n_iter_search=50 --dataset=eicu --task classification --feature_set $feature_set --cost=$cost --rep $rep
    done
done


# dask LR:
# python -m src.dask_ml.dask_main --method lr --result_path $result_path --dataset aumc --task classification --n_iter_search=5 
