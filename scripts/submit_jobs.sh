#!/bin/bash
models=(AttentionModel GRUModel)
# datasets=(PreprocessedPhysionet2019Dataset PreprocessedMIMIC3Dataset PreprocessedEICUDataset PreprocessedHiridDataset)
datasets=(PreprocessedPhysionet2019Dataset PreprocessedMIMIC3Dataset PreprocessedHiridDataset)

res_path=results/hypersearch2
mkdir -p $res_path

for dataset in ${datasets[@]};
    do
    dataset_path=$res_path/${dataset}
    mkdir -p ${dataset_path}
    for model in ${models[@]};
    do
        out_path=${dataset_path}/${model}
        mkdir -p ${out_path}
        for rep in {1..5};
            do
            jobname="${model}-${dataset}-${rep}"
            output_log="${out_path}/${rep}.log"

            squeue --format %j | grep $jobname >> /dev/null
            retVal=$?
            if [ $retVal -ne 0 ] && [ ! -f $output_log ]; then
                sbatch -J $jobname --cpus-per-task 2 --mem-per-cpu 5G -n 1 -p gpu -o $output_log -e $output_log --gres=gpu:1 --wrap "pipenv run train_torch --dataset $dataset --model $model --hyperparam-draws 1 --gpus -1 --log-path $dataset_path --exp-name $model"
            else
                echo Skipping $jobname
            fi
        done
    done
done
