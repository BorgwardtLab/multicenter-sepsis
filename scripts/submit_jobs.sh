#!/bin/bash
models=(AttentionModel GRUModel)
datasets=(PreprocessedPhysionet2019Dataset PreprocessedMIMIC3Dataset PreprocessedEICUDataset PreprocessedHiridDataset)

res_path=results/hypersearch
mkdir -p $res_path

for model in ${models[@]};
do
    for dataset in ${datasets[@]};
        do
        for rep in {1..5};
            do
            jobname="${model}-${dataset}-${rep}"
            output_log="$res_path/$jobname.log"
            mkdir -p $res_path/$jobname

            squeue --format %j | grep $jobname >> /dev/null
            retVal=$?
            if [ $retVal -ne 0 ] && [ ! -f $output_log ]; then
                sbatch -J $jobname --cpus-per-task 2 --mem-per-cpu 5G -n 1 -p gpu -o $output_log -e $output_log --gres=gpu:1 --wrap "pipenv run torch --dataset $dataset --model $model --hyperparam-draws 1 --gpus -1"
            else
                echo Skipping $jobname
            fi
        done
    done
done
