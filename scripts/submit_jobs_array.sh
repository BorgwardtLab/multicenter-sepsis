#!/bin/bash
models=(GRUModel LSTMModel) #(GRUModel) # LSTMModel RNNModel)
models=(AttentionModel) #AttentionModel 
# models=(LSTMModel) #AttentionModel 
datasets=(PreprocessedMIMIC3Dataset PreprocessedEICUDataset PreprocessedHiridDataset PreprocessedAUMCDataset) #PreprocessedPhysionet2019Dataset 
# datasets=(PreprocessedEICUDataset) #PreprocessedPhysionet2019Dataset 
datasets2=(PreprocessedPhysionet2019Dataset)
#datasets2=()

res_path=$1
mkdir -p $res_path

first_index=1
last_index=19
parallel_tasks=6

pos_weights=(10 50 50 10) # assuming order: mimic eicu hirid aumc
pos_weights2=(50) #100

weight_decay=0 #0.001

for index in ${!datasets[*]};
    do
    dataset=${datasets[$index]}
    pos_weight=${pos_weights[$index]}
    dataset_path=$res_path/${dataset}
    mkdir -p ${dataset_path}
    for model in ${models[@]};
    do
        out_path=${dataset_path}/${model}
        jobname="${model}-${dataset}"
        squeue --format %j | grep $jobname >> /dev/null
        retVal=$?
        if [ $retVal -ne 0 ]; then
            output_log="${out_path}/%a.log"
            output_dir=${out_path}/\$SLURM_ARRAY_TASK_ID
            mkdir -p ${out_path}
            command="srun pipenv run train_torch --dataset $dataset --model $model --hyperparam-draws 1 --gpus -1 --log-path $dataset_path --exp-name $model --version \$SLURM_ARRAY_TASK_ID --pos-weight $pos_weight --weight-decay $weight_decay"
            sbatch --array $first_index-$last_index%$parallel_tasks -J $jobname --cpus-per-task 2 --mem-per-cpu 8G -n 1 -p gpu -o $output_log -e $output_log --gres=gpu:1 --wrap "if [ ! -d \"$output_dir\" ]; then echo \"$command\"; $command; fi"
        fi
    done
done

feature_set=challenge
for dataset in ${datasets2[@]};
    do
    dataset_path=$res_path/${dataset}
    mkdir -p ${dataset_path}
    for model in ${models[@]};
    do
        out_path=${dataset_path}/${model}
        jobname="${model}-${dataset}"
        squeue --format %j | grep $jobname >> /dev/null
        retVal=$?
        if [ $retVal -ne 0 ]; then
            output_log="${out_path}/%a.log"
            output_dir=${out_path}/\$SLURM_ARRAY_TASK_ID
            mkdir -p ${out_path}
            command="srun pipenv run train_torch --dataset $dataset --feature-set $feature_set --model $model --hyperparam-draws 1 --gpus -1 --log-path $dataset_path --exp-name $model --version \$SLURM_ARRAY_TASK_ID --pos-weight $pos_weights2 --weight-decay $weight_decay"
            sbatch --array $first_index-$last_index%$parallel_tasks -J $jobname --cpus-per-task 2 --mem-per-cpu 8G -n 1 -p gpu -o $output_log -e $output_log --gres=gpu:1 --wrap "if [ ! -d \"$output_dir\" ]; then echo \"$command\"; $command; fi"
        fi
    done
done
