dataset=IBPhysionet2019Dataset #IBMIMIC3Dataset #PreprocessedEICUDataset #PreprocessedMIMIC3Dataset
model=GRUModel # LSTMModel, RNNModel, GRUModel

res_path=$1 
mkdir -p $res_path

dataset_path=$res_path/${dataset}
mkdir -p ${dataset_path}

pipenv run train_torch --dataset $dataset --model $model --hyperparam-draws 1 --gpus -1 --log-path $dataset_path --exp-name $model
