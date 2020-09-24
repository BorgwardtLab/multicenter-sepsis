dataset=PreprocessedDemoDataset
model=RNNModel

res_path=$1 
mkdir -p $res_path

dataset_path=$res_path/${dataset}
mkdir -p ${dataset_path}

python -m src.torch.train_model --dataset $dataset --model $model --hyperparam-draws 1 --gpus -1 --log-path $dataset_path --exp-name $model
