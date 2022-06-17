"""Training routines for models."""
from argparse import ArgumentParser, Namespace
import json
from collections import defaultdict
from functools import partial
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
import tempfile
import pandas as pd
from sklearn.metrics import auc 

sys.path.append(os.getcwd()) # hack for executing module as script (for wandb)

import src.torch.models
import src.torch.datasets
from src.torch.datasets import CombinedDataset
from src.torch.torch_utils import JsonEncoder, TbWithBestValueLogger
from src.torch.eval_model import online_eval, compute_md5hash, device

# Pat eval:
from scripts.plots.plot_scatterplots import get_coordinates
from src.evaluation.patient_evaluation import main as pat_eval

wandb_api = wandb.Api()

def extract_model_information(run_path, tmp):
    run = wandb_api.run(run_path)
    run_info = run.config
    checkpoint_path = None
    for f in run.files():
        if f.name.endswith('.ckpt'):
            file_desc = f.download(tmp)
            checkpoint_path = file_desc.name
            file_desc.close()
    if checkpoint_path is None:
        raise RuntimeError(
            f'Run "{run_path}" does not have a stored checkpoint file.')

    model_checksum = compute_md5hash(checkpoint_path)
    dataset_kwargs = {}
    for key in run_info.keys():
        if 'dataset_kwargs' in key:
            new_key = key.split('/')[-1]
            dataset_kwargs[new_key] = run_info[key]
    return run, {
        "model": run_info['model'],
        "run_id": run_path,
        "model_path": checkpoint_path,
        "model_checksum": model_checksum,
        "model_params": run_info,
        "dataset_train": run_info['dataset'],
        "task": run_info['task'],
        "label_propagation": run_info['label_propagation'],
        "rep": run_info['rep'],
        "dataset_kwargs": dataset_kwargs
    }

def namespace_without_none_and_arch(namespace):
    """ We drop Nones and architectural hparams, 
        lastly for not creating errors when loading
        models from checkpoint
     """
    new_namespace = Namespace()
    for key, value in vars(namespace).items():
        if value is not None and type(value) != type:
            if hasattr(value, '__len__'):
                if len(value) == 0:
                    continue
            if key in ['batch_size', 'd_model']:
                continue
            setattr(new_namespace, key, value)
    return new_namespace

def main(hparams, model_cls):
    """Main function train model."""
    run_id = os.path.join('sepsis/mc-sepsis', hparams.wandb_run) 
    with tempfile.TemporaryDirectory() as tmp:
        # Download checkpoint to temporary directory
        run, out = extract_model_information(run_id, tmp)
        # here we don't need a manual output file (wandb suffices)
        out['dataset_eval'] = ','.join(hparams.dataset)
        # out['split'] = split

        if len(hparams.dataset) > 1:
            raise RuntimeError('This script is not intended for combined datasets!')
        dataset_cls = getattr(src.torch.datasets, hparams.dataset[0])

        # check that model from argparse agrees with model class fro wandb id
        assert hparams.model == out['model']
        model_cls = getattr(src.torch.models, out['model'])
        if out['rep'] != hparams.rep:
            print(f'Pretrained model was trained on rep {out["rep"]}, current hparams.rep = {hparams.rep}')
    
        model = model_cls.load_from_checkpoint(
            out['model_path'],
            ### dataset=hparams.dataset[0],
            ### dataset_kwargs = hparams.dataset_kwargs
            **vars(namespace_without_none_and_arch(hparams)) # these are incompatible with the loaded model
        )
    model.to(device)
 
    #model = model_cls(**vars(namespace_without_none(hparams)))

    # Wandb logger:
   # Loggers and callbacks
    job_name = f'{hparams.model}_{",".join(hparams.dataset)}'
    # check if slurm array job id is available:
    job_id = os.getenv('SLURM_ARRAY_TASK_ID')
    if job_id is not None:
        job_name += f'_{job_id}' 

    username = os.environ['USER']
    assert username is not None, RuntimeError('Require valid username')
    save_dir = f'/local0/scratch/{username}/'
    os.makedirs(save_dir, exist_ok=True)

    tags = [hparams.model, hparams.task]
    wandb_logger = WandbLogger(
        name=job_name,
        project="mc-sepsis",
        entity="sepsis",
        log_model=True,
        tags=[
            hparams.model,
            ','.join(hparams.dataset),
            hparams.task
        ],
        save_dir=save_dir,
        settings=wandb.Settings(start_method="fork")
    )

    monitor_score = hparams.monitor
    monitor_mode = hparams.monitor_mode

    model_checkpoint_cb = ModelCheckpoint(
        monitor=monitor_score,
        mode=monitor_mode,
        save_top_k=1,
        dirpath=wandb_logger.experiment.dir #checkpoint_dir
    )
    early_stopping_cb = EarlyStopping(
        monitor=monitor_score, patience=20, mode=monitor_mode, strict=True,
        verbose=1)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer(
        callbacks=[early_stopping_cb],
        checkpoint_callback=model_checkpoint_cb,
        max_epochs=hparams.max_epochs,
        logger=wandb_logger,
        gpus=hparams.gpus,
        #profiler='advanced'
    )
    trainer.fit(model)
    #trainer.logger.save()
    print('Loading model with', monitor_mode, monitor_score)
    print(model_checkpoint_cb.best_model_path)
    loaded_model = model_cls.load_from_checkpoint(
        checkpoint_path=model_checkpoint_cb.best_model_path)
    results = trainer.test(loaded_model)
    # results is a single-element list of a dict:
    results = results[0]
    #trainer.test(loaded_model)
    #trainer.logger.save()
    #all_metrics = {**trainer.logger.last, **trainer.logger.best}
    #all_metrics = {n: v for n, v in all_metrics.items() if '/' in n}
    #prefixes = {name.split('/')[0] for name in all_metrics.keys()}
    #results = defaultdict(dict)
    #for name, value in all_metrics.items():
    #    for prefix in prefixes:
    #        if name.startswith(prefix):
    #            results[prefix][name.split('/')[1]] = value

    val_dataset_cls = partial(
        CombinedDataset,
        datasets=(getattr(src.torch.datasets, d) for d in hparams.dataset)
    )
    
    masked_result = online_eval(
        loaded_model,
        val_dataset_cls,
        'validation',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        online_split=True,
        **hparams.dataset_kwargs
    )
    # we need metadata from out dict:
    out.update(masked_result) 
    
    
    pat_results = pat_eval(
        input_file=out, 
        from_dict=True, 
        used_measures=['pat_eval']
    )
    df = pd.DataFrame(pat_results)
    # pat-level eval:
    pat_metrics = {} 
    # scatter
    recalls = [0.8, 0.9, 0.95]
    for recall in recalls:
        x,x_name, y, y_name = get_coordinates(df, recall, 'pat')
        for val, name in zip([x,y], [x_name, y_name]): 
            #wandb_logger.experiment.log({name + f'_at_recall_{recall}' : val})
            pat_metrics[name + f'_at_recall_{recall}'] = val
    tpr = df['pat_recall'].values
    fpr = 1 - df['pat_specificity'].values
    AUROC = auc(fpr, tpr)
    pat_metrics['pat_auroc'] = AUROC
    for key, value in pat_metrics.items():
        wandb_logger.experiment.log({key: value})
 
    masked_result = { 'masked_validation_'+key: value for key, value in masked_result.items()
        if key not in ['labels', 'predictions', 'ids', 'times', 'scores', 'targets']
    }
    for key, value in masked_result.items():
        wandb_logger.experiment.log({key: value})
    #masked_result = { key: value for key, value in masked_result.items()
    #    if key not in ['labels', 'predictions', 'ids', 'times', 'scores', 'targets']
    #}
    results.update(pat_metrics)
    results.update(masked_result)

    #with open(os.path.join(logger.log_dir, 'result.json'), 'w') as f:
    #    json.dump(results, f, cls=JsonEncoder)
    for name, value in results.items():
        if name in ['labels', 'predictions']:
            continue
        wandb_logger.experiment.summary[name] = value
    #wandb_logger.experiment.summary.update(masked_result)

    ##print('MASKED TEST RESULTS')
    ##print({
    ##    key: value for key, value in results['validation_masked'].items()
    ##    if key not in ['labels', 'predictions']
    ##})

    ## Filter out parts of hparams which belong to Hyperargparse
    #config = {
    #    key: value
    #    for key, value in vars(hparams).items()
    #    if not callable(value)
    #}
    #with open(os.path.join(logger.log_dir, 'config.json'), 'w') as f:
    #    json.dump(config, f, cls=JsonEncoder)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--wandb_run', type=str)
    parser.add_argument('--log_path', default='logs')
    parser.add_argument('--exp_name', default='finetune_torch_model')
    parser.add_argument('--version', default=None, type=str)
    parser.add_argument('--model', choices=src.torch.models.__all__, type=str,
                        default='AttentionModel')
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--hyperparam-draws', default=0, type=int)
    parser.add_argument('--monitor', type=str,
                        default='online_val/loss')
    parser.add_argument('--monitor_mode', type=str, choices=['max', 'min'],
                        default='min')
    parser.add_argument('--indicators', type=bool,
                        default=False)
    parser.add_argument('--task', type=str,
                        default='classification')
    parser.add_argument('--cost', type=int, default=5,
                        help='cost parameter for lambda')
    parser.add_argument('--rep', type=int, default=0,
                        help='Repetition fold of splits [0,..,4]')
    parser.add_argument('--dummy_repetition', type=int, default=1,
                        help='inactive argument, used for debugging (enabling grid repetitions)')
    parser.add_argument('--only_physionet_features', type=bool, default=False,
                        help='boolean indicator if physionet variable set should be used')
    parser.add_argument('--feature_set', default=None,
                        help='which feature set should be used: [middle, small,..]')
    parser.add_argument('--finetuning', type=bool,
                        default=True, help='flag to indicate finetuning on external dataset')
    parser.add_argument('--finetuning_size', type=float,
                        default=1., help='ratio of fine-tuning split to use')


    # figure out which model to use
    temp_args = parser.parse_known_args()[0]

    if temp_args.monitor.endswith('loss') and temp_args.monitor_mode == 'max':
        print(
            'It looks like you are trying to run early stopping on a loss '
            'using the wrong monitor mode (max).')
        print('Exiting...')
        import sys
        sys.exit(1)

    # let the model add what it wants
    model_cls = getattr(src.torch.models, temp_args.model)

    parser = model_cls.add_model_specific_args(parser)
    hparams = parser.parse_args()
    if isinstance(hparams.dataset, (list, tuple)) and len(hparams.dataset) == 1:
        hparams.dataset = hparams.dataset[0].split(',')

    hparams.dataset_kwargs = {
        'cost': hparams.cost,
        'fold': hparams.rep, #`rep` naming to conform with shallow models                                        
        'only_physionet_features': hparams.only_physionet_features,
        'finetuning': hparams.finetuning,
        'finetuning_size': hparams.finetuning_size
    }
    if hparams.feature_set is not None:
        hparams.dataset_kwargs['feature_set'] = hparams.feature_set

    #if hparams.hyperparam_draws > 0:
    #    for hyperparam_draw in hparams.trials(hparams.hyperparam_draws):
    #        print(hyperparam_draw)
    #        hyperparam_draw = Namespace(**hyperparam_draw.__getstate__())
    #        main(hyperparam_draw, model_cls)
    #else:
    # Need to do this in order to allow pickling
    hparams = Namespace(**vars(hparams))
    main(hparams, model_cls)
