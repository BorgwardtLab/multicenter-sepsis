"""Get all repetition runs (for given list of sweeps) and start eval_torch.sh script."""
import argparse
import wandb
import os
import subprocess 
api = wandb.Api()

def main(args):
    """Add online_val/best_loss to a sweep or run summary."""
    base_path = args.sweep_path
    sweeps = [os.path.join(base_path, sweep) for sweep in args.sweeps] 
    run_ids = []  
    for sweep in sweeps: 
        runs = api.sweep(sweep).runs
        for run in runs:
            run_ids.append(run.id)

    #select script to run:
    if args.cohorts:
        script = 'eval_cohort_torch.sh'
    else:
        script = 'eval_torch.sh'
    cmd = os.path.join('./revisions/eval_drop_percentiles',script) 

    for run_id in run_ids:
        subprocess.run(
            [ cmd, run_id
            ]
        )
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweeps', type=str, nargs='+')
    parser.add_argument('--sweep_path', type=str, 
        default='sepsis/mc-sepsis/sweeps/')
    parser.add_argument('--cohorts', action='store_true',
        help='flag if model should be evaluated on sub cohorts', 
        default=False)
    args = parser.parse_args()
    main(args)