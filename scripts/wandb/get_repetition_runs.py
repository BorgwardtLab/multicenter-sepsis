"""Get all repetition runs (for given list of sweeps) and start submit_evals script."""
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
    
    for run_id in run_ids:
        subprocess.run(
            [ './scripts/wandb/submit_evals.sh', run_id
            ]
        )
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweeps', type=str, nargs='+')
    parser.add_argument('--sweep_path', type=str, 
        default='sepsis/mc-sepsis/sweeps/')
    args = parser.parse_args()
    main(args)
