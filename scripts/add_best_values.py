"""Add online_val/best_loss to a sweep or run summary."""
import argparse
import os
import wandb
api = wandb.Api()


def main(args):
    """Add online_val/best_loss to a sweep or run summary."""
    
    base_path = args.sweep_path
    sweeps = [os.path.join(base_path, sweep) for sweep in args.sweep]
    runs = []
    for sweep in sweeps: 
        r_ids = api.sweep(sweep).runs
        runs.extend(r_ids)
        #for run in r_ids:
        #    runs.append(run.id)

    from IPython import embed; embed()
 
    for run in runs:
        print(f'Updating run {run}')
        try:
            best = run.history(keys=['online_val/loss']).min()
            best_step, best_loss = best['_step'], best['online_val/loss']
            run.summary['online_val/best_step'] = best_step
            run.summary['online_val/best_loss'] = best_loss
            run.update()
        except:
            print(f'exception in run {run}, skipping ..')
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep', type=str, nargs='+', 
        help='for sweep, only the id is necessary')
    parser.add_argument('--sweep_path', type=str, 
        default='sepsis/mc-sepsis/sweeps/')
    args = parser.parse_args()
    main(args)
