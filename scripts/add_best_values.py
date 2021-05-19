"""Add online_val/best_loss to a sweep or run summary."""
import argparse
import wandb
api = wandb.Api()


def main(run_or_sweep: str):
    """Add online_val/best_loss to a sweep or run summary."""
    if 'sweep' in run_or_sweep:
        runs = api.sweep(run_or_sweep).runs
    else:
        runs = [api.run(run_or_sweep)]

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
    parser.add_argument('run_or_sweep', type=str)

    args = parser.parse_args()
    main(args.run_or_sweep)
