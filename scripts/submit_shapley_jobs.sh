#!/usr/bin/env bash
#
# Submits jobs for calculating Shapley features.

for RUN in 'ea9w4y5t' 'sfxdno1a' 'sz1aa1n9' 'm6fsnuk9' 'hacl0kfp' 'pr9pa8oa' 'f905hzcm' '9rqxww43' 'nwjs5ahk' 'vzitu3r2' 'j76ft4wm' 'ypj1pfcq' 'r6otacys' '0zcdmr2b' 'gjtf48im' 'pa6c95qv' 'xd0mpktn' '4ky293gb' 'hhjb0oz0' 'vx8vbt08' 'lvrvcwuc'; do
  for ITERATION in `seq 5`; do
    echo $RUN-$ITERATION...
    sbatch -p gpu                                         \
      --gres=gpu:1                                        \
      --cpus-per-task 2                                   \
      --mem-per-cpu 8G                                    \
      --job-name "mcsepsis-shapley-$RUN-$ITERATION"       \
      --output "mcsepsis-shapley-$RUN-$ITERATION-%j.out"  \
      --wrap "python -m src.torch.shap_analysis_wandb sepsis/mc-sepsis/runs/$RUN --n-samples 500 --n-samples-background 200 --output results/shapley/${RUN}_${ITERATION}.pkl"
  done
done
