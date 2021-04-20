#!/usr/bin/env bash
#
# Submit jobs for the 'shallow' models. Only CPU and RAM requirements
# are provided because GPU resources are not exploited.

# Path to store results. Will be added to main command. Has to be
# provided by caller of this script.
RESULTS_PATH=${1}

# Main configuration for *what* all jobs will run.
N_ITER=20
FEATURE_SET=middle
COST=5

# Main configuration for *how* all jobs will be run on the cluster.
# It is easiest to specify this here because we can modify it later
# on for individual classifiers.
N_CORES=1
MEM_PER_CORE=524288
RUNTIME=23:59

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -N '"-n ${N_CORES}"' -o "sklearn_%J.out" -R "rusage[mem='${MEM_PER_CORE}']"'
fi

# Evaluates its first argument either by submitting a job, or by
# executing the command without parallel processing. Notice that
# `$2` needs to be provided to specify runtime requirements.
run() {
  if [ -z "$BSUB" ]; then
    eval "$1";
  else
    eval "${BSUB} -W $2 $1";
  fi
}

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python -m src.sklearn.main --result_path ${RESULTS_PATH} --n_iter_search=${N_ITER} --feature_set ${FEATURE_SET} --cost ${COST} "

for data in aumc eicu hirid mimic; do
  for method in lr lgbm; do

    if [ ${method} = "lgbm" ]; then
      TIME=119:59
    else
      TIME=$RUNTIME
    fi

    run "${MAIN} --cv_n_jobs=${N_CORES} --dataset=${data} --method=${method}" $TIME
  done
done
