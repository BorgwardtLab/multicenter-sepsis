#!/usr/bin/env bash
#
# Submit jobs for the 'shallow' models. Only CPU and RAM requirements
# are provided because GPU resources are not exploited.

# Path to store results. Will be added to main command. Has to be
# provided by caller of this script.
RESULTS_PATH=${1}

# Main configuration for *what* all jobs will run.
N_ITER=50
FEATURE_SET=middle
COST=5

# Main configuration for *how* all jobs will be run on the cluster. 
N_CORES=32
MEM_PER_CORE=8192

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -N '"-n ${N_CORES}"' -W 23:59 -o "sklearn_%J.out" -R "rusage[mem='${MEM_PER_CORE}']"'
fi

# Evaluates its first argument either by submitting a job, or by
# executing the command without parallel processing.
run() {
  if [ -z "$BSUB" ]; then
    eval "$1";
  else
    eval "${BSUB} $1";
  fi
}

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python -m src.sklearn.main --result_path ${RESULTS_PATH} --n_iter_search=${N_ITER} --feature_set ${FEATURE_SET} --cost ${COST} "

for data in aumc eicu hirid eicu; do
  for method in lr lgbm; do
    run "${MAIN} --cv_n_jobs=32 --dataset=${data} --method=${method}"
  done
done
