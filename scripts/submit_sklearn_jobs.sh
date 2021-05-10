#!/usr/bin/env bash
#
# Submit jobs for the 'shallow' models. Only CPU and RAM requirements
# are provided because GPU resources are not exploited.

# Path to store results. Will be added to main command. Has to be
# provided by caller of this script.
RESULTS_PATH=${1}

# Main configuration for *how* all jobs will be run on the cluster.
# It is easiest to specify this here because we can modify it later
# on for individual classifiers.
N_CORES=1
MEM_PER_CORE=131072
RUNTIME=119:59
FEATURE_SET=middle
COST=5

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

run "poetry run python -m src.sklearn.main --result_path ${RESULTS_PATH} --feature_set ${FEATURE_SET} --cost ${COST} --cv_n_jobs=20 --n_iter_search=50 --method lr --dataset aumc " $RUNTIME
run "poetry run python -m src.sklearn.main --result_path ${RESULTS_PATH} --feature_set ${FEATURE_SET} --cost ${COST} --cv_n_jobs=20 --n_iter_search=50 --method lr --dataset eicu " $RUNTIME
run "poetry run python -m src.sklearn.main --result_path ${RESULTS_PATH} --feature_set ${FEATURE_SET} --cost ${COST} --cv_n_jobs=20 --n_iter_search=50 --method lr --dataset hirid " $RUNTIME
run "poetry run python -m src.sklearn.main --result_path ${RESULTS_PATH} --feature_set ${FEATURE_SET} --cost ${COST} --cv_n_jobs=20 --n_iter_search=50 --method lr --dataset mimic " $RUNTIME
