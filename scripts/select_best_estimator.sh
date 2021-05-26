#!/usr/bin/env bash
#
# Selects the best estimator from a set of iterations and replicates it
# by renaming all corresponding files (nothing will be overwitten).

INPUT_DIR=$1
N_FILES=50

BEST_ITERATION=`find $1 -maxdepth 1 -name "*_iteration_*.json" -type f \
  | head -n ${N_FILES}                                                 \
  | xargs jq -r '(.val_neg_log_loss|tostring) + " " + input_filename'  \
  | sort -nr                                                           \
  | head -n 1                                                          \
  | cut -f 2 -d ' '                                                    \
  | sed -n -e 's/^.*_\(iteration_[[:digit:]]\+\).json/\1/p'`

echo "Found best iteration:" `echo $BEST_ITERATION | sed -n -e 's/iteration_//p'`

for FILE in $1/*_${BEST_ITERATION}.*; do
  NEW_FILENAME=`echo $FILE | sed -n -e 's/_iteration_[[:digit:]]\+//p'`
  echo "Creating $NEW_FILENAME..."
  cp -n $FILE $NEW_FILENAME
done
