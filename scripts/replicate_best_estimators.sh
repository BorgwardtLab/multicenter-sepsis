for DATASET in 'aumc' 'eicu' 'hirid' 'mimic' 'physionet2019'; do
  NAME=${DATASET}_lr
  scp ../results/${NAME}/best_estimator.pkl \
      ../results/${NAME}/cv_results.csv     \
      ../results/${NAME}/results.json       \
      bs-borgwardt02:/links/groups/borgwardt/Projects/sepsis/multicenter-sepsis/results/${NAME}_euler
done
