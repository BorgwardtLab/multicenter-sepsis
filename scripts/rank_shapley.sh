#!/usr/bin/env bash
#
# Calculate rankings of the Shapley features. Unlike the visualisation
# scenario, this one runs faster.

python -m src.torch.rank_shapley_values results/shapley/*.pkl -c
for HOURS in 1 2 4 8 16; do
  python -m src.torch.rank_shapley_values results/shapley/*.pkl -H ${HOURS} -c
done
