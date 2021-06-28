#!/usr/bin/env bash
#
# Calculate rankings of the Shapley features. Unlike the visualisation
# scenario, this one runs faster.

HOURS=16

python -m src.torch.rank_shapley_values results/shapley/*.pkl -H ${HOURS}
