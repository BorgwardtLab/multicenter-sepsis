#!/usr/bin/env bash
#
# Create visualisations of the Shapley features. This looks a little bit
# stupid at first, but we first run the 'shorter' calculations before
# tackling the unfiltered scenario.

HOURS=16

echo 'Calculating bar plot and rankings...'

python -m scripts.visualize_shapley_bar results/shapley/0zcdmr2b_*.pkl \
                                        results/shapley/4ky293gb_*.pkl \
                                        results/shapley/9rqxww43_*.pkl \
                                        results/shapley/hacl0kfp_*.pkl \
                                        -H ${HOURS}

echo 'Calculating swarm plots...'

for RUN in '0zcdmr2b' '4ky293gb' '9rqxww43' 'hacl0kfp'; do
  python -m scripts.visualize_shapley_swarm results/shapley/${RUN}_*.pkl -H ${HOURS}
done
