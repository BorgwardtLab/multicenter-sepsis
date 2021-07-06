#!/usr/bin/env bash
#
# Create visualisations of the Shapley features. This looks a little bit
# stupid at first, but we first run the 'shorter' calculations before
# tackling the unfiltered scenario.

HOURS=16

for FEATURES in '' '-i'; do
  echo 'Calculating bar plot and rankings...'

  python -m scripts.visualize_shapley_bar results/shapley/j76ft4wm_*.pkl \
                                          results/shapley/vx8vbt08_*.pkl \
                                          results/shapley/gjtf48im_*.pkl \
                                          results/shapley/pr9pa8oa_*.pkl \
                                          ${FEATURES} -H ${HOURS}

  echo 'Calculating swarm plots...'

  for RUN in 'j76ft4wm' 'vx8vbt08' 'gjtf48im' 'pr9pa8oa'; do
    python -m scripts.visualize_shapley_swarm results/shapley/${RUN}_*.pkl ${FEATURES} -H ${HOURS}
  done
done
