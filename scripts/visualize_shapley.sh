#!/usr/bin/env bash
#
# Create visualisations of the Shapley features.

for RUN in 'sfxdno1a' 'sz1aa1n9' 'm6fsnuk9' 'hacl0kfp' 'pr9pa8oa' 'f905hzcm' '9rqxww43' 'nwjs5ahk' 'vzitu3r2' 'j76ft4wm' 'ypj1pfcq' 'r6otacys' '0zcdmr2b' 'gjtf48im' 'pa6c95qv' 'xd0mpktn' '4ky293gb' 'hhjb0oz0' 'vx8vbt08' 'lvrvcwuc'; do
  python -m src.torch.visualize_global_shapley_values results/shapley/${RUN}_*.pkl -c
done
