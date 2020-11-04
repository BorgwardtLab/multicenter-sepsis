# run on slurm cluster:
module use /usr/local/borgwardt/modulefiles
module load python/3.7.7
srun -p cpu --cpus-per-task 4 --mem-per-cpu 4G tensorboard --logdir .

# then connect to cluster via tunnel:
# ssh -L 6006:localhost:6006 bs-hpsvr08
