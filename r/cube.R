#!/usr/bin/env Rscript

#BSUB -W 24:00
#BSUB -n 8
#BSUB -R rusage[mem=1000]
#BSUB -J model[1-2]
#BSUB -o data-res/model_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

args <- check_index(
  parse_args(job_index),
  train_src = "mimic",
  feat_set = "full",
  predictor = c("linear", "rf"),
  target = "reg"
)

dir <- file.path(data_path("res"), paste0("model_", jobid()))

prof(
  do.call(fit_predict, c(args, list(res_dir = dir)))
)
