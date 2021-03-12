#!/usr/bin/env Rscript

#BSUB -W 1:00
#BSUB -n 2
#BSUB -R rusage[mem=1000]
#BSUB -J model[1-2]
#BSUB -o data-res/model_%J/lsf.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

args <- check_index(
  parse_args(job_index),
  train_src = "mimic_demo",
  feat_set = "basic",
  predictor = c("linear", "rf"),
  target = "reg",
  res_dir = file.path(data_path("res"), paste0("model_", jobname()))
)

prof(do.call(fit_predict, args))
