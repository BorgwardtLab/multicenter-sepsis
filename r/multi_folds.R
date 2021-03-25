#!/usr/bin/env Rscript

#BSUB -W 8:00
#BSUB -n 16
#BSUB -R rusage[mem=5000]
#BSUB -J folds[1-60]
#BSUB -o data-res/folds_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

args <- check_index(
  parse_args(job_index),
  train_src = c("mimic", "aumc"),
  feat_set = c("locf", "basic"),
  target = c("class", "hybrid", "reg"),
  split = paste("split", 0:4, sep = "_")
)

redir <- file.path(data_path("res"), paste0("folds_", jobid()))
extra <- list(test_src = c("mimic", "aumc"), predictor = "rf",
              res_dir = redir)

invisible(
  prof(do.call(fit_predict, c(args, extra)))
)
