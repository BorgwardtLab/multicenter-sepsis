#!/usr/bin/env Rscript

#BSUB -W 24:00
#BSUB -n 16
#BSUB -R rusage[mem=4000]
#BSUB -J model[1-2]
#BSUB -o data-res/model_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

args <- check_index(
  parse_args(job_index),
  predictor = c("linear", "rf")
)

redir <- file.path(data_path("res"), paste0("model_", jobid()))
extra <- list(train_src = "aumc", res_dir = redir)

invisible(
  prof(do.call(fit_predict, c(args, extra)))
)
