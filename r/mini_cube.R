#!/usr/bin/env Rscript

#BSUB -W 24:00
#BSUB -n 16
#BSUB -R rusage[mem=8000]
#BSUB -J cube[1-24]
#BSUB -o data-res/cube_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

srcs <- c("minic", "aumc", "hirid", "eicu")

args <- check_index(
  parse_args(job_index),
  train_src = srcs,
  feat_set = c("locf", "basic"),
  target = c("class", "hybrid", "reg")
)

redir <- file.path(data_path("res"), paste0("cube_", jobid()))
extra <- list(test_src = srcs, predictor = "rf",
              res_dir = redir)

invisible(
  prof(do.call(fit_predict, c(args, extra)))
)
