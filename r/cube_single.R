#!/usr/bin/env Rscript

#BSUB -W 24:00
#BSUB -n 16
#BSUB -R rusage[mem=8000]
#BSUB -J model
#BSUB -o data-res/model_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

redir <- file.path(data_path("res"), paste0("model_", jobid()))
extra <- list(test_src = c("mimic", "aumc"), res_dir = redir)

invisible(
  prof(
    fit_predict("mimid", feat_set = "full", predictor = "rf", res_dir = redir)
  )
)
