#!/usr/bin/env Rscript

#BSUB -W 24:00
#BSUB -n 20
#BSUB -R rusage[mem=8000]
#BSUB -J model
#BSUB -o data-res/model_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

redir <- file.path(data_path("res"), paste0("model_", jobid()))

invisible(
  prof(
    fit_predict("mimic",
      feat_set = "full", predictor = "lgbm", target = "reg", res_dir = redir
    )
  )
)
