#!/usr/bin/env Rscript

#BSUB -W 24:00
#BSUB -n 20
#BSUB -R rusage[mem=8000]
#BSUB -J cube
#BSUB -o data-res/cube_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

redir <- file.path(data_path("res"), paste0("cube_", jobid()))

invisible(
  prof(
    fit_predict("mimic",
      feat_set = "full", predictor = "lgbm", target = "reg", res_dir = redir
    )
  )
)
