#!/usr/bin/env Rscript

#BSUB -W 8:00
#BSUB -n 16
#BSUB -R rusage[mem=4000]
#BSUB -J cube
#BSUB -o data-res/cube_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

redir <- file.path(data_path("res"), paste0("cube_", jobid()))

invisible(prof(fit_predict("aumc", res_dir = redir)))
