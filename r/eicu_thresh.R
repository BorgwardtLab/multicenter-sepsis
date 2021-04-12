#!/usr/bin/env Rscript

#BSUB -W 4:00
#BSUB -n 4
#BSUB -R rusage[mem=16000]
#BSUB -J export[1-6]
#BSUB -o data-export/export_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

thresh <- check_index(
  parse_args(job_index),
  seq(0.05, 0.3, by = 0.05)
)

dir <- file.path("data-export", paste0("eicu_", sub("\\.", "-", thresh)))

export_data("eicu", dest_dir = dir, eicu_hosp_thresh = thresh)
