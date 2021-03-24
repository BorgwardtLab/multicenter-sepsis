#!/usr/bin/env Rscript

#BSUB -W 8:00
#BSUB -n 1
#BSUB -R rusage[mem=32000]
#BSUB -J export[1-5]
#BSUB -o data-export/export_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

args <- parse_args(job_index, out_dir)

src <- check_index(args, list(
  c("mimic_demo", "eicu_demo", "physionet2019"),
  "mimic", "eicu", "hirid", "aumc"
))

dir <- check_dir(args)

for (x in src) {
  msg("exporting data for {x}\n")
  export_data(x, dest_dir = dir)
}
