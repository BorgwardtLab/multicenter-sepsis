#!/usr/bin/env Rscript

#BSUB -W 1:00
#BSUB -n 1
#BSUB -R rusage[mem=24000]
#BSUB -J cohorts
#BSUB -o results/cohorts_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

args <- parse_args(src_opt, out_dir)

src <- check_src(args, "challenge")
dir <- check_dir(args)

for (x in src) {
  msg("\n\nexporting data for {x}\n\n")
  export_data(x, dest_dir = dir)
}
