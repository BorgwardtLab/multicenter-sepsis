#!/usr/bin/env Rscript

#BSUB -W 8:00
#BSUB -n 8
#BSUB -R rusage[mem=4000]
#BSUB -J export
#BSUB -o results/export_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

args <- parse_args(src_opt, out_dir)

src <- check_src(args, "challenge")
dir <- check_dir(args)

data.table::setDTthreads(n_cores())

msg("\n\nUsing {data.table::getDTthreads()} OMP thread{?s}\n\n")

for (x in src) {
  msg("\n\nexporting data for {x}\n\n")
  export_data(x, dest_dir = dir)
}
