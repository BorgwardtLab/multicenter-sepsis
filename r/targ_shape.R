#!/usr/bin/env Rscript

#BSUB -W 4:00
#BSUB -n 16
#BSUB -R rusage[mem=4000]
#BSUB -J shape[1-400]%50
#BSUB -o data-res/shape_%J.out

invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

args <- check_index(
  parse_args(job_index),
  train_src = c("mimic", "aumc"),
  target = c("class", "reg"),
  targ_param_1 = seq_len(10L),
  targ_param_2 = seq_len(10L)
)

redir <- file.path(data_path("res"), paste0("shape_", jobid()))
extra <- list(test_src = c("mimic", "aumc"), predictor = "rf",
              feat_set = "locf", res_dir = redir)

invisible(
  prof(do.call(fit_predict, c(args, extra)))
)

fit_predict(target = "reg", targ_param_1 = 10, targ_param_2 = 1)
