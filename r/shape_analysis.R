
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

targs <- paste(
  rep(c("class", "reg"), each = 100),
  rep(seq_len(10), 20),
  rep(rep(seq_len(10), each = 10), 2),
  sep = "_"
)

targs <- paste(
  rep(c("class", "reg"), each = 4),
  rep(seq_len(2), 4),
  rep(rep(seq_len(2), each = 2), 2),
  sep = "_"
)

shp <- read_ress("mimic", feat_set = "locf", predictor = "rf",
                 target = targs, jobid = "shape")
shp <- shp[,
  c("target", "targ_param_1", "targ_param_2") := data.table::tstrsplit(
    target, "_", fixed = TRUE
)]

evl <- patient_eval(shp, c("target", "targ_param_1", "targ_param_2"))

attr(evl, "stats")
