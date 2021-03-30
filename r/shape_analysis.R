
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

targs <- paste(
  rep(c("class", "reg"), each = 25),
  rep(seq.int(1, 10, 2), 10),
  rep(rep(seq.int(1, 10, 2), each = 5), 2),
  sep = "_"
)

shp <- read_ress("mimic", feat_set = "locf", predictor = "rf",
                 target = targs, split = paste0("split_", 0:5),
                 jobid = "shape")
shp <- shp[,
  c("target", "targ_param_1", "targ_param_2") := data.table::tstrsplit(
    target, "_", fixed = TRUE
)]

evl <- patient_eval(shp, c("target", "targ_param_1", "targ_param_2", "split"))

attr(evl, "stats")
