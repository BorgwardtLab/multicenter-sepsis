
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

fld <- read_ress("aumc", feat_set = c("locf", "basic"), predictor = "rf",
                 target = "class", split = paste0("split_", 0:5),
                 jobid = "folds")
fld <- patient_eval(fld, c("feat_set", "split"))

patient_plot(fld, mod_col = "feat_set", rep_col = "split")
