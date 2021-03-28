
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

cub <- read_ress("aumc", feat_set = c("locf", "basic", "sig", "wav", "full"),
                 predictor = "rf", target = "class", jobid = "cube")
cub <- patient_eval(cub, "feat_set")

patient_plot(cub, mod_col = "feat_set")
