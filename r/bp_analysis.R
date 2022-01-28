
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

dat <- read_parquet("mimic",
  cols = c("stay_id", "stay_time", "sbp_raw", "map_raw", "dbp_raw",
           "ventialtion", "vasopressors", "onset_delta")
)

dat <- dat[, c("ventialtion", "vasopressors") := lapply(.SD, is_true),
           .SDcols = c("ventialtion", "vasopressors")]

dat[, lapply(.SD, sd, na.rm = TRUE),
    .SDcols = c("sbp_raw", "map_raw", "dbp_raw"), by = "vasopressors"]

dat[is_true(onset_delta >= -24 & onset_delta <= 0),
    lapply(.SD, sd, na.rm = TRUE),
    .SDcols = c("sbp_raw", "map_raw", "dbp_raw")]

dat[is.na(onset_delta) & stay_time >= 0 & stay_time <= 24,
    lapply(.SD, sd, na.rm = TRUE),
    .SDcols = c("sbp_raw", "map_raw", "dbp_raw")]
