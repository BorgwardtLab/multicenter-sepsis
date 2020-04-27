
sepsis3_score <- function(source, pids = NULL) {

  sofa_score <- sofa(source, id_type = "icustay", patient_ids = pids)

  if (grepl("eicu", source)) {

    susp_infec <- si(source, abx_min_count = 2L, positive_cultures = TRUE,
                     id_type = "icustay", patient_ids = pids, mode = "or")

  } else if (identical(source, "hirid")) {

    susp_infec <- si(source, abx_min_count = 2L, id_type = "icustay",
                     patient_ids = pids, mode = "or")

  } else {

    susp_infec <- si(source, id_type = "icustay", patient_ids = pids)
  }

  sepsis_3(sofa_score, susp_infec)
}

dump_dataset <- function(source = "mimic_demo", dir = tempdir()) {

  if (identical(source, "challenge")) {

    data_dir <- file.path(dir, "training_setB")

    if (!dir.exists(data_dir)) {
      stop("need directory ", data_dir, " to continue")
    }

    dat <- read_psv(data_dir, col_spec = challenge_spec, id_col = "ID")

    dat <- dat[, Gender := data.table::fifelse(Gender == 0L, "Female", "Male")]
    dat <- dat[, O2Sat := rowMeans(.SD, na.rm=TRUE),
               .SDcols = c("O2Sat", "SaO2")]
    dat <- dat[, ICULOS := as.difftime(ICULOS - 1, units = "hours")]
    dat <- as_ts_tbl(dat, "ID")

    sep <- dat[(SepsisLabel), list(ICULOS = min(ICULOS) + 6), by = "ID"]
    sep <- rename_cols(sep, "sep3_time", "ICULOS")

    dat <- rm_cols(dat, "SepsisLabel")

  } else {

    dat <- load_dictionary(source, challenge_map[[2L]], id_type = "icustay")
    ids <- unique(dat[age >= 14, id(dat), with = FALSE])
    dat <- merge(dat, ids, all.y = TRUE)

    dat <- rename_cols(dat, c("ICULOS", challenge_map[[1L]]),
                            c(index(dat), challenge_map[[2L]]),
                       skip_absent = TRUE)

    win <- stay_windows(source, id_type = "icustay", win_type = "icustay",
                        in_time = "intime", out_time = "outtime")

    dat  <- dat[, c("join_time") := list(get(index(dat)))]

    join <- c(paste(id(dat), "==", id(win)), "join_time <= outtime")
    dat <- dat[win, on = join]
    dat <- rm_cols(dat, c("join_time", "intime"))

    sep <- sepsis3_score(source, ids)
    sep  <- sep[, c("join_time") := list(get(index(sep)))]

    join <- c(paste(id(sep), "==", id(win)), "join_time >= intime",
                                             "join_time <= outtime")
    sep <- sep[win, on = join]
    sep <- rm_cols(sep, data_cols(sep))
  }

  sep <- data.table::set(sep, j = "SepsisLabel", value = 1L)

  res <- merge(dat, sep, all.x = TRUE)
  res <- res[, c("SepsisLabel") := data.table::nafill(SepsisLabel, "locf"),
             by = c(id(res))]
  res <- res[, c("SepsisLabel") := data.table::nafill(SepsisLabel, fill = 0L)]

  res <- rm_cols(res, setdiff(data_cols(res),
                              c(challenge_map[[1L]], "SepsisLabel")))

  miss_cols <- setdiff(challenge_map[[1L]], data_cols(res))

  if (length(miss_cols)) {
    res <- data.table::set(res, j = miss_cols, value = NA_real_)
  }

  res <- data.table::setcolorder(res, c(meta_cols(res), challenge_map[[1L]]))

  dir <- file.path(dir, source)

  if (dir.exists(dir)) unlink(dir, recursive = TRUE)

  dir.create(dir, recursive = TRUE)

  write_psv(res, dir, na_rows = TRUE)
}
