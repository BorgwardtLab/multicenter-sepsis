
sepsis3_crit <- function(source, pids = NULL) {

  sofa <- load_concepts("sofa", source, id_type = "icustay",
                        patient_ids = pids)

  if (grepl("eicu", source)) {

    si <- load_concepts("susp_inf", source, abx_min_count = 2L,
                        positive_cultures = TRUE, id_type = "icustay",
                        patient_ids = pids, si_mode = "or")

  } else if (identical(source, "hirid")) {

    si <- load_concepts("susp_inf", source, abx_min_count = 2L,
                        id_type = "icustay", patient_ids = pids,
                        si_mode = "or")

  } else {

    si <- load_concepts("susp_inf", source, id_type = "icustay",
                        patient_ids = pids)
  }

  sep3(sofa, si)
}

cohort <- function(source, min_age = 14) {

  res <- load_concepts("age", source, id_type = "icustay")
  res <- res[age > min_age, ]

  if (grepl("eicu", source)) {
    hosp <- load_id("patient", source, cols = c(id_var(res), "hospitalid"))
    hosp <- hosp[hospitalid %in% eicu_hospitals, ]
    res <- merge(res, hosp)
  }

  id_col(res)
}

dump_dataset <- function(source = "mimic_demo", dir = tempdir()) {

  if (identical(source, "challenge")) {

    data_dir <- file.path(dir, "physionet2019", "data", "training_setB")

    if (!dir.exists(data_dir)) {
      stop("need directory ", data_dir, " to continue")
    }

    feat_map <- concepts[!is.na(concepts[["callenge"]]), ]
    feats <- do.call(readr::cols,
      setNames(feat_map[["col_spec"]], feat_map[["callenge"]])
    )

    dat <- read_psv(data_dir, col_spec = feats, id_var = "stay_id")

    dat <- dat[, Gender := data.table::fifelse(Gender == 0L, "Female", "Male")]
    dat <- dat[, O2Sat := rowMeans(.SD, na.rm=TRUE),
               .SDcols = c("O2Sat", "SaO2")]
    dat <- dat[, stay_time := as.difftime(ICULOS - 1, units = "hours")]
    dat <- as_ts_tbl(dat, "stay_id", "stay_time")

    sep <- dat[
      (SepsisLabel), list(stay_time = min(stay_time) + 6, sep3 = TRUE),
      by = "stay_id"
    ]

    dat <- rm_cols(dat, c("SaO2", "ICULOS", "SepsisLabel"), by_ref = TRUE)

    feats <- setNames(feat_map[["concept"]], feat_map[["callenge"]])
    feats <- feats[data_vars(dat)]

    dat <- rename_cols(dat, feats, names(feats))

  } else {

    pid <- cohort(source)

    feats <- concepts[["concept"]]
    feats <- feats[!is.na(feats)]

    dat <- load_concepts(feats, source, id_type = "icustay",
                         patient_ids = pid)
    dat <- rename_cols(dat, c("stay_id", "stay_time"), meta_vars(dat))

    win <- stay_windows(source, id_type = "icustay", win_type = "icustay",
                        in_time = "intime", out_time = "outtime")

    dat  <- dat[, c("join_time") := list(get("stay_time"))]

    join <- c(paste("stay_id ==", id_vars(win)), "join_time <= outtime")
    dat <- dat[win, on = join]
    dat <- rm_cols(dat, c("join_time", "intime"), by_ref = TRUE)

    sep <- sepsis3_crit(source, pid)
    sep  <- sep[, c("join_time") := list(get(index_var(sep)))]

    join <- c(paste(id_vars(sep), "==", id_vars(win)), "join_time >= intime",
                                                       "join_time <= outtime")
    new <- sep[win, on = join, nomatch = NULL]
    flt <- setdiff(id_col(sep), id_col(new))
    sep <- rm_cols(new, setdiff(data_vars(new), "sep3"))

    if (length(flt) > 0L) {
      dat <- dat[!get(id_var(dat)) %in% flt, ]
    }
  }

  sep <- sep[, c("sep3") := as.integer(get("sep3"))]

  res <- merge(dat, sep, all.x = TRUE)
  res <- res[, c("sep3") := data.table::nafill(sep3, "locf"),
             by = c(id_vars(res))]
  res <- res[, c("sep3") := data.table::nafill(sep3, fill = 0L)]

  res <- res[, c("sep3") := as.logical(get("sep3"))]

  feats <- concepts[["concept"]]
  feats <- feats[!is.na(feats)]

  res <- rm_cols(res, setdiff(data_vars(res), c(feats, "sep3")))

  miss_cols <- setdiff(feats, data_vars(res))

  if (length(miss_cols)) {
    res <- data.table::set(res, j = miss_cols, value = NA_real_)
  }

  res <- data.table::setcolorder(res, c(meta_vars(res), feats))

  dir <- file.path(dir, source)

  if (dir.exists(dir)) unlink(dir, recursive = TRUE)

  dir.create(dir, recursive = TRUE)

  write_psv(res, dir, na_rows = TRUE)
}
