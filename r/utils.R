
sepsis3_crit <- function(source, dat, win, pids = NULL) {

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

  sep3(data.table::copy(dat[["sofa"]]), si)
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

truncate_dat <- function(dat, win, flt) {

  if (is_ts_tbl(dat)) {
    repl_meta <- c("stay_id", "stay_time")
  } else {
    repl_meta <- "stay_id"
  }

  dat <- rename_cols(dat, repl_meta, meta_vars(dat), by_ref = TRUE)

  if (is_ts_tbl(dat)) {

    dat  <- dat[, c("join_time") := list(get("stay_time"))]

    join <- c(paste("stay_id ==", id_vars(win)), "join_time <= outtime")
    dat <- dat[win, on = join, nomatch = NULL]
    dat <- rm_cols(dat, c("join_time", "intime"), by_ref = TRUE)
  }

  if (length(flt) > 0L) {
    dat <- dat[!get(id_var(dat)) %in% flt, ]
  }

  dat
}

dump_dataset <- function(source = "mimic_demo", dir = tempdir()) {

  merge_all <- function(x, y) merge(x, y, all = TRUE)

  merge_tbl <- function(x) {

    ts <- vapply(x, is_ts_tbl, logical(1L))
    id <- vapply(x, is_id_tbl, logical(1L)) & ! ts

    ind <- c(which(ts), which(id))

    Reduce(merge_all, x[ind])
  }

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

    win <- stay_windows(source, id_type = "icustay", win_type = "icustay",
                        in_time = "intime", out_time = "outtime")

    dat <- load_concepts(feats, source, merge_data = FALSE,
                         id_type = "icustay", patient_ids = pid)
    sep <- sepsis3_crit(source, dat, pid)

    sep  <- sep[, c("join_time") := list(get(index_var(sep)))]

    join <- c(paste(id_vars(sep), "==", id_vars(win)), "join_time >= intime",
                                                       "join_time <= outtime")
    new <- sep[win, on = join, nomatch = NULL]
    flt <- setdiff(id_col(sep), id_col(new))
    sep <- rm_cols(new, setdiff(data_vars(new), "sep3"), by_ref = TRUE)

    dat <- lapply(dat, truncate_dat, win, flt)

    is_ts <- vapply(dat, is_ts_tbl, logical(1L))
    is_id <- vapply(dat, is_id_tbl, logical(1L)) & ! is_ts

    dat <- dat[c(which(is_ts), which(is_id))]

    while(length(dat) > 1L) {
      dat[[1L]] <- merge(dat[[1L]], dat[[2L]], all = TRUE)
      dat[[2L]] <- NULL
    }

    dat <- dat[[1L]]
  }

  sep <- sep[, c("sep3") := as.integer(get("sep3"))]

  dat <- merge(dat, sep, all.x = TRUE)
  dat <- dat[, c("sep3") := data.table::nafill(sep3, "locf"),
             by = c(id_vars(dat))]
  dat <- dat[, c("sep3") := data.table::nafill(sep3, fill = 0L)]

  dat <- dat[, c("sep3") := as.logical(get("sep3"))]

  feats <- concepts[["concept"]]
  feats <- feats[!is.na(feats)]

  dat <- rm_cols(dat, setdiff(data_vars(dat), c(feats, "sep3")),
                 by_ref = TRUE)

  miss_cols <- setdiff(feats, data_vars(dat))

  if (length(miss_cols)) {
    dat <- data.table::set(dat, j = miss_cols, value = NA_real_)
  }

  dat <- data.table::setcolorder(dat, c(meta_vars(dat), feats))

  dir <- file.path(dir, source)

  if (dir.exists(dir)) unlink(dir, recursive = TRUE)

  dir.create(dir, recursive = TRUE)

  write_psv(dat, dir, na_rows = TRUE)
}
