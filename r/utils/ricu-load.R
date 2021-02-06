
load_challenge <- function(dir = data_path("challenge"),
                           cfg = cfg_path("variables.json")) {

  concepts <- read_var_json(cfg)
  data_dir <- file.path(dir, "training_setB")

  assert(dir.exists(data_dir), msg = "Directory {data_dir} not found.")

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

  dat <- rename_cols(dat, feats, names(feats), by_ref = TRUE)

  merge(dat, sep, all = TRUE)
}

load_ricu <- function(source, var_cfg = cfg_path("variables.json"),
                      coh_cfg = cfg_path("cohorts.json"),
                      min_onset = hours(4L), max_onset = days(7L),
                      cut_case = hours(24L), cut_ctrl = max_onset + cut_case) {

  truncate_dat <- function(dat, win, flt) {

    if (length(flt) > 0L) {
      dat <- dat[!get(id_var(dat)) %in% flt, ]
    }

    if (is_ts_tbl(dat)) {
      repl_meta <- c("stay_id", "stay_time")
    } else {
      repl_meta <- "stay_id"
    }

    dat <- rename_cols(dat, repl_meta, meta_vars(dat), by_ref = TRUE)

    if (is_ts_tbl(dat)) {

      dat  <- dat[, c("join_time") := list(get("stay_time"))]

      join <- c(paste("stay_id ==", id_vars(win)), "join_time <= cuttime")
      dat <- dat[win, on = join, nomatch = NULL]
      dat <- rm_cols(dat, "join_time", by_ref = TRUE)
    }

    dat
  }

  feats <- read_var_json(var_cfg)[["concept"]]
  feats <- feats[!is.na(feats)]
  pids  <- unlist(jsonlite::read_json(coh_cfg)[[source]]$cohort)

  win <- stay_windows(source, id_type = "icustay", win_type = "icustay",
                      in_time = "intime", out_time = "outtime")

  dat <- load_concepts(feats, source, merge_data = FALSE,
                       id_type = "icustay", patient_ids = pids)
  sep <- sepsis3_crit(source, pids, dat)

  sep  <- sep[, c("join_time") := list(get(index_var(sep)))]

  join <- c(paste(id_vars(sep), "==", id_vars(win)), "join_time >= intime",
                                                     "join_time <= outtime")
  new <- sep[win, on = join, nomatch = NULL]
  tmp <- nrow(new)

  msg("\n\n--> removing {nrow(sep) - tmp} patients due to onsets",
      " outside of icu stay.\n")

  new <- new[get(index_var(new)) >= min_onset &
             get(index_var(new)) <= max_onset, ]

  msg("\n\n--> removing {tmp - nrow(new)} patients due to onsets",
      " outside of [{format(min_onset)}, {format(max_onset)}].\n")

  flt <- setdiff(id_col(sep), id_col(new))

  sep <- rm_cols(new, setdiff(data_vars(new), "sep3"), by_ref = TRUE)
  sep <- rename_cols(sep, "sep3_time", index_var(sep))
  win <- merge(win, rm_cols(as_id_tbl(sep), "sep3", by_ref = TRUE),
               all.x = TRUE)
  win <- win[, cuttime := data.table::fifelse(
    is.na(sep3_time), pmin(outtime, cut_ctrl), sep3_time + cut_case
  )]

  msg("\n\n--> removing up to {format(sum(win$outtime - win$cuttime))} due",
      " to censoring data {format(cut_case)} after onsets and",
      " {format(cut_ctrl)} into stays.\n")

  win <- rm_cols(win, c("intime", "outtime", "sep3_time"), by_ref = TRUE)
  dat <- lapply(dat, truncate_dat, win, flt)

  is_ts <- vapply(dat, is_ts_tbl, logical(1L))
  is_id <- vapply(dat, is_id_tbl, logical(1L)) & ! is_ts

  dat <- dat[c(which(is_ts), which(is_id))]

  while(length(dat) > 1L) {
    dat[[1L]] <- merge(dat[[1L]], dat[[2L]], all = TRUE)
    dat[[2L]] <- NULL
  }

  merge(dat[[1L]], sep, all = TRUE)
}

sepsis3_crit <- function(source, pids = NULL,
  dat = load_concepts("sofa", source, patient_ids = pids)) {

  if (!is_ts_tbl(dat)) {
    dat <- data.table::copy(dat[["sofa"]])
  }

  stopifnot(is_ts_tbl(dat))

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

  sep3(dat, si)
}

export_data <- function(src, dest_dir = data_path("export"), ...) {

  assert(is.string(src), dir.exists(dir))

  if (identical(src, "challenge")) {
    dat <- load_challenge(...)
  } else {
    dat <- load_ricu(src, ...)
  }

  dat <- dat[, onset := sep3]
  dat <- replace_na(dat, type = "locf", by_ref = TRUE, vars = "sep3",
                    by = id_vars(dat))
  dat <- replace_na(dat, FALSE, by_ref = TRUE, vars = "sep3")

  create_parquet(dat, file.path(dest_dir, src))
}
