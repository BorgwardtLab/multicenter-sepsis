
load_physionet <- function(var_cfg = cfg_path("variables.json"),
                           coh_cfg = cfg_path("cohorts.json"),
                           data_dir = data_path("physionet2019")) {

  concepts <- read_var_json(var_cfg)
  data_dir <- file.path(data_dir, "training_setB")

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

  win <- collapse(dat, start = "intime", end = "outtime")
  dat <- rename_cols(dat, feats, names(feats), by_ref = TRUE)

  res <- unmerge(dat)
  names(res) <- vapply(res, data_vars, character(1L))

  cnc <- concepts$concept[!is.na(concepts$concept)]
  mis <- !cnc %in% names(res)

  if (any(mis)) {
    add <- res[[1L]][0L, ]
    add <- Map(rename_cols, list(add), cnc[mis], data_var(add))
    names(add) <- cnc[mis]
    res <- c(res, add)
  }

  list(win = win, dat = res[cnc], sep = sep)
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

  sep3(dat, si, si_window = "any")
}

load_ricu <- function(source, var_cfg = cfg_path("variables.json"),
                      coh_cfg = cfg_path("cohorts.json")) {

  feats <- read_var_json(var_cfg)[["concept"]]
  feats <- feats[!is.na(feats)]
  cohor <- jsonlite::read_json(coh_cfg, simplifyVector = TRUE, flatten = TRUE)
  pids  <- unlist(cohor[[source]]$initial)

  win <- stay_windows(source, id_type = "icustay", win_type = "icustay",
                      in_time = "intime", out_time = "outtime",
                      patient_ids = pids)

  dat <- load_concepts(feats, source, merge_data = FALSE,
                       id_type = "icustay", patient_ids = pids)
  sep <- sepsis3_crit(source, pids, dat)

  list(win = win, dat = dat, sep = sep)
}

load_data <- function(source, var_cfg = cfg_path("variables.json"), ...,
                      min_stay_len = hours(6L), min_n_meas = 4L,
                      min_onset = hours(4L), max_onset = days(7L),
                      cut_case = hours(24L), cut_ctrl = max_onset + cut_case,
                      max_miss_win = hours(12L)) {

  pmin_dt <- function(x, y) {
    x[x > y] <- `units<-`(y, units(x))
    x
  }

  truncate_dat <- function(dat, win) {

    join <- paste(id_var(dat), "==", id_var(win))
    vars <- c(meta_vars(dat), data_var(dat))

    if (is_ts_tbl(dat)) {
      dat  <- dat[, c("join_time") := list(get(index_var(dat)))]
      join <- c(join, "join_time <= cuttime")
    }

    dat <- dat[win, on = join, nomatch = NULL]
    dat <- rm_cols(dat, setdiff(colnames(dat), vars))

    dat
  }

  miss_runs <- function(x, ival, thresh) {

    if (all(x)) {
      wins <- 0L
    } else {
      tmp <- rle(x)
      wins <- tmp$lengths[!tmp$values]
    }

    !any(wins * ival > thresh)
  }

  assert(is.string(source))

  dat <- prof(
    if (identical(source, "physionet2019")) {
      load_physionet(var_cfg = var_cfg, ...)
    } else {
      load_ricu(source, var_cfg = var_cfg, ...)
    }
  )

  nav <- vapply(dat$dat, nrow, integer(1L)) == 0

  msg("--> successfully loaded {length(dat$dat)} features, {sum(nav)}",
      " of which are fully missing")

  tmp <- rm_cols(dat$sep, data_vars(dat$sep))
  tmp <- rename_cols(tmp, "sep_time", index_var(dat$sep), by_ref = TRUE)

  win <- merge(dat$win, tmp, all.x = TRUE)
  tmp <- nrow(win)

  win <- win[is.na(sep_time) | (sep_time >= intime & sep_time <= outtime), ]

  msg("--> removing {tmp - nrow(win)} patients due to onsets",
      " outside of icu stay.\n")

  tmp <- nrow(win)
  win <- win[is.na(sep_time) | (sep_time >= min_onset &
                                sep_time <= max_onset), ]

  msg("--> removing {tmp - nrow(win)} patients due to onsets",
      " outside of [{format_unit(min_onset)},",
      " {format_unit(max_onset)}].\n")

  tmp <- nrow(win)
  win <- win[outtime > min_stay_len, ]

  msg("--> removing {tmp - nrow(win)} patients due to stay length <",
      " {format_unit(min_stay_len)}].\n")

  win <- win[, cuttime := data.table::fifelse(
    is.na(sep_time), pmin_dt(outtime, cut_ctrl), sep_time + cut_case
  )]

  msg("--> removing up to {format_unit(sum(win$outtime - win$cuttime))}",
      " due to censoring data {format_unit(cut_case)} after onsets",
      " and {format_unit(cut_ctrl)} into stays.\n")

  dat <- lapply(c(dat$dat, list(dat$sep)), truncate_dat, win)

  is_ts <- vapply(dat, is_ts_tbl, logical(1L))
  is_id <- vapply(dat, is_id_tbl, logical(1L)) & ! is_ts

  sta <- dat[which(is_id)]
  dat <- dat[which(is_ts)]

  while(length(dat) > 1L) {
    dat[[1L]] <- merge(dat[[1L]], dat[[2L]], all = TRUE)
    dat[[2L]] <- NULL
  }

  tmp <- nrow(dat[[1L]])

  dat <- dat[[1L]][, all_miss := FALSE]
  dat <- fill_gaps(dat)
  dat <- dat[, all_miss := is.na(all_miss)]

  msg("--> adding {nrow(dat) - tmp} time points to create a regular time",
      " series.\n")

  if (length(sta) > 0L) {

    while(length(sta) > 1L) {
      sta[[1L]] <- merge(sta[[1L]], sta[[2L]], all = TRUE)
      sta[[2L]] <- NULL
    }

    dat <- merge(dat, sta[[1L]], all = TRUE)
  }

  dat <- rm_na(dat, meta_vars(dat), "any")
  dat <- rename_cols(dat, c("stay_id", "stay_time"), meta_vars(dat),
                     by_ref = TRUE)

  cfg <- read_var_json(var_cfg)
  cfg <- cfg[!is.na(cfg$concept), ]

  tsn <- cfg[!cfg$category %in% c("static", "baseline"), ]
  exp <- !nav[tsn$concept]
  exp <- names(exp[exp])

  dat <- dat[, ts_miss := as.integer(
    rowSums(data.table::setDF(lapply(.SD, is.na)))
  ), .SDcols = exp]
  dat <- dat[, ts_avail := ts_miss < length(exp)]

  cnt <- dat[stay_time >= 0, list(ts_count = sum(ts_avail)), by = c("stay_id")]
  cnt <- cnt[ts_count >= min_n_meas, ]
  cnt <- cnt[, ts_count := NULL]

  tmp <- nrow(dat)
  dat <- merge(dat, cnt, all.y = TRUE)

  msg("--> removing {tmp - nrow(dat)} patients due to fewer",
      " than {min_n_meas} in-icu measurements.")

  mis <- dat[stay_time > 0, list(
    keep = miss_runs(ts_avail, interval(dat), max_miss_win)
  ), by = c("stay_id")]
  mis <- mis[(keep), ]
  mis <- mis[, keep := NULL]

  tmp <- nrow(dat)
  dat <- merge(dat, mis, all.y = TRUE)

  msg("--> removing {tmp - nrow(dat)} patients due to missing windows",
      " of size larger than {format_unit(max_miss_win)}.")

  dat <- rename_cols(dat, cfg$name, cfg$concept, by_ref = TRUE,
                     skip_absent = TRUE)

  stopifnot(all(cfg$name %in% data_vars(dat)))

  exp <- c(meta_vars(dat), cfg$name)
  dat <- data.table::setcolorder(dat, c(exp, setdiff(colnames(dat), exp)))

  dat <- rename_cols(dat, paste0(tsn$name, "_raw"), tsn$name, by_ref = TRUE)

  msg("--> loading complete")

  dat
}

augment <- function(x, fun, suffix,
                    cols = grep("_raw$", colnames(x), value = TRUE),
                    by = NULL, win = NULL, tmpdir = tempdir(),
                    names = sub("_raw$", paste0("_", suffix), cols), ...) {

  inf_to_na <- function(x) replace(x, is.infinite(x), NA)

  msg("--> augmentation step {suffix}")

  if (is.numeric(win)) {
    win <- as.difftime(win, units = "hours")
  }

  if (is.null(win) && is.null(by)) {

    res <- x[, lapply(.SD, fun, ...), .SDcols = c(cols)]

  } else if (is.null(win)) {

    res <- x[, lapply(.SD, fun, ...), .SDcols = c(cols), by = c(by)]

  } else {

    res <- slide(x, lapply(.SD, fun, ...), win, .SDcols = c(cols))
  }

  assert(identical(nrow(x), nrow(res)))

  res <- rename_cols(res, names, cols, by_ref = TRUE)
  res <- rm_cols(res, setdiff(colnames(res), names), by_ref = TRUE)

  res
}

prof <- function(expr, envir = parent.frame()) {

  mem <- memuse::Sys.procmem()
  tim <- Sys.time()

  res <- eval(expr, envir = envir)

  cur <- memuse::Sys.procmem()
  cil <- cur[["peak"]] - mem[["peak"]]

  msg("    Runtime: {format(Sys.time() - tim, digits = 4)}")
  if (length(cil)) msg("    Memory ceiling increased by: {as.character(cil)}")
  msg("    Current memory usage: {as.character(cur[['size']])}")

  res
}

export_data <- function(src, dest_dir = data_path("export"),
                        coh_cfg = cfg_path("cohorts.json"), ...) {

  augment_prof <- function(...) prof(augment(...))

  assert(is.string(src), dir.exists(dir))

  dat <- load_data(src, coh_cfg = coh_cfg, ...)

  cohor <- jsonlite::read_json(coh_cfg, simplifyVector = TRUE, flatten = TRUE)
  cohor[[src]]$final <- unique(id_col(dat))
  jsonlite::write_json(cohor, coh_cfg, pretty = TRUE)

  dat <- dat[, c("female") := list(female == "Female")]

  dat <- dat[, onset := sep3]
  dat <- replace_na(dat, type = "locf", by_ref = TRUE, vars = "sep3",
                    by = id_vars(dat))
  dat <- replace_na(dat, FALSE, by_ref = TRUE, vars = "sep3")

  dat <- dat[, is_case := any(sep3), by = stay_id]

  ind <- augment_prof(dat, Negate(is.na), "ind")
  lof <- augment_prof(dat, data.table::nafill, "locf", by = id_vars(dat),
                      type = "locf")

  dat <- dat[, c(index_var(dat)) := as.double(
    get(index_var(dat)), units = "hours"
  )]

  res <- c(dat, ind, lof)
  res <- data.table::setDT(res)

  create_parquet(res,
    file.path(dest_dir, paste(src, packageVersion("ricu"), sep = "_")),
    chunk_size = 1e3
  )
}
