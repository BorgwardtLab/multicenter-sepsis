
eicu_hosp_coh <- function(sep, coh, thresh = 0.05) {

  assert(is.numeric(thresh), is.scalar(thresh), thresh > 0, thresh < 1)

  idv <- id_var(sep)
  hid <- "hospitalid"
  sep <- unique(sep[, c(idv, "sep3"), with = FALSE])

  hsp <- load_id("patient", "eicu", cols = c(idv, hid))
  hsp <- merge(hsp, coh, all.y = TRUE)

  dat <- merge(sep, hsp, all = TRUE)
  dat <- dat[, list(septic = sum(sep3, na.rm = TRUE),
                    total = .N), by = c(hid)]
  dat <- dat[, prop := septic / total]
  nrw <- nrow(dat)
  dat <- dat[prop >= thresh, hid, with = FALSE]

  msg("--> selecting {nrow(dat)} hospitals from {nrw} based on a sep3",
      " prevalence of {thresh}\n")

  res <- merge(hsp, dat, all.y = TRUE, by = hid)
  nrw <- nrow(coh)

  msg("--> removing {nrw - nrow(res)} from {nrw} ids based on hosp",
      " selection\n")
}

load_physionet <- function(var_cfg = cfg_path("variables.json"),
                           data_dir = data_path("physionet2019"),
                           min_age = 14) {

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

  dat <- dat[age >= min_age, ]
  coh <- unique(id_col(dat))

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

  list(win = win, dat = res[cnc], sep = sep, coh = list(initial = coh))
}

sepsis3_crit <- function(source, pids = NULL, keep_components = FALSE,
                         dat = NULL) {

  if (is.null(dat)) {
    dat <- load_concepts("sofa", source, patient_ids = pids,
                         keep_components = keep_components)
  } else if (!is_ts_tbl(dat)) {
    dat <- data.table::copy(dat[["sofa"]])
  }

  stopifnot(is_ts_tbl(dat))

  if (grepl("eicu", source)) {

    si <- load_concepts("susp_inf", source, abx_min_count = 2L,
                        positive_cultures = TRUE, id_type = "icustay",
                        patient_ids = pids, si_mode = "or",
                        keep_components = keep_components)

  } else if (identical(source, "hirid")) {

    si <- load_concepts("susp_inf", source, abx_min_count = 2L,
                        id_type = "icustay", patient_ids = pids,
                        si_mode = "or", keep_components = keep_components)

  } else {

    si <- load_concepts("susp_inf", source, id_type = "icustay",
                        patient_ids = pids, keep_components = keep_components)
  }

  sep3(dat, si, si_window = "any", keep_components = keep_components)
}

load_ricu <- function(source, var_cfg = cfg_path("variables.json"),
                      min_age = 14, eicu_hosp_thresh = 0.05) {

  feats <- read_var_json(var_cfg)[["concept"]]
  feats <- setdiff(feats[!is.na(feats)], "sofa")

  msg("--> determining cohort for {source}\n")

  coh <- load_concepts("age", source, id_type = "icustay")
  nrw <- nrow(coh)
  pid <- coh[age >= min_age, ]

  msg("--> removing {nrw - nrow(coh)} from {nrw} ids due to min age of",
      " {min_age}\n")

  sof <- load_concepts("sofa", source, patient_ids = pid)
  sep <- sepsis3_crit(source, pid, dat = sof)

  if (identical("eicu", source)) {
    pid <- eicu_hosp_coh(sep, pid, thresh = eicu_hosp_thresh)
  }

  win <- stay_windows(source, id_type = "icustay", win_type = "icustay",
                      in_time = "intime", out_time = "outtime",
                      patient_ids = pid)

  dat <- load_concepts(feats, source, merge_data = FALSE,
                       id_type = "icustay", patient_ids = pid)

  list(win = win, dat = c(dat, list(sofa = sof)), sep = sep,
       coh = list(initial = unique(id_col(pid))))
}

load_data <- function(source, var_cfg = cfg_path("variables.json"), ...,
                      min_stay_len = hours(6L), min_n_meas = 4L,
                      min_onset = hours(4L), max_onset = days(7L),
                      ub_case = hours(24L), ub_ctrl = max_onset + ub_case,
                      lb_all = -hours(48L), max_miss_win = hours(12L)) {

  pmin_dt <- function(x, y) {
    x[x > y] <- `units<-`(y, units(x))
    x
  }

  truncate_dat <- function(dat, win, lower) {

    join <- paste(id_var(dat), "==", id_var(win))
    vars <- c(meta_vars(dat), data_var(dat))

    if (is_ts_tbl(dat)) {
      dat  <- dat[get(index_var(dat)) >= lower, ]
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

  coh <- dat$coh
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
    is.na(sep_time), pmin_dt(outtime, ub_ctrl), sep_time + ub_case
  )]

  msg("--> removing time-points due to censoring data {format_unit(ub_case)}",
      " after onsets and [{format_unit(lb_all)}, {format_unit(ub_ctrl)}]",
      " w.r.t. stay admission.\n")

  dat <- lapply(c(dat$dat, list(dat$sep)), truncate_dat, win, lb_all)

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

  tsn <- cfg[!cfg$category %in% c("static", "baseline", "extra"), ]
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

  coh$final <- unique(id_col(dat))

  list(dat = dat, coh = coh)
}

augment <- function(x, fun, suffix,
                    cols = grep("_raw$", colnames(x), value = TRUE),
                    by = NULL, win = NULL, ...) {

  msg("--> augmentation step {suffix}")

  if (is.numeric(win)) {
    win <- as.difftime(win, units = "hours")
  }

  orig_rows <- data.table::copy(nrow(x))

  if (is.character(fun)) {

    assert_that(is.null(by))

    inf_to_na <- function(x) replace(x, is.infinite(x), NA)
    nan_to_na <- function(x) replace(x, is.nan(x), NA)

    ival  <- interval(x)
    idcol <- id_vars(x)

    win <- as.double(win, units = units(ival))

    lags <- seq_len(
      ceiling(win / as.double(ival))
    )

    funs <- c(
      min = function(x) inf_to_na(matrixStats::rowMins(x, na.rm = TRUE)),
      max = function(x) inf_to_na(matrixStats::rowMaxs(x, na.rm = TRUE)),
      mean = function(x) nan_to_na(rowMeans(x, na.rm = TRUE)),
      var = function(x) matrixStats::rowVars(x, na.rm = TRUE)
    )[fun]

    assert_that(all(vapply(funs, is.function, logical(1L))))

    for (col in cols) {

      tmp_cols <- paste0(col, lags)
      targ_col <- paste0(sub("_raw$", "", col), "_", fun, as.integer(win),
                         suffix)

      x <- x[, c(tmp_cols) := data.table::shift(get(col), lags), by = c(idcol)]
      x <- x[, c(targ_col) := lapply(funs, do.call, list(as.matrix(.SD))),
             .SDcols = c(col, tmp_cols)]
      x <- x[, c(col, tmp_cols) := NULL]
    }

  } else {

    names <- sub("_raw$", paste0("_", suffix), cols)

    if (is.null(win) && is.null(by)) {

      x <- x[, lapply(.SD, fun, ...), .SDcols = c(cols)]

    } else if (is.null(win)) {

      x <- x[, lapply(.SD, fun, ...), .SDcols = c(cols), by = c(by)]

    } else {

      x <- slide(x, lapply(.SD, fun, ...), win, .SDcols = c(cols))
    }

    x <- rename_cols(x, names, cols, by_ref = TRUE)
    x <- rm_cols(x, setdiff(colnames(x), names), by_ref = TRUE)
  }

  assert(identical(orig_rows, nrow(x)))

  x
}

create_splits <- function(x, test_size = 0.1, val_size = 0.1, boost_size = 0.8,
                          n_splits = 5, seed = 11L) {

  set.seed(seed)

  ids <- x$stay_id
  cas <- x$is_case
  nms <- paste0("split_", seq_len(n_splits) - 1)

  sep <- ids[cas]
  ctr <- ids[!cas]

  tst_sep <- sample(sep, ceiling(length(sep) * test_size))
  tst_ctr <- sample(ctr, ceiling(length(ctr) * test_size))
  dev_sep <- setdiff(sep, tst_sep)
  dev_ctr <- setdiff(ctr, tst_ctr)

  tst_all <- sort(c(tst_sep, tst_ctr))
  dev_all <- sort(c(dev_sep, dev_ctr))

  dev <- replicate(n_splits, {
    val <- c(sample(dev_sep, ceiling(length(dev_sep) * val_size)),
             sample(dev_ctr, ceiling(length(dev_ctr) * val_size)))
    list(train = sort(setdiff(dev_all, val)), validation = sort(val))
  }, simplify = FALSE)

  names(dev) <- nms

  dev <- c(list(total = dev_all), dev)

  tst <- replicate(n_splits,
    sort(c(sample(tst_sep, ceiling(length(tst_sep) * boost_size)),
           sample(tst_ctr, ceiling(length(tst_ctr) * boost_size)))),
    simplify = FALSE
  )

  names(tst) <- nms

  tst <- c(list(total = tst_all), tst)

  list(total = ids, labels = as.integer(cas), dev = dev, test = tst)
}

export_data <- function(src, dest_dir = data_path("export"), legacy = FALSE,
                        seed = 11L, ...) {

  augment_prof <- function(...) prof(augment(...))

  assert(is.string(src), dir.exists(dest_dir))

  dat <- load_data(src, ...)

  atr <- list(
    ricu = list(
      id_vars = id_vars(dat$dat), index_var = index_var(dat$dat),
      time_unit = units(interval(dat$dat)), time_step = time_step(dat$dat)
    ),
    mcsep = list(cohorts = dat$coh)
  )

  dat <- dat$dat

  dat <- dat[, c("female") := list(is_true(female == "Female"))]

  dat <- dat[, onset := sep3]
  dat <- replace_na(dat, type = "locf", by_ref = TRUE, vars = "sep3",
                    by = id_vars(dat))
  dat <- replace_na(dat, FALSE, by_ref = TRUE, vars = "sep3")

  dat <- dat[, is_case := any(sep3), by = stay_id]

  spt <- create_splits(unique(dat[, c("stay_id", "is_case"), with = FALSE]),
                       seed = seed)
  atr$mcsep$splits <- spt
  spt <- lapply(spt$dev[grepl("^split_", names(spt$dev))], `[[`, "train")

  ind <- augment_prof(dat, Negate(is.na), "ind")
  lof <- augment_prof(dat, data.table::nafill, "locf", by = id_vars(dat),
                      type = "locf")

  lbk <- if (!legacy) {

    funs <- c("min", "max", "mean", "var")
    wins <- c(4L, 8L, 16L)

    do.call("c",
      Map(augment_prof, list(dat), fun = list(funs), suffix = "lbk",
          win = hours(wins))
    )
  }

  dat <- dat[, c(index_var(dat)) := as.double(
    get(index_var(dat)), units = "hours"
  )]

  res <- c(dat, ind, lof, lbk)
  res <- data.table::setDT(res)
  fil <- file.path(dest_dir, paste(src, packageVersion("ricu"), sep = "_"))

  create_parquet(res, fil, atr, chunk_size = 1e3)
}
