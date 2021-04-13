
eicu_hosp_coh <- function(sep, coh, thresh) {

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

  res
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
                      min_age = 14, eicu_hosp_thresh = 0.15) {

  feats <- read_var_json(var_cfg)[["concept"]]
  feats <- setdiff(feats[!is.na(feats)], "sofa")

  msg("--> determining cohort for {source}\n")

  coh <- load_concepts("age", source, id_type = "icustay")
  pid <- coh[age >= min_age, ]

  msg("--> removing {nrow(coh) - nrow(pid)} from {nrow(coh)} ids due to",
      " min age of {min_age}\n")

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
                    names = sub("_raw$", paste0("_", suffix), cols),
                    by = NULL, win = NULL, ...) {

  if (is.character(fun)) {

    win <- ceiling(win / as.double(interval(x)))
    fun <- switch(fun, min = roll::roll_min,  max = roll::roll_max,
                      mean = roll::roll_mean, var = roll::roll_var)

    x <- x[, c(names) := lapply(.SD, fun, win, min_obs = 1),
           .SDcols = cols, by = c(id_vars(x))]

  } else {

    assert_that(is.null(win))

    if (is.null(win) && is.null(by)) {

      x <- x[, c(names) := lapply(.SD, fun, ...), .SDcols = c(cols)]

    } else if (is.null(win)) {

      x <- x[, c(names) := lapply(.SD, fun, ...), .SDcols = c(cols),
             by = c(by)]

    }
  }

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

  list(total = list(ids = ids, labels = as.integer(cas)),
       dev = dev, test = tst)
}

derived_feats <- function(dat, funs = c(derived_sirs, derived_mews,
                                        derived_qsof, derived_sofa,
                                        derived_sdet, derived_sshk,
                                        derived_misc)) {

  for (fun in funs) {
    dat <- fun(dat)
  }

  dat
}

derived_sirs <- function(dat) {
  dat <- dat[, sirs_der := list(
    ((temp_locf > 38) | (temp_locf < 36)) +
    (hr_locf > 90) +
    ((resp_locf > 20) | (pco2_locf < 32)) +
    ((wbc_locf < 4) | (wbc_locf > 12))
  )]
}

derived_mews <- function(dat) {
  dat <- dat[, mews_der := list(
    (sbp_locf <= 70) * 3 +
    ((70 < sbp_locf) & (sbp_locf <= 80)) * 2 +
    ((80 < sbp_locf) & (sbp_locf <= 100)) +
    (sbp_locf >= 200) * 2 +
    ((40 < hr_locf) & (hr_locf <= 50)) +
    ((100 < hr_locf) & (hr_locf <= 110)) +
    ((110 < hr_locf) & (hr_locf < 130)) * 2 +
    (hr_locf >= 130) * 3 +
    (resp_locf < 9) * 2 +
    ((15 < resp_locf) & (resp_locf <= 20)) +
    ((20 < resp_locf) & (resp_locf < 30)) * 2 +
    (resp_locf >= 30) * 3 +
    (temp_locf < 35) * 2 +
    (temp_locf >= 38.5) * 2
  )]
}

derived_qsof <- function(dat) {
  dat <- dat[, qsofa_der := list((resp_locf >= 22) + (sbp_locf <= 100))]
}

derived_sofa <- function(dat) {
  dat <- dat[, c("scoag_der", "sliver_der", "scardio_der", "srenal_der") :=
    list(
      ((100 <= plt_locf) & (plt_locf < 150)) +
      ((50 <= plt_locf) & (plt_locf < 100)) * 2 +
      ((20 <= plt_locf) & (plt_locf < 50)) * 3 +
      (plt_locf < 20) * 4,
      ((1.2 <= bili_locf) & (bili_locf <= 1.9)) +
      ((1.9 < bili_locf) & (bili_locf <= 5.9)) * 2 +
      ((5.9 < bili_locf) & (bili_locf <= 11.9)) * 3 +
      (bili_locf > 11.9) * 4,
      (map_locf < 70) * 1,
      ((1.2 <= crea_locf) & (crea_locf <= 1.9)) +
      ((1.9 < crea_locf) & (crea_locf <= 3.4)) * 2 +
      ((3.4 < crea_locf) & (crea_locf <= 4.9)) * 3 +
      (crea_locf > 4.9) * 4
    )
  ]
  dat <- dat[, sofa_der := scoag_der + sliver_der + scardio_der + srenal_der]
}

derived_sdet <- function(dat) {
  dat <- augment(dat, "min", cols = "sofa", names = "sofa_min24",
                 win = hours(24L))
  dat <- dat[, c("sodet_der", "sofa_min24") := list(
    ((sofa - sofa_min24) >= 2) * 1, NULL
  )]
}

derived_sshk <- function(dat) {
  dat <- dat[, sepshk_der := list((map_locf < 65) + (lact_locf > 2))]
}

derived_misc <- function(dat) {
  dat <- dat[, c("shkind_der", "buncr_der", "pafi_der") := list(
    hr_locf / sbp_locf,
    bun_locf / crea_locf,
    po2_locf / fio2_locf
  )]
}

class_objective <- function(x, left = -6, right = Inf) {
  is_true(x >= left & x <= right)
}

reg_objective <- function(x, left = -12, right = 6, mid = -6, u_fp = -0.05) {

  data.table::fifelse(
    x < left, u_fp,
    data.table::fifelse(
      x < mid, (x - left) / (mid - left),
      data.table::fifelse(x == mid, 1,
        data.table::fifelse(
          x <= right, 1 - (x - mid) / (right - mid), 0
        )
      )
    ), u_fp
  )
}

phys_pos_objective <- function(x) {

  data.table::fifelse(
    x < -12, -0.05, data.table::fifelse(
      x <= -6, (x + 12) / 6, data.table::fifelse(
        x <= 3, 1 - ((x + 6) / 9), 0
      )
    ), -0.05
  )
}

phys_neg_objective <- function(x) {

  data.table::fifelse(
    x <= -6, 0, data.table::fifelse(
      x <= 3, -2 * (x + 6) / 9, 0
    ), 0
  )
}

phys_score_calc <- function(dat) {

  dat <- dat[, c("phys_pos_utility", "phys_neg_utility") := list(
    phys_pos_objective(onset_delta), phys_neg_objective(onset_delta)
  )]

  dat <- dat[, c("phys_cum_utility", "phys_opt_utility") := list(
    phys_pos_utility - phys_neg_utility,
    pmax(phys_pos_utility, phys_neg_utility)
  )]

  dat
}

objective_calc <- function(dat) {

  dat <- dat[,
    c("class_m6_p6", "class_m6_inf", "reg_m8_m1", "reg_p4_m1") := list(
      class_objective(onset_delta, -6, 6),
      class_objective(onset_delta, -6, Inf),
      reg_objective(onset_delta, mid = -8, u_fp = -1),
      reg_objective(onset_delta, mid = 4, u_fp = -1)
    )
  ]
}

pos_train <- function(x, train) {
  lapply(train, function(x, id, pos) pos & (id %in% x), id_col(x),
         index_col(x) > 0)
}

get_split <- function(x, name) {

  is_test <- identical(name, "test")

  if (is_test) {
    x <- x[["test"]]
  } else {
    x <- x[["dev"]]
  }

  x <- x[grepl("^split_", names(x))]

  if (is_test) {
    return(x)
  }

  lapply(x, `[[`, name)
}

lambda_calc <- function(train_ind, x) {

  col <- c("phys_pos_utility", "phys_neg_utility", "phys_opt_utility")
  res <- setNames(vector("numeric", length = 3L), col)

  sel <- x[["is_case"]] & train_ind

  for (i in col) {
    res[i] <- sum(x[[i]] * sel)
  }

  denom <-     res["phys_pos_utility"] -
           2 * res["phys_neg_utility"] +
               res["phys_opt_utility"]

  sel <- !x[["is_case"]] & train_ind

  for (i in col) {
    res[i] <- sum(x[[i]] * sel)
  }

  numer <- 2 * res["phys_neg_utility"] -
               res["phys_pos_utility"] -
               res["phys_opt_utility"]

  numer / denom
}

train_lambdas <- function(x, train_rows) {
  lapply(pos_train(x, train_rows), lambda_calc, x)
}

col_stat_calc <- function(train_ind, x) {

  mean_sd <- function(x, ind) {
    tmp <- x[ind]
    list(mean = mean(tmp, na.rm = TRUE), sd = sd(tmp, na.rm = TRUE))
  }

  nan_to_na <- function(x) if (is.nan(x)) NA_real_ else x

  cols <- data_vars(x)

  res <- list(
    means = setNames(vector("list", length(cols)), cols),
    stds  = setNames(vector("list", length(cols)), cols)
  )

  for (col in cols) {
    tmp <- mean_sd(x[[col]], train_ind)
    res$means[[col]] <- nan_to_na(tmp[["mean"]])
    res$stds[[col]]  <- nan_to_na(tmp[["sd"]])
  }

  res
}

train_col_stats <- function(x, train_rows) {
  lapply(pos_train(x, train_rows), col_stat_calc, x)
}

dataset_stats <- function(x, splits) {

  apply_splits <- function(x, spt, name) {
    lapply(pos_train(x, get_split(spt, name)), dataset_stat_calc, x)
  }

  list(
    total = dataset_stat_calc(NULL, x),
    train = apply_splits(x, splits, "train"),
    val = apply_splits(x, splits, "validation"),
    test = apply_splits(x, splits, "test")
  )
}

dataset_stat_calc <- function(rows, x) {

  if (!is.null(rows)) {
    x <- x[, c("stay_id", "stay_time", "is_case", "onset_ind"), with = FALSE]
    x <- x[rows, ]
  }

  time <- as.data.frame(
    table(as.double(x[["stay_time"]], units = "hours"), x[["is_case"]])
  )
  colnames(time) <- c("stay_time", "is_case", "counts")

  ons <- x[["stay_time"]][is_true(x[["onset_ind"]])]

  pat <- unique(x[, c("stay_id", "is_case"), with = FALSE])
  npt <- length(unique(pat$stay_id))
  ncs <- sum(pat$is_case)

  list(
    n_patients = npt,
    n_case = ncs,
    n_control = sum(!pat$is_case),
    prevalence = ncs / npt,
    onset_times = as.double(ons, units = "hours"),
    stay_times = time
  )
}

export_data <- function(src, dest_dir = data_path("export"), legacy = FALSE,
                        seed = 11L, ...) {

  augment_prof <- function(name, ...) {
    msg("--> augmentation step {name}")
    prof(augment(...))
  }

  lbk_wrap <- function(win, x) {

    for (fun in c("min", "max", "mean", "var")) {
      x <- augment(x, fun, paste0(fun, as.integer(win), "lbk"), win = win)
      gc()
    }

    x
  }

  old_opts <- options(datatable.alloccol = 2048L)
  on.exit(options(old_opts))

  if (!dir.exists(dest_dir)) {
    dir.create(dest_dir)
  }

  assert(is.string(src))

  dat <- load_data(src, ...)

  atr <- list(
    ricu = list(
      id_vars = id_vars(dat$dat), index_var = index_var(dat$dat),
      time_unit = units(interval(dat$dat)), time_step = time_step(dat$dat)
    ),
    mcsep = list(cohorts = dat$coh)
  )

  dat <- dat$dat
  dat <- dat[, c("female_static") := list(is_true(female_static == "Female"))]

  dat <- dat[, onset_ind := sep3]
  dat <- dat[, onset_delta := as.double(
    stay_time - stay_time[which(onset_ind)[1L]], units = "hours"),
    by = "stay_id"
  ]
  dat <- dat[, is_case := any(onset_ind, na.rm = TRUE), by = "stay_id"]

  if (!legacy) {

    dat <- dat[, sep3 := NULL]

    spt <- create_splits(unique(dat[, c("stay_id", "is_case"), with = FALSE]),
                         seed = seed)
    atr$mcsep$splits <- spt

    msg("--> datset stats calculation")
    atr$mcsep[["data_stats"]] <- prof(dataset_stats(dat, spt))
    msg("--> sep3 prevalence of {round(
         atr$mcsep$data_stats$total$prevalence * 100, 2)}% at {
         atr$mcsep$data_stats$total$n_patients} patients")

    spt <- get_split(spt, "train")

    msg("--> physionet score calculation")
    dat <- prof(phys_score_calc(dat))

    msg("--> lambda calculation")
    atr$mcsep[["lambda"]] <- prof(train_lambdas(dat, spt))

    msg("--> objective calculation")
    dat <- prof(objective_calc(dat))

  } else {

    dat <- replace_na(dat, type = "locf", by_ref = TRUE, vars = "sep3",
                      by = id_vars(dat))
    dat <- replace_na(dat, FALSE, by_ref = TRUE, vars = "sep3")
  }

  dat <- augment_prof("ind", dat, Negate(is.na), "ind")
  dat <- augment_prof("locf", dat, data.table::nafill, "locf",
                      by = id_vars(dat), type = "locf")
  col <- grep("_ind$", colnames(dat), value = TRUE)
  dat <- augment_prof("cnt", dat, cumsum, cols = col,
                      names = sub("_ind$", "_cnt", col),
                      by = id_vars(dat))

  if (!legacy) {

    msg("--> derived feature calculation")
    dat <- prof(derived_feats(dat))

    for (win in hours(c(4L, 8L, 16L))) {
      msg("--> augmentation step lbk{as.integer(win)}")
      dat <- prof(lbk_wrap(win, dat))
    }

    msg("--> train set stats calculation")
    atr$mcsep[["normalization"]] <- prof(train_col_stats(dat, spt))
  }

  dat <- dat[, c(index_var(dat)) := as.double(
    get(index_var(dat)), units = "hours"
  )]

  dat <- as.data.frame(dat)
  fil <- file.path(dest_dir,
    paste0(src, "-", gsub("\\.", "-", packageVersion("ricu")))
  )

  jsonlite::write_json(atr$mcsep$splits,
    file.path(cfg_path("splits"), paste0(basename(fil), ".json")),
    pretty = TRUE
  )

  create_parquet(dat, fil, atr, chunk_size = 1e3)
}
