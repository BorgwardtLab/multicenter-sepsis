
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
