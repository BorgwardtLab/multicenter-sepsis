
fit_predict <- function(train_src = "mimic_demo", test_src = train_src,
                        feat_set = c("locf", "basic", "wav", "sig", "full"),
                        feat_reg, predictor = c("linear", "rf", "lgbm"),
                        target = c("class", "hybrid", "reg"),
                        split = "split_0", data_dir = data_path("mm"),
                        res_dir = data_path("res"), seed = 11, ...) {

  set.seed(seed)

  if (!dir.exists(res_dir)) {
    dir.create(res_dir, showWarnings = FALSE)
  }

  if (missing(feat_reg)) {

    feat_set <- match.arg(feat_set)
    feat_reg <- switch(feat_set,
      basic = "_(hours|locf|derived)$",
      locf = "_locf$",
      wav = "_(hours|locf|derived|wavelet_[0-9]+)$",
      sig = "_(hours|locf|derived|signature_[0-9]+)$",
      full = "_(hours|locf|derived|wavelet_[0-9]+|signature_[0-9]+)$"
    )

  } else {

    feat_set <- "custom"
  }

  target    <- match.arg(target)
  predictor <- match.arg(predictor)

  msg("training `", predictor, "` model on `", train_src, "` with `", target,
      "` response and using `", feat_set, "` feature set")

  job <- file.path(res_dir, paste(predictor, target, feat_set, train_src,
                                  sep = "-"))

  pids <- coh_split(train_src, "train", split)

  y <- switch(target,
    class = y_class(train_src, 6L, Inf, path = data_dir,
                    split = split),
    hybrid = y_class(train_src, 6L, 6L, path = data_dir,
                     split = split),
    reg = y_reg(train_src, path = data_dir, split = split)
  )

  msg("reading train data")

  read_x_fun <- switch(predictor,
    rf = read_to_mat, linear = read_to_bm, lgbm = read_to_mat
  )

  x <- prof(
    read_x_fun(train_src, cols = feature_sel(feat_reg, predictor),
      path = data_dir, split = split
    )
  )

  msg("training model")

  pids <- read_to_df(train_src, data_dir, cols = c("stay_id", "sep3"),
                     norm_cols = NULL, split = split, pids = pids)
  pids <- pids[, sep3 := as.logical(sep3)]
  pids <- pids[, sep3 := any(sep3), by = "stay_id"]

  n_fold <- 5L

  case <- sample(unique(pids[ (sep3), ])[["stay_id"]])
  ctrl <- sample(unique(pids[!(sep3), ])[["stay_id"]])

  cafd <- rep(seq_len(n_fold), ceiling(length(case) / n_fold))[seq_along(case)]
  ctfd <- rep(seq_len(n_fold), ceiling(length(ctrl) / n_fold))[seq_along(ctrl)]

  pids <- pids[, fold := c(cafd, ctfd)[which(stay_id == c(case, ctrl))],
               by = "stay_id"]

  fun <- switch(predictor,
    linear = train_lin, rf = function(...) train_rf(..., seed = seed),
    lgbm = train_lgbm
  )
  mod <- prof(fun(x, y, !identical(target, "reg"), pids$fold, n_cores(), ...))

  qs::qsave(mod, paste0(job, ".qs"))

  for (src in test_src) {

    rm(x, y)

    pids <- coh_split(src, "validation", split)

    msg("reading `", src, "` validation data")

    x <- prof(
      read_x_fun(src, cols = feature_sel(feat_reg, predictor),
        path = data_dir, split = split, pids = pids
      )
    )

    msg("predicting")

    fun <- switch(predictor, linear = pred_lin, rf = pred_rf, lgbm = pred_lgbm)
    res <- prof(fun(mod, x))

    y <- switch(target,
      class = y_class(src, 6L, Inf, path = data_dir,
                      split = split, pids = pids),
      hybrid = y_class(src, 6L, 6L, path = data_dir,
                       split = split, pids = pids),
      reg = y_reg(src, path = data_dir, split = split, pids = pids)
    )

    pids <- read_to_df(src, data_dir, cols = c("stay_id", "stay_time"),
                       norm_cols = NULL, split = split, pids = pids)
    pids <- pids[, stay_time := as.double(stay_time)]

    y <- split(y, pids[["stay_id"]])
    res <- split(res, pids[["stay_id"]])

    pids <- split(pids, by = "stay_id", keep.by = FALSE)
    pids <- lapply(pids, `[[`, "stay_time")

    res <- list(
      model = paste(predictor, target, feat_reg, sep = "::"),
      dataset_train = train_src,
      dataset_eval = src,
      split = split,
      labels = y,
      scores = res,
      times = pids,
      ids = names(pids)
    )

    jsonlite::write_json(res, paste0(job, "-", src, ".json"))
  }

  invisible(NULL)
}

train_rf <- function(x, y, is_class, folds, n_cores, ...) {

  folds <- as.integer(folds == 1)

  opt_mns <- 10
  opt_ope <- Inf

  for (mns in c(10, 30, 100, 500)) {

    mod <- ranger::ranger(
      y = y, x = x, probability = is_class, min.node.size = mns,
      num.threads = n_cores, case.weights = ids[["folds"]], holdout = TRUE,
      ...
    )

    if (mod$prediction.error < opt_ope) {

      opt_ope <- mod$prediction.error
      opt_mns <- mns
    }
  }

  ranger::ranger(
    y = y, x = x, probability = is_class, min.node.size = opt_mns,
    importance = "impurity", num.threads = n_cores, ...
  )
}

train_lin <- function(x, y, is_class, folds, n_cores, ...) {
  biglasso::cv.biglasso(
    x, y, family = ifelse(is_class, "binomial", "gaussian"),
    ncores = n_cores, cv.ind = folds, ...
  )
}

pred_rf <- function(mod, x) {

  res <- predict(mod, x, type = "response")
  cls <- identical(res$treetype, "Probability estimation")

  res <- res$predictions

  if (cls) {
    res <- res[, 2L]
  }

  res
}

pred_lin <- function(mod, x) {
  predict(mod, x, type = "response")[, 1L]
}

train_lgbm <- function(x, y, is_class, folds, n_cores, ...) {

  folds <- lapply(seq_along(unique(folds)), `==`, folds)
  folds <- lapply(folds, which)

  dtrain <- lightgbm::lgb.Dataset(x, label = y)
  params <- list(objective = ifelse(is_class, "binary", "regression"))

  best_score <- Inf
  opt_params <- c(30, 100, 0.001)

  for (num_leaf in c(30, 50, 100))  {
    for (num_trees in c(100, 200, 500)) {
      for (lr in c(0.001, 0.01, 0.1, 0.5, 1)) {

        lgb_CV <- lightgbm::lgb.cv(
          params = params, data = dtrain, num_leaves = num_leaf,
          nrounds = num_trees, learning_rate = lr, boosting = "gbdt",
          folds = pts_folds, verbose = -1L, num_threads = n_cores
        )

        if (lgb_CV$best_score < best_score) {
          opt_params <- c(num_leaf, num_trees, lr)
          best_score <- lgb_CV$best_score
        }
      }
    }
  }

  lightgbm::lgb.train(
    params = params, data = dtrain, num_leaves = opt_params[1],
    nrounds = opt_params[2], learning_rate = opt_params[3],
    boosting = "gbdt", verbose = -1L, num_threads = n_cores
  )
}

pred_lgbm <- function(mod, x) {
  browser()
  predict(mod, x)
}
