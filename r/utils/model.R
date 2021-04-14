
fit_predict <- function(train_src = "mimic_demo", test_src = train_src,
                        feat_set = c("locf", "basic", "wav", "sig", "full"),
                        feat_reg, predictor = c("rf", "linear", "lgbm"),
                        target = c("class", "hybrid", "reg"),
                        split = "split_0", data_dir = data_path("mm"),
                        res_dir = data_path("res"), targ_param_1 = NULL,
                        targ_param_2 = NULL, seed = 11, case_prop = 0.15,
                        n_fold = 5L, ...) {

  if (!dir.exists(res_dir)) {
    dir.create(res_dir, showWarnings = FALSE)
  }

  if (missing(feat_reg)) {

    feat_set <- match.arg(feat_set)
    feat_reg <- switch(feat_set,
      locf = "_locf$",
      basic = "_(hours|locf|derived)$",
      wav = "_(hours|locf|derived|wavelet_[0-9]+)$",
      sig = "_(hours|locf|derived|signature_[0-9]+)$",
      full = "_(hours|locf|derived|wavelet_[0-9]+|signature_[0-9]+)$"
    )

  } else {

    feat_set <- "custom"
  }

  target <- match.arg(target)

  targ_has_params <- !is.null(targ_param_1) || !is.null(targ_param_2)
  is_class <- !identical(target, "reg")

  if (targ_has_params) {

    targ_opts1 <- targ_param_opts(target, 1)
    targ_opts2 <- targ_param_opts(target, 2)

    assert_that(is.count(targ_param_1), targ_param_1 <= length(targ_opts1),
                is.count(targ_param_2), targ_param_2 <= length(targ_opts1))

    targ_opts1 <- targ_opts1[targ_param_1]
    targ_opts2 <- targ_opts2[targ_param_2]
    targ_type  <- target
    target     <- paste(target, targ_param_1, targ_param_2, sep = "_")
  }

  predictor <- match.arg(predictor)

  msg("training `", predictor, "` model on `", train_src,
      "` ({sub('_', ' ', split)}) with `", target,
      "` response and using `", feat_set, "` feature set")

  job <- file.path(res_dir,
    paste(predictor, target, feat_set, train_src, split, sep = "-")
  )

  pids <- coh_split(train_src, "train", split, case_prop = case_prop,
                    seed = as.integer(sub("^split_", "", split)) + seed)

  y <- if (targ_has_params) {

    switch(targ_type,
      class = y_class(train_src, targ_opts1, targ_opts2, path = data_dir,
                      split = split, pids = pids),
      reg = y_reg2(train_src, mid = targ_opts1, u_fp = targ_opts2,
                   path = data_dir, split = split, pids = pids)
    )

  } else {

    switch(target,
      class = y_class(train_src, 6, Inf, path = data_dir,
                      split = split, pids = pids),
      hybrid = y_class(train_src, 6, 6, path = data_dir,
                       split = split, pids = pids),
      reg = y_reg(train_src, path = data_dir, split = split, pids = pids)
    )
  }

  msg("reading train data")

  read_x_fun <- switch(predictor,
    rf = read_to_mat, linear = read_to_bm, lgbm = read_to_mat
  )

  x <- prof(
    read_x_fun(train_src, cols = feature_sel(feat_reg, predictor),
      path = data_dir, split = split, pids = pids
    )
  )

  msg("training model on {nrow(x)} x {ncol(x)} data")

  pids <- read_to_df(train_src, data_dir, cols = c("stay_id", "sep3"),
                     norm_cols = NULL, split = split, pids = pids)
  pids <- pids[, sep3 := as.logical(sep3)]
  pids <- pids[, sep3 := any(sep3), by = "stay_id"]

  set.seed(seed)

  case <- sample(unique(pids[ (sep3), ])[["stay_id"]])
  ctrl <- sample(unique(pids[!(sep3), ])[["stay_id"]])

  cafd <- rep(seq_len(n_fold), ceiling(length(case) / n_fold))[seq_along(case)]
  ctfd <- rep(seq_len(n_fold), ceiling(length(ctrl) / n_fold))[seq_along(ctrl)]

  pids <- pids[, fold := c(cafd, ctfd)[which(stay_id == c(case, ctrl))],
               by = "stay_id"]

  msg("fold sizes are {table(pids$fold)}")

  fun <- switch(predictor,
    linear = train_lin, rf = function(...) train_rf(..., seed = seed),
    lgbm = train_lgbm
  )
  mod <- prof(fun(x, y, is_class, pids$fold, n_cores(), job, ...))

  for (src in test_src) {

    rm(x)

    pids <- coh_split(src, "validation", split, case_prop = case_prop,
                      seed = as.integer(sub("^split_", "", split)) + seed)

    msg("reading `", src, "` validation data")

    x <- prof(
      read_x_fun(src, cols = feature_sel(feat_reg, predictor),
        path = data_dir, split = split, pids = pids
      )
    )

    msg("predicting")

    fun <- switch(predictor, linear = pred_lin, rf = pred_rf, lgbm = pred_lgbm)
    res <- prof(fun(mod, x))

    reg <- y_reg(src, path = data_dir, split = split, pids = pids)

    if (identical(target, "reg")) {

      y <- reg
      reg <- NULL

    } else {

      y <- if (targ_has_params) {

        switch(targ_type,
          class = y_class(src, targ_opts1, targ_opts2, path = data_dir,
                          split = split, pids = pids),
          reg = y_reg2(src, mid = targ_opts1, u_fp = targ_opts2,
                       path = data_dir, split = split, pids = pids)
        )

      } else {

        switch(target,
          class = y_class(src, 6, Inf, path = data_dir,
                          split = split, pids = pids),
          hybrid = y_class(src, 6, 6, path = data_dir,
                           split = split, pids = pids)
        )
      }
    }

    pids <- read_to_df(src, data_dir, cols = c("stay_id", "stay_time", "sep3"),
                       norm_cols = NULL, split = split, pids = pids,
                       add_id = TRUE)

    y <- split(y, pids[["unique_id"]])
    res <- split(res, pids[["unique_id"]])

    if (!identical(target, "reg")) {
      reg <- split(reg, pids[["unique_id"]])
    }

    pids <- split(pids, by = "unique_id", keep.by = FALSE, sorted = TRUE)
    onse <- lapply(pids, `[[`, "sep3")

    onse <- lapply(lapply(lapply(onse, `==`, 1), which), `[`, 1L)
    onse <- Map(`[`, lapply(pids, `[[`, "stay_time"), onse)

    pids <- lapply(pids, `[[`, "stay_time")

    res <- list(
      model = paste(predictor, target, feat_reg, sep = "::"),
      dataset_train = train_src,
      dataset_eval = src,
      split = split,
      labels = unname(y),
      utility = unname(reg),
      scores = unname(res),
      times = unname(pids),
      ids = names(pids),
      onset = unname(onse),
      pids = lapply(lapply(pids, `[[`, "stay_id"), unique)
    )

    jsonlite::write_json(res, paste0(job, "-", src, ".json"))
  }

  invisible(NULL)
}

targ_param_opts <- function(target, param, ind) {

  chr_to_ind <- function(x, opts) {
    if (is.numeric(x)) x else match(x, opts)
  }

  res <- array(
    c(seq(-12, -3),
      seq(-2, 7),
      seq(-12, 6, by = 2),
      c(0, -0.001, -0.01, -0.02, -0.05, -0.1, -0.2, -0.5, -1, -2)
    ),
    dim = c(10, 2, 2),
    dimnames = list(ind = 1:10, param = c("a", "b"), targ = c("class", "reg"))
  )

  ind <- cbind(
    chr_to_ind(ind, dimnames(res)[[1L]]),
    chr_to_ind(param, dimnames(res)[[2L]]),
    chr_to_ind(target, dimnames(res)[[3L]])
  )

  res[ind]
}

train_rf <- function(x, y, is_class, folds, n_cores, job_name, ...) {

  folds <- as.integer(folds != 1)

  opt_mns <- 10
  opt_ope <- Inf

  for (mns in c(10, 30, 100, 1000)) {

    msg("trying min node size: {mns}")

    mod <- ranger::ranger(
      y = y, x = x, probability = is_class, min.node.size = mns,
      num.threads = n_cores, case.weights = folds, holdout = TRUE,
      ...
    )

    msg("--> pred error: {mod$prediction.error}")

    if (mod$prediction.error < opt_ope) {

      opt_ope <- mod$prediction.error
      opt_mns <- mns
    }
  }

  msg("choosing min node size: {opt_mns}")

  mod <- ranger::ranger(
    y = y, x = x, probability = is_class, min.node.size = opt_mns,
    importance = "impurity", num.threads = n_cores, ...
  )

  qs::qsave(mod, paste0(job_name, ".qs"))

  mod
}

train_lin <- function(x, y, is_class, folds, n_cores, job_name, ...) {

  mod <- biglasso::cv.biglasso(
    x, y, family = ifelse(is_class, "binomial", "gaussian"),
    ncores = n_cores, cv.ind = folds, nlambda = 50, verbose = TRUE, ...
  )

  qs::qsave(mod, paste0(job_name, ".qs"))

  mod
}

train_lgbm <- function(x, y, is_class, folds, n_cores, job_name, ...) {

  folds <- lapply(seq_along(unique(folds)), `==`, folds)
  folds <- lapply(folds, which)

  dtrain <- lightgbm::lgb.Dataset(x, label = y)
  params <- list(
    objective = ifelse(is_class, "binary", "regression"),
    force_col_wise = TRUE
  )

  num_leaves <- c(31, 50, 100)
  num_trees  <- c(250, 500)
  learn_rate <- c(0.001, 0.01, 0.1, 1)

  best_score <- Inf
  opt_params <- c(num_leaves[1L], num_trees[1L], learn_rate[1L])

  for (num_leaf in num_leaves)  {
    for (num_trees in num_trees) {
      for (lr in learn_rate) {

        msg("trying num leaves: {num_leaf}, num trees: {num_trees}, learning ",
            "rate: {lr}")

        lgb_CV <- lightgbm::lgb.cv(
          params = params, data = dtrain, num_leaves = num_leaf,
          nrounds = num_trees, learning_rate = lr, boosting = "gbdt",
          folds = folds, num_threads = n_cores, reset_data = TRUE
        )

        msg("--> best score: {lgb_CV$best_score}")

        if (lgb_CV$best_score < best_score) {
          opt_params <- c(num_leaf, num_trees, lr)
          best_score <- lgb_CV$best_score
        }

        gc(verbose = FALSE)
      }
    }
  }

  msg("choosing: num leaves: {opt_params[1]}, num trees: {opt_params[2]}, ",
      "learning rate: {opt_params[3]}")

  mod <- lightgbm::lgb.train(
    params = params, data = dtrain, num_leaves = opt_params[1],
    nrounds = opt_params[2], learning_rate = opt_params[3],
    boosting = "gbdt", verbose = -1L, num_threads = n_cores
  )

  set_na <- if (is.na(mod$raw)) {
    mod$save()
    TRUE
  }

  qs::qsave(mod, paste0(job_name, ".qs"))

  if (isTRUE(set_na)) {
    mod$raw <- NA
  }

  mod
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

pred_lgbm <- function(mod, x) predict(mod, x)
