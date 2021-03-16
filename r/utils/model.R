
fit_predict <- function(train_src = "mimic_demo", test_src = train_src,
                        feat_set = c("basic", "wav", "sig", "full"), feat_reg,
                        predictor = c("linear", "rf"),
                        target = c("class", "hybrid", "reg"),
                        split = "split_0", data_dir = data_path("mm"),
                        res_dir = data_path("res"), ...) {

  if (!dir.exists(res_dir)) {
    dir.create(res_dir, showWarnings = FALSE)
  }

  if (missing(feat_reg)) {

    feat_set <- match.arg(feat_set)
    feat_reg <- switch(feat_set,
      basic = "_(hours|locf|derived)$",
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

  mat <- switch(predictor, rf = "mem", linear = "big")
  job <- file.path(res_dir, paste(predictor, target, feat_set, train_src,
                                  sep = "-"))
  
  pids <- coh_split(train_src, "train", split)
  
  y <- switch(target,
    class = y_class(train_src, hours(6L), hours(Inf), path = data_dir,
                    split = split, pids = pids),
    hybrid = y_class(train_src, hours(6L), hours(6L), path = data_dir,
                     split = split, pids = pids),
    reg = y_reg(train_src, path = data_dir, split = split, pids = pids)
  )

  
  
  msg("reading train data")

  x <- prof(
    read_to_mat(train_src, cols = feature_sel(feat_reg, predictor),
      path = data_dir, split = split, mat_type = mat, pids = pids
    )
  )
  
  msg("training model")
  
  pids <- read_to_df(train_src, data_dir, cols = c("stay_id", "stay_time"),
                     norm_cols = NULL, split = split, pids = pids)
  
  fun <- switch(predictor, linear = train_lin, rf = train_rf)
  mod <- prof(fun(x, y, !identical(target, "reg"), pids, n_cores(), ...))

  qs::qsave(mod, paste0(job, ".qs"))

  for (src in test_src) {

    rm(x, y)

    pids <- coh_split(src, "validation", split)

    msg("reading `", src, "` validation data")

    x <- prof(
      read_to_mat(src, cols = feature_sel(feat_reg, predictor),
        path = data_dir, split = split, pids = pids, mat_type = mat
      )
    )

    msg("predicting")

    fun <- switch(predictor, linear = pred_lin, rf = pred_rf)
    res <- prof(fun(mod, x))

    y <- switch(target,
      class = y_class(src, hours(6L), hours(Inf), path = data_dir,
                      split = split, pids = pids),
      hybrid = y_class(src, hours(6L), hours(6L), path = data_dir,
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

train_rf <- function(x, y, is_class, id_vec, n_cores, ...) {
  
  pids <- unique(id_vec$stay_id)
  folds <- sample(1:5, length(pids), replace = T)
  
  id_vec <- id_vec[, fold := folds[which(stay_id == pids)], by = "stay_id"]
  id_vec <- id_vec[, fold := as.integer(fold == 5)]
  
  opt_ope <- Inf
  for (mns in c(10, 20, 50, 100, 500)) {
    
    mod_cand <- ranger::ranger(
      y = y, x = x, probability = is_class, min.node.size = mns,
      importance = "impurity", num.threads = n_cores, 
      case.weights = id_vec[["folds"]], holdout = T,
      ...
    )
    
    if (mod_cand$prediction.error < opt_ope) {
      
      opt_ope <- mod_cand$prediction.error
      mod <- mod_cand
      
    }
    
  }
  
  mod
  
}

train_lin <- function(x, y, is_class, id_vec, n_cores, ...) {
  
  pids <- unique(id_vec$stay_id)
  folds <- sample(1:5, length(pids), replace = T)
  
  id_vec <- id_vec[, fold := folds[which(stay_id == pids)], by = "stay_id"]
  
  biglasso::cv.biglasso(
    x, y, family = ifelse(is_class, "binomial", "gaussian"),
    lambda = c(0.0036, 0.0031), ncores = n_cores, cv.ind = id_vec[["fold"]], ...
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
