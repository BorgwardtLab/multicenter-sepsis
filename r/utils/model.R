
fit_predict <- function(train_src = "mimic_demo", test_src = train_src,
                        feat_reg = "_(locf|hours|derived)$",
                        predictor = c("linear", "rf"),
                        target = c("class", "hybrid", "reg"),
                        split = "split_0", data_dir = data_path("mm"),
                        res_dir = data_path("res"), ...) {

  target <- match.arg(target)
  predictor <- match.arg(predictor)
  mat_type <- switch(predictor, rf = "mem", linear = "big")
  is_class <- !identical(target, "reg")
  is_rf <- identical(predictor, "rf")

  job <- file.path(res_dir, paste0("model_", jobname()))

  y <- switch(target,
    class = y_class(train_src, hours(6L), hours(Inf), path = data_dir,
                    split = split),
    hybrid = y_class(train_src, hours(6L), hours(6L), path = data_dir,
                     split = split),
    reg = y_reg(train_src, path = data_dir, split = split)
  )

  x <- read_to_mat(train_src, cols = feature_sel(feat_reg, predictor),
    path = data_dir, split = split, mat_type = mat_type
  )

  if (is_rf) {

    nde <- c(10, 20, 50, 100)
    mod <- vector("list", length(nde))

    for (i in seq_along(mod)) {
      mod[[i]] <- ranger::ranger(
        y = y, x = x, probability = is_class, min.node.size = nde[i],
        num.threads = n_cores(), ...
      )
    }

    mod <- mod[[
      which.min(vapply(mod, `[[`, numeric(1L), "prediction.error"))
    ]]

  } else {

    mod <- biglasso::cv.biglasso(
      x, y, family = ifelse(is_class, "binomial", "gaussian"), nfolds = 3L,
      nlambda = 5L, ncores = n_cores(), ...
    )
  }

  rm(x, y)

  qs::qsave(mod, paste0(job, ".qs"))

  pids <- coh_split(test_src, "validation", split)

  x <- read_to_mat(test_src, cols = feature_sel(feat_reg, predictor),
    path = data_dir, split = split, pids = pids, mat_type = mat_type
  )

  pred <- predict(mod, x, type = "response")

  if (is_rf) {

    pred <- pred$predictions

    if (is_class) {
      pred <- pred[, 2L]
    }

  } else {

    pred <- pred[, 1L]
  }

  y <- switch(target,
    class = y_class(test_src, hours(6L), hours(Inf), path = data_dir,
                    split = split, pids = pids),
    hybrid = y_class(test_src, hours(6L), hours(6L), path = data_dir,
                     split = split, pids = pids),
    reg = y_reg(test_src, path = data_dir, split = split, pids = pids)
  )

  pids <- read_to_df(test_src, data_dir, cols = c("stay_id", "stay_time"),
                     norm_cols = NULL, split = split, pids = pids)
  pids <- pids[, stay_time := as.double(stay_time)]

  y <- split(y, pids[["stay_id"]])
  pred <- split(pred, pids[["stay_id"]])

  pids <- split(pids, by = "stay_id", keep.by = FALSE)
  pids <- lapply(pids, `[[`, "stay_time")

  res <- list(
    model = paste(predictor, target, feat_reg, sep = "::"),
    dataset_train = train_src,
    dataset_eval = test_src,
    split = split,
    labels = y,
    scores = pred,
    times = pids,
    ids = names(pids)
  )

  jsonlite::write_json(res, paste0(job, ".json"))

  invisible(NULL)
}
