
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

dims <- list(
  train_src = "mimic",
  feat_set = c("locf"),
  predictor = c("rf"),
  target = c("class")
)

res <- array(vector("list", prod(lengths(dims))), lengths(dims), dims)
evl <- array(vector("list", prod(lengths(dims))), lengths(dims), dims)

for (train in dims$train_src) {
  for (feat in dims$feat_set) {
    for (pred in dims$predictor) {
      for (targ in dims$target) {
        message("processing ", train, " ", feat, " ", pred, " ", targ)
        tmp <- read_res(train, feat_set = feat, predictor = pred,
                        target = targ, jobid = 165674415)
        if (is.null(tmp)) next
        res[train, feat, pred, targ] <- list(tmp)
        evl[train, feat, pred, targ] <- list(patient_eval(tmp))
      }
    }
  }
}

patient_plot(evl["mimic", "locf", "rf", "class"][[1L]])

