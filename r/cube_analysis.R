
invisible(
  lapply(list.files(here::here("r", "utils"), full.names = TRUE), source)
)

library(ggplot2)
library(precrec)

dims <- list(
  train_src = c("mimic", "aumc"),
  feat_set = c("basic", "wav", "sig", "full"),
  predictor = c("linear", "rf"),
  target = c("class", "hybrid", "reg")
)

res <- array(vector("list", prod(lengths(dims))), lengths(dims), dims)

for (train in dims$train_src) {
  for (feat in dims$feat_set) {
    for (pred in dims$predictor) {
      for (targ in dims$target) {
        res[train, feat, pred, targ] <- list(
          read_res(train, feat_set = feat, predictor = pred, target = targ)
        )
      }
    }
  }
}

evl <- evalmod(
  scores = lapply(res["mimic", , "rf", "hybrid"], `[[`, "prediction"),
  labels = lapply(res["mimic", , "rf", "hybrid"], `[[`, "label"),
  modnames = dims$feat_set
)

autoplot(evl)
