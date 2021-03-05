library(ricu)
library(jsonlite)
library(stringr)
library(assertthat)
library(dplyr)
library(glmnet)
library(ranger)

root <- file.path(rprojroot::find_root(".git/index"))
utils_dir <- file.path(root, "r", "utils")
invisible(lapply(list.files(utils_dir, full.names = TRUE), source))

feature_module <- function(reg, predictor = "linear") {

  x <- feature_set("full-large")
  cols <- grep(reg, x, value = T)
  cols <- c(cols, c("age", "female", "height", "weight"))
  if (predictor == "RF")
    cols <- c(cols, grep("_(indicator|count)", x, value = T))

  unique(cols)

}

cube_pos <- function(source = "demo",
                     feat_reg = "_(locf|hours|derived)",
                     predictor = "linear",
                     target = "sep3")
{
  # load train data
  y <- read_to_vec(source, col = target)
  x <- read_to_df(source, cols = feature_module(feat_reg, predictor),
                  pids = coh_split(source, "train"))
  x <- as.matrix(replace_na(x, 0))
  is_class <- length(unique(y)) == 2L

  if (predictor == "RF") {

    min_node_grid <- c(10, 20, 50, 100)
    RFs <- lapply(min_node_grid,
      function(node.size) ranger(y = y, x = x, probability = is_class,
                                 min.node.size = node.size))

    forest.idx <- which.min(sapply(RFs, function(forest)
      forest[["prediction.error"]]))
    opt.pred <- RFs[[forest.idx]]

  } else {

    opt.pred <- cv.glmnet(
      x, y, family = ifelse(is_class, "binomial", "gaussian"), nfolds = 3L,
      nlambda = 2L
    )

  }

  # load val data
  x_test <- read_to_df(source, cols = feature_module(feat_reg, predictor),
                  pids = coh_split(source, "val"))
  x_test <- as.matrix(replace_na(x_test, 0))
  # y_test <- read_to_vec(source, col = target, pids = coh_split(source, "val"))

  pred <- predict(opt.pred, x_test, type = "response")

  if (predictor == "RF") {

    res <- pred$predictions
    if (is_class) res <- res[, 2]

  } else res <- as.vector(pred)

  res

}


src <- "demo"
feat_combos <- c(
  "_(hours|locf|derived)", # basic: look-back + current + derived
  "_(hours|locf|derived|wavelet)", # basic + wavelets,
  "_(hours|locf|derived|signature)", # basic + signatures
  "_(hours|locf|derived|wavelet|signature)" # basic + wavelets
)

for (feat_reg in feat_combos) {

  for (predictor in c("linear", "RF")) {

    for (target in c("sep3", "utility")) {

      print(cube_pos(src, feat_reg, predictor, target))

    }

  }

}
