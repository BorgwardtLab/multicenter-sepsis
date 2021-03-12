
n_cores <- function() {
  as.integer(
    Sys.getenv("LSB_DJOB_NUMPROC", unset = parallel::detectCores() / 2L)
  )
}

jobid <- function() {
  Sys.getenv("LSB_JOBID", unset = as.numeric(Sys.time()))
}

cfg_path <- function(file) file.path(root_dir, "config", file)

coh_split <- function(src, set = c("train", "validation", "test"),
                      split = paste0("split_", 0:4)) {

  set <- match.arg(set)
  split <- match.arg(split)

  res <- jsonlite::read_json(
    file.path(cfg_path("splits"), paste0("splits_", src, ".json")),
    simplifyVector = TRUE, flatten = TRUE
  )

  if (identical(set, "test")) {
    res[[set]][[split]]
  } else {
    res[["dev"]][[split]][[set]]
  }
}

is_lsf <- function() !is.na(Sys.getenv("LSB_JOBID", unset = NA))

data_path <- function(type = "export") {

  dir <- file.path(root_dir, paste("data", type, sep = "-"))

  assert(dir.exists(dir))

  dir
}

feature_set <- function(name = c("full-small", "full-large",
                                 "physionet-small", "physionet-large")) {

  res <- jsonlite::read_json(paste0(cfg_path("features"), ".json"),
                             simplifyVector = TRUE, flatten = TRUE)

  name <- strsplit(match.arg(name), "-")[[1L]]

  res[[name[1L]]][[name[2L]]][["columns"]]
}

norm_sel <- function(x) {
  setdiff(
    grep("_(indicator|count)$", x, invert = TRUE, value = TRUE),
    c("stay_id", "stay_time", "female")
  )
}

feature_sel <- function(regex, predictor = "linear",
                        feats = feature_set("full-large"),
                        var_cfg = cfg_path("variables.json")) {

  sta <- read_var_json(var_cfg)
  sta <- sta[is_true(sta$category == "static"), "name"]

  res <- grep(regex, feats, value = TRUE)
  res <- c(res, sta)

  if (!identical(predictor, "linear")) {
    res <- c(res, grep("_(indicator|count)", feats, value = TRUE))
  }

  unique(res)
}

zscore <- function(x, mean, std) (x - mean) / std

zero_impute <- function(x) replace(x, is.na(x), 0)

read_colstats <- function(source, split = paste0("split_", 0:4), cols = NULL) {

  file <- paste0("normalizer_", source, "_rep",
                 sub("^split", "", match.arg(split)), ".json")

  res <- jsonlite::read_json(cfg_path(file.path("normalizer", file)))
  mis <- lapply(res, vapply, is.null, logical(1L))
  res <- Map(`[<-`, res, mis, NA_real_)
  res <- lapply(res, unlist, recursive = FALSE)

  cmp <- Reduce(union, lapply(res, names))

  if (is.null(cols)) {
    cols <- cmp
  }

  assert_that(is.character(cols), all(cols %in% cmp))

  vapply(res, `[`, numeric(length(cols)), cols)
}

read_lambda <- function(source, split = paste0("split_", 0:4)) {

  file <- paste0("lambda_", source, "_rep",
                 sub("^split", "", match.arg(split)), ".json")

  jsonlite::read_json(cfg_path(file.path("lambdas", file)))$lam
}

y_class <- function(source, start_offset = hours(6L),
                    end_offset = hours(Inf), ...) {

  dat <- read_to_df(source, cols = c("stay_id", "stay_time", "sep3"),
                    norm_cols = NULL, ...)
  sep <- dat[sep3 == 1L, head(.SD, n = 1L), by = "stay_id"]

  units(start_offset) <- units(interval(dat))
  units(end_offset) <- units(interval(dat))

  win <- merge(sep,
    dat[, list(in_time = min(stay_time), out_time = max(stay_time)),
        by = "stay_id"],
    all.x = TRUE
  )

  win <- win[, c("start_time", "end_time") := list(
    pmax(stay_time - start_offset, in_time),
    pmin(stay_time + end_offset, out_time)
  )]

  sep <- expand(win, start_var = "start_time", end_var = "end_time")
  sep <- sep[, sep3 := TRUE]

  dat <- merge(dat[, sep3 := NULL], sep, all.x = TRUE)

  is_true(dat[["sep3"]])
}

y_reg <- function(source, split = "split_0", ...) {

  dat <- read_to_df(source,
    cols = c("stay_id", "stay_time", "sep3", "utility"),
    norm_cols = NULL, ..., split = split
  )

  lmb <- read_lambda(source, split)
  dat <- dat[, is_case := any(sep3 == 1L), by = "stay_id"]
  dat <- dat[, lambda := data.table::fifelse(is_case, lmb, 1)]

  dat[["utility"]] * dat[["lambda"]]
}

delayedAssign("root_dir", here::here())
