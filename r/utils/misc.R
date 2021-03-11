
n_cores <- function() {

  res <- Sys.getenv("LSB_DJOB_NUMPROC", unset = parallel::detectCores() / 2L)

  message("using ", res, " cores")

  as.integer(res)
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
    c("stay_id", "stay_time")
  )
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

  all <- Reduce(union, lapply(res, names))

  if (is.null(cols)) {
    cols <- all
  }

  assert_that(is.character(cols), all(cols %in% all))

  vapply(res, `[`, numeric(length(cols)), cols)
}

delayedAssign("root_dir", here::here())
