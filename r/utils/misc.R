
n_cores <- function() {
  as.integer(
    Sys.getenv("LSB_DJOB_NUMPROC", unset = parallel::detectCores() / 2L)
  )
}

jobid <- function() {
  Sys.getenv("LSB_JOBID", unset = as.numeric(Sys.time()))
}

cfg_path <- function(...) file.path(root_dir, "config", ...)

coh_split <- function(src, set = c("train", "validation", "test"),
                      split = paste0("split_", 0:4), case_prop = NULL,
                      seed = as.integer(sub("^split_", "", split))) {

  set <- match.arg(set)
  split <- match.arg(split)

  res <- jsonlite::read_json(
    file.path(cfg_path("splits"), paste0("splits_", src, ".json")),
    simplifyVector = TRUE, flatten = TRUE
  )

  ids <- res[["total"]][["ids"]]
  lab <- res[["total"]][["labels"]]

  if (identical(set, "test")) {
    res <- res[[set]][[split]]
  } else {
    res <- res[["dev"]][[split]][[set]]
  }

  set.seed(seed)

  if (!is.null(case_prop)) {

    ctrl <- split(res, lab[match(res, ids)])
    case <- ctrl[["1"]]
    ctrl <- ctrl[["0"]]

    orig <- length(ctrl) / length(res)

    fact <- (length(case) / case_prop - length(case)) / length(ctrl)
    ctrl <- rep(sample(ctrl), ceiling(fact))[
      seq_len(ceiling(fact * length(ctrl)))
    ]

    res <- sample(c(ctrl, case))
    prp <- length(ctrl) / length(res)

    msg("resampling controls to a proportion of {format(prp,
         digits = 3)} (from {format(orig, digits = 3)})")
  }

  attr(res, "ids") <- ids
  attr(res, "lables") <- lab

  res
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
    c("stay_id", "stay_time", "female", "sep3")
  )
}

feature_sel <- function(regex, predictor = "linear",
                        feats = feature_set("full-large"),
                        var_cfg = cfg_path("variables.json")) {

  sta <- read_var_json(var_cfg)
  sta <- sub("_static$", "", sta[is_true(sta$category == "static"), "name"])

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

try_id_tbl <- function(x) {

  if ("stay_id" %in% colnames(x)) {

    if ("stay_time" %in% colnames(x)) {

      x <- data.table::set(x, j = "stay_time",
        value = as.difftime(x[["stay_time"]], units = "hours")
      )

      x <- as_ts_tbl(x, id_vars = "stay_id", index_var = "stay_time",
                     interval = hours(1L), by_ref = TRUE)

    } else {

      x <- as_id_tbl(x, id_vars = "stay_id", by_ref = TRUE)
    }
  }

  x
}

prof <- function(expr, envir = parent.frame()) {

  mem <- memuse::Sys.procmem()
  tim <- Sys.time()

  res <- eval(expr, envir = envir)

  cur <- memuse::Sys.procmem()
  cil <- cur[["peak"]] - mem[["peak"]]

  msg("    Runtime: {format(Sys.time() - tim, digits = 4)}")
  if (length(cil)) msg("    Memory ceiling increased by: {as.character(cil)}")
  msg("    Current memory usage: {as.character(cur[['size']])}")

  res
}

delayedAssign("root_dir", here::here())

preproc_version <- "001"
