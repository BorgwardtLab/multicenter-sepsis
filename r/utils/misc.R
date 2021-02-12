
n_cores <- function() {

  res <- Sys.getenv("LSB_DJOB_NUMPROC", unset = parallel::detectCores() / 2L)

  message("using ", res, " cores")

  as.integer(res)
}

cfg_path <- function(file) file.path(root_dir, "config", file)

is_lsf <- function() !is.na(Sys.getenv("LSB_JOBID", unset = NA))

data_path <- function(type = c("export", "challenge")) {

  type <- match.arg(type)

  dir <- file.path(root_dir, paste("data", type, sep = "-"))

  assert(dir.exists(dir))

  dir
}

delayedAssign("root_dir", here::here())
