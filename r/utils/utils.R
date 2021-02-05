
n_cores <- function() {

  res <- Sys.getenv("LSB_DJOB_NUMPROC", unset = parallel::detectCores() / 2L)

  message("using ", res, " cores")

  as.integer(res)
}

cfg_path <- function(file) {
  file.path(here::here("r", "config"), file)
}

is_lsf <- function() !is.na(Sys.getenv("LSB_JOBID", unset = NA))
