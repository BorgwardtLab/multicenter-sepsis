
pkgs <- c("here", "arrow", "bigmemory", "jsonlite", "data.table", "readr",
          "optparse", "assertthat", "cli", "ricu", "memuse", "dplyr",
          "biglasso", "ranger", "qs")

if (!all(vapply(pkgs, requireNamespace, logical(1L)))) {
  stop("Packages {pkgs} are required in order to proceed.")
  if (!interactive()) q("no", status = 1, runLast = FALSE)
}

library(ricu)
library(assertthat)
library(ggplot2)

arrow::set_cpu_count(n_cores())
data.table::setDTthreads(n_cores())
