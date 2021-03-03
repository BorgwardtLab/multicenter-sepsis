
pkgs <- c("here", "arrow", "bigmemory", "jsonlite", "data.table", "readr",
          "optparse", "assertthat", "cli", "ricu", "memuse", "dplyr")

if (!all(vapply(pkgs, requireNamespace, logical(1L)))) {
  stop("Packages {pkgs} are required in order to proceed.")
  if (!interactive()) q("no", status = 1, runLast = FALSE)
}

library(ricu)
library(assertthat)
