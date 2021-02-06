
pkgs <- c("here", "arrow", "bigmemory", "jsonlite", "data.table", "readr",
          "optparse", "assertthat", "cli", "ricu")

assert_that(all(vapply(pkgs, requireNamespace, logical(1L))),
            msg = "Packages {pkgs} are required in order to proceed.")

library(ricu)
library(assertthat)
