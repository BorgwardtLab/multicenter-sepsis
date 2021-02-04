
pkgs <- c("here", "arrow", "bigmemory", "jsonlite", "data.table", "readr",
          "optparse")

stopifnot(all(vapply(pkgs, requireNamespace, logical(1L))))

library(ricu)
