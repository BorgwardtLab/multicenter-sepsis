#!/usr/bin/env Rscript

library(optparse)
library(here)
library(ricu)

source(here("r", "config.R"))
source(here("r", "utils.R"))

option_list <- list(
  make_option(c("-s", "--source"), type = "character", default = "mimic_demo",
              help = "data source name (e.g. \"mimic\")",
              metavar = "source", dest = "src"),
  make_option(c("-d", "--dir"), type = "character",
              default = "datasets",
              help = "output directory [default = \"%default\"]",
              metavar = "dir", dest = "path")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

demo <- c("mimic_demo", "eicu_demo")
prod <- c("mimic", "eicu", "hirid")
all  <- c(demo, prod)

data_opts <- c(all, "demo", "prod", "all", "challenge")

if (!(length(opt$src) == 1L && opt$src %in% data_opts)) {
  cat("\nSelect a data source among the following options:\n  ",
      paste0("\"", data_opts, "\"", collapse = ", "), "\n\n")
  print_help(opt_parser)
  q("no", status = 1, runLast = FALSE)
}

if (!dir.exists(opt$path)) {
  cat("\nThe output directory must be a valid directory. You requested\n  ",
      opt$path, "\n\n")
  print_help(opt_parser)
  q("no", status = 1, runLast = FALSE)
}

setwd(opt$path)

if (identical(opt$src, "all")) {
  sources <- all
} else if (identical(opt$src, "demo")) {
  sources <- demo
} else if (identical(opt$src, "prod")) {
  sources <- prod
} else {
  sources <- opt$src
}

for (src in sources) {

  message("dumping `", src, "`")

  dump_dataset(source = src, dir = ".")

  if (!dir.exists("shared")) {
    dir.create("shared")
  }

  zip_file <- file.path("shared", paste0(src, ".zip"))

  if (file.exists(zip_file)) {
    unlink(zip_file)
  }

  zip(zip_file, src, flags = "-qr9X")
}
