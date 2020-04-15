#!/usr/bin/env Rscript

library(optparse)

get_wd <- function() {
  args <- commandArgs()
  this_dir <- dirname(
    regmatches(args, regexpr("(?<=^--file=).+", args, perl = TRUE))
  )
  stopifnot(length(this_dir) == 1L)
  this_dir
}

wd <- get_wd()

option_list <- list(
  make_option(c("-s", "--source"), type = "character", default = "mimic_demo",
              help = "data source name (e.g. \"mimic\")",
              metavar = "source", dest = "src"),
  make_option(c("-d", "--dir"), type = "character",
              default = file.path(wd, "..", "datasets"),
              help = "output directory [default = \"%default\"]",
              metavar = "dir", dest = "path")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

data_sources <- c("mimic", "mimic_demo", "eicu", "eicu_demo", "hirid")

if (!(length(opt$src) == 1L && opt$src %in% data_sources)) {
  cat("\nSelect a data source among the following options:\n  ",
      paste0("\"", data_sources, "\"", collapse = ", "), "\n\n")
  print_help(opt_parser)
  q("no", status = 1, runLast = FALSE)
}

if (!dir.exists(opt$path)) {
  cat("\nThe output directory must be a valid directory. You requested\n  ",
      opt$path, "\n\n")
  print_help(opt_parser)
  q("no", status = 1, runLast = FALSE)
}

library(ricu)
source(file.path(wd, "utils.R"))

dump_dataset(source = opt$src, dir = opt$path)
