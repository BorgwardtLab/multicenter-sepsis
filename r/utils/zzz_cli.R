
src_opt <- optparse::make_option(
  c("-s", "--source"),
  type = "character",
  default = "mimic_demo",
  help = "data source name (e.g. \"mimic\")",
  metavar = "source",
  dest = "src"
)

out_dir <- optparse::make_option(
  c("-d", "--dir"),
  type = "character",
  default = "data",
  help = "output directory [default = \"%default\"]",
  metavar = "dir",
  dest = "path"
)

parse_args <- function(...) {
  parser <- optparse::OptionParser(option_list = list(...))
  res <- optparse::parse_args(parser)
  attr(res, "parser") <- parser
  res
}

check_args <- function(check, parser, ...) {

  if (isTRUE(check)) {
    return(invisible(TRUE))
  }

  cat(...)
  print_help(parser)
  q("no", status = 1, runLast = FALSE)
}

check_src <- function(opt, extra = NULL) {

  demo <- c("mimic_demo", "eicu_demo")
  prod <- c("mimic", "eicu", "hirid", "aumc")
  all  <- c(demo, prod)

  data_opts <- c(all, "demo", "prod", "all", extra)

  if (is.list(opt)) {

    check_args(
      length(opt$src) == 1L && opt$src %in% data_opts,
      attr(opt, "parser"),
      "\nSelect a data source among the following options:\n  ",
      paste0("\"", data_opts, "\"", collapse = ", "), "\n\n"
    )

    opt <- opt$src
  }

  if (identical(opt, "all")) {
    all
  } else if (identical(opt, "demo")) {
    demo
  } else if (identical(opt, "prod")) {
    prod
  } else {
    opt
  }
}
