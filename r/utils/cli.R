
delayedAssign("src_opt",
  optparse::make_option(
    c("-s", "--source"),
    type = "character",
    default = if (is_lsf()) "all" else "demo",
    help = "data source name (e.g. \"mimic\")",
    metavar = "source",
    dest = "src"
  )
)

delayedAssign("out_dir",
  optparse::make_option(
    c("-d", "--dir"),
    type = "character",
    default = "data-export",
    help = "output directory [default = \"%default\"]",
    metavar = "dir",
    dest = "path"
  )
)

delayedAssign("job_index",
  optparse::make_option(
    c("-i", "--index"),
    type = "integer",
    default = 1L,
    help = "job index [default = \"%default\"]",
    metavar = "index",
    dest = "ind"
  )
)

parse_args <- function(...) {
  parser <- optparse::OptionParser(option_list = list(...))
  tryCatch(optparse::parse_args(parser), error = function(e) {
    optparse::print_help(parser)
    q("no", status = 1, runLast = FALSE)
  })
}

assert <- function(..., msg = NULL) {

  res <- see_if(..., env = parent.frame(), msg = msg)

  if (res) {
    return(invisible(TRUE))
  }

  stop(cli::pluralize(attr(res, "msg"), .envir = parent.frame()))

  if (!interactive()) {
    q("no", status = 1, runLast = FALSE)
  }
}

msg <- function(..., env = parent.frame()) {
  message(cli::pluralize(..., .envir = env))
}

check_src <- function(opt, extra = NULL) {

  demo <- c("mimic_demo", "eicu_demo")
  prod <- c("mimic", "eicu", "hirid", "aumc")
  all  <- c(demo, prod, extra)

  data_opts <- c(all, "demo", "prod", "all")

  assert(is.string(opt$src), opt$src %in% data_opts,
    msg = "\nSelect a data source among the following options:
      {data_opts}\n\n")

  if (identical(opt$src, "all")) {
    all
  } else if (identical(opt$src, "demo")) {
    demo
  } else if (identical(opt$src, "prod")) {
    prod
  } else {
    opt$src
  }
}

check_dir <- function(opt) {

  res <- opt$path

  assert(dir.exists(res),
    msg = "\nThe output directory must be an existing directory, not
      {res}.\n\n")

  res
}

check_index <- function(opt, ...) {

  arg_opts <- expand.grid(..., KEEP.OUT.ATTRS = FALSE,
                          stringsAsFactors = FALSE)

  opt <- as.integer(Sys.getenv("LSB_JOBINDEX", unset = opt$ind))

  assert(is.count(opt), opt <= nrow(arg_opts))

  as.list(arg_opts[opt, ])
}

format_unit <- function(x, units = "hours") format(`units<-`(x, units))
