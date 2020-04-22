#!/usr/bin/env Rscript

library(optparse)
library(ricu)
library(readr)
library(data.table)

challenge_map <- data.frame(
  rbind(
      c("HR",               "heart_rate")
    , c("O2Sat",            "o2_saturation")
    , c("Temp",             "temperature")
    , c("SBP",              "systolic_bp")
    , c("MAP",              "mean_bp")
    , c("DBP",              "diastolic_bp")
    , c("Resp",             "respiratory_rate")
    , c("EtCO2",            "et_co2")
    , c("BaseExcess",       "base_excess")
    , c("HCO3",             "bicarbonate")
    , c("FiO2",             "fi_o2")
    , c("pH",               "ph")
    , c("PaCO2",            "pa_co2")
  # , c("SaO2",             "?")
    , c("AST",              "asparate_aminotransferase")
    , c("BUN",              "urea_nitrogen")
    , c("Alkalinephos",     "alkaline_phosphatase")
    , c("Calcium",          "calcium")
    , c("Chloride",         "chloride")
    , c("Creatinine",       "creatinine")
    , c("Bilirubin_direct", "bilirubin_direct")
    , c("Glucose",          "glucose")
    , c("Lactate",          "lactate")
    , c("Magnesium",        "magnesium")
    , c("Phosphate",        "phosphate")
    , c("Potassium",        "potassium")
    , c("Bilirubin_total",  "bilirubin_total")
    , c("TroponinI",        "troponin_i")
    , c("Hct",              "hematocrit")
    , c("Hgb",              "hemoglobin")
    , c("PTT",              "ptt")
    , c("WBC",              "white_blood_cells")
    , c("Fibrinogen",       "fibrinogen")
    , c("Platelets",        "platelet_count")
    , c("Age",              "age")
    , c("Gender",           "sex")
  # , c("Unit1",            "?")
  # , c("Unit2",            "?")
  # , c("HospAdmTime",      "?")
  # , c("ICULOS",           "?")
  # , c("SepsisLabel",      "?")
  ),
  stringsAsFactors = FALSE
)

challenge_spec <- cols(
  HR = col_double(),
  O2Sat = col_double(),
  Temp = col_double(),
  SBP = col_double(),
  MAP = col_double(),
  DBP = col_double(),
  Resp = col_double(),
  EtCO2 = col_double(),
  BaseExcess = col_double(),
  HCO3 = col_double(),
  FiO2 = col_double(),
  pH = col_double(),
  PaCO2 = col_double(),
  SaO2 = col_double(),
  AST = col_double(),
  BUN = col_double(),
  Alkalinephos = col_double(),
  Calcium = col_double(),
  Chloride = col_double(),
  Creatinine = col_double(),
  Bilirubin_direct = col_double(),
  Glucose = col_double(),
  Lactate = col_double(),
  Magnesium = col_double(),
  Phosphate = col_double(),
  Potassium = col_double(),
  Bilirubin_total = col_double(),
  TroponinI = col_double(),
  Hct = col_double(),
  Hgb = col_double(),
  PTT = col_double(),
  WBC = col_double(),
  Fibrinogen = col_double(),
  Platelets = col_double(),
  Age = col_integer(),
  Gender = col_integer(),
  Unit1 = col_integer(),
  Unit2 = col_integer(),
  HospAdmTime = col_double(),
  ICULOS = col_integer(),
  SepsisLabel = col_logical()
)

sepsis3_score <- function(source, pids = NULL) {

  sofa_score <- sofa(source, id_type = "icustay", patient_ids = pids)

  if (grepl("eicu", source)) {

    susp_infec <- si(source, abx_min_count = 2L, positive_cultures = TRUE,
                     id_type = "icustay", patient_ids = pids, mode = "or")

  } else if (identical(source, "hirid")) {

    susp_infec <- si(source, abx_min_count = 2L, id_type = "icustay",
                     patient_ids = pids, mode = "or")

  } else {

    susp_infec <- si(source, id_type = "icustay", patient_ids = pids)
  }

  sepsis_3(sofa_score, susp_infec)
}

dump_dataset <- function(source = "mimic_demo", dir = tempdir()) {

  if (identical(source, "challenge")) {

    data_dir <- file.path(dir, "training_setB")

    if (!dir.exists(data_dir)) {
      stop("need directory ", data_dir, " to continue")
    }

    dat <- read_psv(data_dir, col_spec = challenge_spec, id_col = "ID")

    dat <- dat[, Gender := fifelse(Gender == 0L, "Female", "Male")]
    dat <- dat[, O2Sat := rowMeans(.SD, na.rm=TRUE),
               .SDcols = c("O2Sat", "SaO2")]
    dat <- dat[, ICULOS := as.difftime(ICULOS, units = "hours")]
    dat <- as_ts_tbl(dat, "ID")

    sep <- dat[(SepsisLabel), .(ICULOS = min(ICULOS) + 6), by = "ID"]
    sep <- rename_cols(sep, "sep3_time", "ICULOS")

    dat <- rm_cols(dat, "SepsisLabel")

  } else {

    dat <- load_dictionary(source, challenge_map[[2L]], id_type = "icustay")
    ids <- unique(dat[age >= 14, id(dat), with = FALSE])
    dat <- merge(dat, ids, all.y = TRUE)

    dat <- rename_cols(dat, c("ICULOS", challenge_map[[1L]]),
                            c(index(dat), challenge_map[[2L]]),
                       skip_absent = TRUE)

    win <- stay_windows(source, id_type = "icustay", win_type = "icustay",
                        in_time = "intime", out_time = "outtime")

    dat  <- dat[, c("join_time") := list(get(index(dat)))]

    join <- c(paste(id(dat), "==", id(win)), "join_time <= outtime")
    dat <- dat[win, on = join]
    dat <- rm_cols(dat, c("join_time", "intime"))

    sep <- sepsis3_score(source, ids)
    sep  <- sep[, c("join_time") := list(get(index(sep)))]

    join <- c(paste(id(sep), "==", id(win)), "join_time >= intime",
                                             "join_time <= outtime")
    sep <- sep[win, on = join]
    sep <- rm_cols(sep, data_cols(sep))
  }

  sep <- set(sep, j = "SepsisLabel", value = 1L)

  res <- merge(dat, sep, all.x = TRUE)
  res <- res[, c("SepsisLabel") := nafill(SepsisLabel, "locf"),
             by = c(id(res))]
  res <- res[, c("SepsisLabel") := nafill(SepsisLabel, fill = 0L)]

  res <- rm_cols(res, setdiff(data_cols(res), challenge_map[[1L]]))

  miss_cols <- setdiff(challenge_map[[1L]], data_cols(res))

  if (length(miss_cols)) {
    res <- set(res, j = miss_cols, value = NA_real_)
  }

  res <- setcolorder(res, c(meta_cols(res), challenge_map[[1L]]))

  dir <- file.path(dir, source)

  if (dir.exists(dir)) unlink(dir, recursive = TRUE)

  dir.create(dir, recursive = TRUE)

  write_psv(res, dir, na_rows = TRUE)
}

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

data_opts <- c(all, "demo", "production", "all", "challenge")

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

  tar(paste0(src, ".tar.gz"), src, "gzip")
}
