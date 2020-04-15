#!/usr/bin/env Rscript

library(optparse)
library(ricu)

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

  dat <- load_dictionary(source, challenge_map[[2L]], id_type = "icustay")
  ids <- unique(dat[age >= 14, id(dat), with = FALSE])
  dat <- merge(dat, ids, all.y = TRUE)

  sep3 <- sepsis3_score(source, ids)
  sep3 <- data.table::set(sep3, j = "SepsisLabel", value = 1L)

  dat <- rename_cols(dat, challenge_map[[1L]], challenge_map[[2L]],
                     skip_absent = TRUE)

  res <- merge(dat, sep3[, c(meta_cols(sep3), "SepsisLabel"), with = FALSE],
               all.x = TRUE)
  res <- res[, c("SepsisLabel") := data.table::nafill(SepsisLabel, "locf"),
             by = c(id(res))]
  res <- res[, c("SepsisLabel") := data.table::nafill(SepsisLabel, fill = 0L)]

  wins <- stay_windows(source, id_type = "icustay", win_type = "icustay",
                       in_time = "intime", out_time = "outtime")
  res  <- res[, c("join_time") := list(get(index(res)))]

  join <- c(paste(id(res), "==", id(wins)), "join_time <= outtime")
  res <- res[wins, on = join]
  res <- rm_cols(res, c("join_time", "intime"))

  dir <- file.path(dir, source)

  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)

  write_psv(res, dir, na_rows = TRUE)
}

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

dump_dataset(source = opt$src, dir = opt$path)
