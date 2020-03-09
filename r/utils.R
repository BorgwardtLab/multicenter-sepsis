
challenge_map <- data.frame(rbind(
    c("HR", "heart_rate")
  , c("O2Sat", "o2_saturation")
  , c("Temp", "temperature")
  , c("SBP", "systolic_bp")
  , c("MAP", "mean_bp")
  , c("DBP", "diastolic_bp")
  , c("Resp", "respiratory_rate")
  , c("EtCO2", "et_co2")
  , c("BaseExcess", "base_excess")
  , c("HCO3", "bicarbonate")
  , c("FiO2", "fi_o2")
  , c("pH", "ph")
  , c("PaCO2", "pa_co2")
# , c("SaO2", "?")
  , c("AST", "asparate_aminotransferase")
  , c("BUN", "urea_nitrogen")
  , c("Alkalinephos", "alkaline_phosphatase")
  , c("Calcium", "calcium")
  , c("Chloride", "chloride")
  , c("Creatinine", "creatinine")
  , c("Bilirubin_direct", "bilirubin_direct")
  , c("Glucose", "glucose")
  , c("Lactate", "lactate")
  , c("Magnesium", "magnesium")
  , c("Phosphate", "phosphate")
  , c("Potassium", "potassium")
  , c("Bilirubin_total", "bilirubin_total")
  , c("TroponinI", "troponin_i")
  , c("Hct", "hematocrit")
  , c("Hgb", "hemoglobin")
  , c("PTT", "ptt")
  , c("WBC", "white_blood_cells")
  , c("Fibrinogen", "fibrinogen")
  , c("Platelets", "platelet_count")
# , c("Age", "?")
# , c("Gender", "?")
# , c("Unit1", "?")
# , c("Unit2", "?")
# , c("HospAdmTime", "?")
# , c("ICULOS", "?")
# , c("SepsisLabel", "?")
) , stringsAsFactors = FALSE)

sepsis3_score <- function(source) {

  sofa <- sofa_data(source)
  sofa <- sofa_window(sofa)
  sofa <- sofa_compute(sofa)

  suin <- si_data(source)
  suin <- si_windows(suin)

  sepsis_3(sofa, suin)
}

dump_dataset <- function(source = "mimic_demo", dir = tempdir()) {

  sep3 <- sepsis3_score(source)

  dat <- load_concepts(source, challenge_map[[2L]])
  dat <- rename_cols(dat, challenge_map[[1L]], challenge_map[[2L]])

  res <- merge(dat, sep3, all = TRUE)
  res <- data.table::set(res, j = c("SepsisLabel", "si_time"),
    value = list(ifelse(is.na(res[["si_time"]]), NA_integer_, 1L), NULL)
  )
  res <- res[, c("SepsisLabel") := data.table::nafill(SepsisLabel, "locf"),
             by = c(key(res))]

  dir <- file.path(dir, source)

  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)

  write_psv(res, dir, na_rm = TRUE)
}
